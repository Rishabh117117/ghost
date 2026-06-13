"""
engram_v6_5.py - allocation, the knee, and composition. INFERENCE-ONLY.

Three questions, pure forward passes:
 1. Does FLOOR-THEN-GREEDY allocation (sink + per-fact-span floor + greedy
    fill, equal budget) move the knee left vs v6's global top-k?
 2. Where exactly is the knee under each allocator?
    G24/G40/G48 (+G32 carried from v6) vs F16/F24/F32.
 3. Does the S32 format survive MULTIPLE entities loaded at once?
    C1/C2/C4: segments captured at per-slot RoPE offsets (i * SEG), stacked
    before the prompt, question offset after the last segment - standard
    multi-document mechanics. Report per-entity recall, co-load intrusion,
    stranger theft - gated and ungated.

Working point = smallest config with verbatim >= 80% AND no attribute below
70% (the floor bar exists so aggregates can never hide starvation again).
A1 is carried from v6 (not rerun). No training anywhere in this run.
"""
import argparse
import json
import os
import random
import time
from datetime import datetime, timezone

import torch
import torch.nn.functional as F

from ghost import MODEL_NAME, DEVICE, SEED
from bank import keep_awake
from sweep_ccat50 import start_heartbeat, STATUS
from engram_score import scored_hit, is_confabulation, qa_prompt
from engram import (cloze, load_jsonl, make_gen, eval_arm, wikitext_ppl,
                    load_base)
from engram_v3 import Addressing
from engram_v4 import smoke_base_qwen, smoke_data
from engram_v5 import eval_deep_arm, rank_map, ctx_text
from engram_deep import DeepMem, gen_mem
from engram_select import (calib_texts, calibrate_mass, select_topk,
                           storage_mb)
from engram_alloc import sentence_token_spans, floor_then_greedy, stack_segments

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- pinned constants ----------------------------------------------------------
G_KS       = (24, 40, 48)          # global allocator (G32 carried from v6)
F_KS       = (16, 24, 32)          # floor-then-greedy allocator
K_COMPOSE  = 32                    # composition uses the S32 format
COMPOSE_NS = (1, 2, 4)
N_EVAL     = 400
N_COMP     = 200                   # composition rows per Cn (budget: pinned)
N_COMP_DIS = 100
GEN_MAX_NEW = 16
K_ADDR     = 4096
KEPT_MAP_N = 5
HUB_PREFIX = "engram-v6_5"

# ---- gates & bars (pinned) -------------------------------------------------------
BAR_A0_RECALL = 0.05
BAR_E0        = 0.90
BAR_WORKING   = 0.80
BAR_ATTR_FLOOR = 0.70
BAR_C4_RATIO  = 0.90
BAR_G_STRANGER = 0.02


def run(smoke=False, only=None, out_root=None):
    t0 = time.time()
    budget = 12 * 60 if smoke else int(1.5 * 3600)
    keep_awake()
    if smoke:
        base, tok = smoke_base_qwen()
        ents, facts, qa, dis = smoke_data()
        wiki = [f"general filler sentence number {i} about ordinary things."
                for i in range(60)]
        n_eval, n_comp, n_comp_dis = 50, 20, 5
    else:
        base, tok = load_base()
        ents = load_jsonl("entities.jsonl")
        facts = load_jsonl("facts_train.jsonl")
        qa = load_jsonl("qa_eval.jsonl")
        dis = load_jsonl("distractors.jsonl")
        from generic_test import load_domain3
        wiki, _ = load_domain3()
        n_eval, n_comp, n_comp_dis = N_EVAL, N_COMP, N_COMP_DIS
    ent_by_id = {e["entity_id"]: e for e in ents}
    rng = random.Random(SEED)
    qa_eval = qa[:]
    rng.shuffle(qa_eval)
    qa_eval = qa_eval[:min(n_eval, len(qa_eval))]
    base_params = sum(p.numel() for p in base.parameters())

    addr = Addressing(ents, "entity", k=K_ADDR, n_null=0)
    os.makedirs(out_root, exist_ok=True)
    ranks = rank_map(addr)

    dm = DeepMem(base)
    cfg = base.config
    n_kv = cfg.num_key_value_heads
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    n_layers = dm.n_layers

    ctx_len = {e["entity_id"]: len(tok(ctx_text(e)).input_ids) for e in ents}
    SEG = max(ctx_len.values()) + 8          # composition slot width
    mean_ctx = sum(ctx_len.values()) / len(ctx_len)
    span_memo = {}

    def spans_of(eid):
        if eid not in span_memo:
            span_memo[eid] = sentence_token_spans(tok, ent_by_id[eid])[0]
        return span_memo[eid]

    def ap(n):
        return os.path.join(out_root, f"arm_{n}.json")

    def load_or(n):
        return json.load(open(ap(n))) if os.path.exists(ap(n)) else None

    def save_arm(d):
        json.dump(d, open(ap(d["arm"]), "w"), indent=1)

    def base_fwd(ids):
        out = base(ids)
        lg = out.logits[:, :-1].contiguous()
        lb = ids[:, 1:].contiguous()
        return out.logits, F.cross_entropy(lg.view(-1, lg.size(-1)), lb.view(-1))

    dm.clear()
    STATUS["phase"] = "base_ppl"
    base_ppl = wikitext_ppl(base_fwd, tok, wiki)
    print(f"base wikitext ppl = {base_ppl:.3f}", flush=True)

    kv_memo = {}

    def cached(eid, offset=0):
        key = (eid, offset)
        if key not in kv_memo:
            if len(kv_memo) > 30:
                kv_memo.clear()
            kv_memo[key] = dm.capture_ctx(tok, ctx_text(ent_by_id[eid]),
                                          DEVICE, offset=offset)
        return kv_memo[key]

    mass_memo = {}

    def mass_for(eid):
        if eid not in mass_memo:
            kv, T = cached(eid)
            mass_memo[eid] = calibrate_mass(
                dm, base, tok, kv, calib_texts(ent_by_id[eid]), T, DEVICE)
        return mass_memo[eid]

    def e0_mem(eid, donor=None):
        kv, _ = cached(donor if donor is not None else eid)
        dm.set_cached(kv)

    def g_mem(k):
        def arm_mem(eid, donor=None):
            e = donor if donor is not None else eid
            kv, _ = cached(e)
            sel, _, _ = select_topk(kv, mass_for(e), k)
            dm.set_cached(sel)
        return arm_mem

    def f_mem(k):
        def arm_mem(eid, donor=None):
            e = donor if donor is not None else eid
            kv, _ = cached(e)
            sel, _, _, _ = floor_then_greedy(kv, mass_for(e), k, spans_of(e))
            dm.set_cached(sel)
        return arm_mem

    def offset_of(eid):
        return cached(eid)[1]

    want = set(only) if only else None

    def wanted(n):
        return want is None or n in want

    arms = []
    mb_of = {"E0": round(storage_mb(mean_ctx, n_layers, n_kv, head_dim), 2)}
    recap, audits = {}, {}

    # ---- A0 gate ----
    if wanted("A0"):
        a = load_or("A0")
        if a is None:
            STATUS["phase"] = "A0"
            a = eval_arm("A0", make_gen(base, tok), None, qa_eval, dis, wiki,
                         tok, base_ppl, params=(base_params, 0))
            save_arm(a)
        arms.append(a)
        if not smoke and a["verbatim"] > BAR_A0_RECALL:
            json.dump({"reason": "A0 contamination", "a0": a},
                      open(os.path.join(out_root, "ABORT.json"), "w"))
            write_report(out_root, arms, a, {"smoke": smoke, "aborted": True},
                         "aborted at gate", mb_of=mb_of)
            return 3

    # ---- A1 carried from v6 (do not rerun) ----
    v6_a1 = os.path.join(HERE, "results", "engram_v6", "arm_A1.json")
    if os.path.exists(v6_a1) and not smoke:
        a1 = json.load(open(v6_a1))
        a1["carried_from"] = "engram-v6"
        arms.append(a1)

    # ---- smoke-only invariants ----
    if smoke:
        r = qa_eval[0]
        eid = r["entity_id"]
        e0_mem(eid)
        g_ref = gen_mem(base, tok, qa_prompt(r["question"]), offset_of(eid), DEVICE)
        f_mem(999)(eid)
        g_f = gen_mem(base, tok, qa_prompt(r["question"]), offset_of(eid), DEVICE)
        assert g_f == g_ref, "F selection-identity FAILED"
        kv, _ = cached(eid)
        _, kmap, _, audit = floor_then_greedy(kv, mass_for(eid), 6, spans_of(eid))
        assert all(0 in heads for layer in kmap for heads in [layer[0]]), \
            "sink (pos 0) missing from F keep-set"
        # composition C1 at slot 0 == plain S-format single segment
        seg, _ = cached(eid)
        sel, _, _ = select_topk(seg, mass_for(eid), K_COMPOSE)
        dm.set_cached(sel)
        g_s = gen_mem(base, tok, qa_prompt(r["question"]), offset_of(eid), DEVICE)
        dm.set_cached(stack_segments([sel]))
        g_c = gen_mem(base, tok, qa_prompt(r["question"]), offset_of(eid), DEVICE)
        assert g_c == g_s, "composition C1 identity FAILED"
        dm.clear()
        print("smoke invariants (F-identity, sink-in-floor, C1-identity): OK",
              flush=True)

    # ---- E0 regression gate ----
    if wanted("E0"):
        a = load_or("E0")
        if a is None:
            STATUS["phase"] = "E0"
            a = eval_deep_arm("E0", base, tok, dm, e0_mem, qa_eval, dis, wiki,
                              base_ppl, (base_params, 0), ent_by_id, ranks,
                              offset_of)
            save_arm(a)
        arms.append(a)
        if not smoke and a["verbatim"] < BAR_E0:
            write_report(out_root, arms, arms[0],
                         {"smoke": smoke, "regression_abort": True},
                         f"wall {int((time.time()-t0)/60)} min", mb_of=mb_of)
            return 4

    # ---- G32 carried from v6 ----
    v6_s32 = os.path.join(HERE, "results", "engram_v6", "arm_S32-mass.json")
    if os.path.exists(v6_s32) and not smoke:
        g32 = json.load(open(v6_s32))
        g32["arm"] = "G32"
        g32["carried_from"] = "engram-v6 S32-mass"
        arms.append(g32)
        mb_of["G32"] = round(storage_mb(32, n_layers, n_kv, head_dim), 2)
        recap["G32"] = g32.get("mass_recapture")

    # ---- allocation curves ----
    kept_maps = {}
    specs = [(f"G{k}", g_mem(k), k, "G") for k in G_KS] + \
            [(f"F{k}", f_mem(k), k, "F") for k in F_KS]
    for name, mem, k, kind in specs:
        if not wanted(name) or time.time() - t0 > budget:
            continue
        a = load_or(name)
        if a is None:
            STATUS["phase"] = name
            a = eval_deep_arm(name, base, tok, dm, mem, qa_eval, dis, wiki,
                              base_ppl, (base_params, 0), ent_by_id, ranks,
                              offset_of)
            rs, n = 0.0, 0
            for r in qa_eval[:KEPT_MAP_N]:
                eid = r["entity_id"]
                kv, _ = cached(eid)
                if kind == "G":
                    sel, kmap, rc = select_topk(kv, mass_for(eid), k)
                    audit = None
                else:
                    sel, kmap, rc, audit = floor_then_greedy(
                        kv, mass_for(eid), k, spans_of(eid))
                rs += rc
                n += 1
                kept_maps.setdefault(name, {})[str(eid)] = kmap
            a["mass_recapture"] = round(rs / max(n, 1), 4)
            if kind == "F":
                a["budget_audit"] = audit
            save_arm(a)
        arms.append(a)
        mb_of[name] = round(storage_mb(k, n_layers, n_kv, head_dim), 2)
        recap[name] = a.get("mass_recapture")
        if a.get("budget_audit"):
            audits[name] = a["budget_audit"]
    if kept_maps:
        json.dump(kept_maps, open(os.path.join(out_root, "kept_maps.json"), "w"))

    # ---- composition ----
    comp = load_or("composition")
    if comp is None and time.time() - t0 <= budget:
        STATUS["phase"] = "composition"
        crng = random.Random(SEED + 13)
        eids_all = sorted(ent_by_id)
        comp_rows = qa_eval[:n_comp]
        comp_dis = dis[:n_comp_dis]
        comp = {"arm": "composition", "seg": SEG, "k": K_COMPOSE, "rows": {}}

        def seg_for(eid, slot):
            kv, _ = cached(eid, offset=slot * SEG)
            sel, _, _ = select_topk(kv, mass_for(eid), K_COMPOSE)
            return sel

        for n_load in COMPOSE_NS:
            shuffled = eids_all[:]
            crng.shuffle(shuffled)
            group_of = {}
            for i in range(0, len(shuffled), n_load):
                g = shuffled[i:i + n_load]
                for eid in g:
                    group_of[eid] = g
            hits = intr = 0
            for r in comp_rows:
                eid = r["entity_id"]
                group = group_of[eid]
                dm.set_cached(stack_segments(
                    [seg_for(e, i) for i, e in enumerate(group)]))
                g_out = gen_mem(base, tok, qa_prompt(r["question"]),
                                len(group) * SEG, DEVICE, max_new=GEN_MAX_NEW)
                hits += int(scored_hit(g_out, r["answer"]))
                intr += int(any(scored_hit(g_out, str(ent_by_id[e][r["attr"]]))
                                for e in group if e != eid))
            # stranger theft: distractor question, n entities loaded
            theft = hedge = 0
            for j, r in enumerate(comp_dis):
                group = group_of[shuffled[(j * n_load) % len(shuffled)]]
                dm.set_cached(stack_segments(
                    [seg_for(e, i) for i, e in enumerate(group)]))
                g_out = gen_mem(base, tok, qa_prompt(r["question"]),
                                len(group) * SEG, DEVICE, max_new=GEN_MAX_NEW)
                theft += int(any(scored_hit(g_out, str(ent_by_id[e][r["attr"]]))
                                 for e in group))
                hedge += int(not is_confabulation(g_out))
            dm.clear()
            # gated stranger: gate loads nothing -> model alone
            g_theft = g_hedge = 0
            for j, r in enumerate(comp_dis):
                group = group_of[shuffled[(j * n_load) % len(shuffled)]]
                g_out = gen_mem(base, tok, qa_prompt(r["question"]), 0, DEVICE,
                                max_new=GEN_MAX_NEW)
                g_theft += int(any(scored_hit(g_out, str(ent_by_id[e][r["attr"]]))
                                   for e in group))
                g_hedge += int(not is_confabulation(g_out))
            comp["rows"][f"C{n_load}"] = {
                "recall": round(hits / max(len(comp_rows), 1), 4),
                "intrusion": round(intr / max(len(comp_rows), 1), 4),
                "stranger_theft": round(theft / max(len(comp_dis), 1), 4),
                "stranger_hedge": round(hedge / max(len(comp_dis), 1), 4),
                "gated_stranger_theft": round(g_theft / max(len(comp_dis), 1), 4),
                "gated_model_hedge": round(g_hedge / max(len(comp_dis), 1), 4)}
            print(f"C{n_load}: {comp['rows'][f'C{n_load}']}", flush=True)
        save_arm(comp)

    cost = f"wall {int((time.time() - t0) / 60)} min"
    a0 = next((a for a in arms if a["arm"] == "A0"), {"verbatim": 0.0})
    meta = {"smoke": smoke, "model": "tiny-random-qwen3" if smoke else MODEL_NAME,
            "device": DEVICE, "seed": SEED, "seg": SEG,
            "g_ks": list(G_KS), "f_ks": list(F_KS), "k_compose": K_COMPOSE,
            "n_comp_rows": n_comp, "inference_only": True}
    write_report(out_root, arms, a0, meta, cost, mb_of=mb_of, recap=recap,
                 audits=audits, comp=comp)
    return 0


# ============================================================ report ==========
def write_report(out_root, arms, a0, meta, cost, mb_of=None, recap=None,
                 audits=None, comp=None):
    e0 = next((a for a in arms if a["arm"] == "E0"), None)
    gate_a0 = a0["verbatim"] <= BAR_A0_RECALL
    e0_ok = e0 is not None and e0["verbatim"] >= BAR_E0
    sel_arms = [a for a in arms if a["arm"][:1] in ("G", "F") and
                a["arm"] not in ("A0", "A1")]

    def attr_floor(a):
        pa = a.get("per_attr_verbatim") or {}
        return min(pa.values()) if pa else 0.0

    working = None
    for a in sorted(sel_arms, key=lambda a: (mb_of or {}).get(a["arm"], 1e9)):
        if a["verbatim"] >= BAR_WORKING and attr_floor(a) >= BAR_ATTR_FLOOR:
            working = a
            break
    if not gate_a0:
        verdict = f"**ABORT — contamination (A0 {a0['verbatim']:.1%})**"
    elif e0 is not None and not e0_ok:
        verdict = f"**REGRESSION ABORT — E0 {e0['verbatim']:.1%} < {BAR_E0:.0%}**"
    elif working is not None:
        verdict = (f"**ENGRAM-V6.5: working point {working['arm']} — "
                   f"{working['verbatim']:.1%} verbatim, attr floor "
                   f"{attr_floor(working):.1%}, "
                   f"{(mb_of or {}).get(working['arm'], 0):.2f} MB/entity**")
    else:
        best = max(sel_arms, key=lambda a: (a["verbatim"], attr_floor(a)),
                   default=None)
        verdict = (f"**ENGRAM-V6.5: no working point — best "
                   f"{best['arm'] if best else '-'} "
                   f"{best['verbatim'] if best else 0:.1%} "
                   f"(attr floor {attr_floor(best) if best else 0:.1%})**")
    if meta.get("smoke"):
        verdict = f"SMOKE RUN — plumbing only, numbers meaningless. {verdict}"

    L = ["# engram-v6.5: allocation, the knee, and composition\n", f"{verdict}\n",
         f"_Generated {datetime.now(timezone.utc).isoformat()} | meta: {json.dumps(meta)}_\n",
         "Part 0: ANALYSIS_V6_SINKS.md committed to main (pos-0 kept 273-281/288 "
         "by mass vs 124/288 by norm; starvation table) - every number "
         "re-derived from v6 kept_maps.json.\n",
         "## Both curves (equal budget at equal k)\n",
         "| arm | MB/entity | verbatim | QA | attr floor | theft | drift_loaded | recapture |",
         "|---|---|---|---|---|---|---|---|"]
    for a in arms:
        if a["arm"] in ("A0", "composition"):
            continue
        mb = (mb_of or {}).get(a["arm"])
        rc = (recap or {}).get(a["arm"])
        carried = " (carried)" if a.get("carried_from") else ""
        L.append(f"| {a['arm']}{carried} | " + (f"{mb:.2f}" if mb else "-")
                 + f" | {a['verbatim']:.1%} | {a['qa']:.1%} | "
                 + (f"{attr_floor(a):.1%}" if a.get("per_attr_verbatim") else "-")
                 + " | " + (f"{a['theft']:.1%}" if "theft" in a else "-")
                 + " | " + (f"{a['drift_loaded']:+.1%}" if "drift_loaded" in a else "-")
                 + " | " + (f"{rc:.1%}" if rc is not None else "-") + " |")
    L.append(f"\nWorking point = smallest config with verbatim >= "
             f"{BAR_WORKING:.0%} AND every attribute >= {BAR_ATTR_FLOOR:.0%}. "
             f"A0 {a0['verbatim']:.1%}.\n")

    L.append("## Per-attribute verbatim (the starvation check)\n")
    attrs = sorted({k for a in sel_arms for k in (a.get("per_attr_verbatim") or {})})
    if attrs:
        L.append("| arm | " + " | ".join(attrs) + " |")
        L.append("|" + "---|" * (len(attrs) + 1))
        for a in sel_arms:
            pa = a.get("per_attr_verbatim") or {}
            L.append(f"| {a['arm']} | " +
                     " | ".join(f"{pa.get(k, 0):.0%}" for k in attrs) + " |")
    if audits:
        L.append("\n## Budget-waste audit (cost of the floor guarantee)\n")
        L.append("| arm | floor/lh | duplicates greedy | forced picks |")
        L.append("|---|---|---|---|")
        for arm, ad in audits.items():
            L.append(f"| {arm} | {ad['floor_per_lh']} | "
                     f"{ad['floor_dup_of_greedy_per_lh']} | "
                     f"{ad['floor_forced_per_lh']} |")
    if comp:
        L.append("\n## Composition (S32 segments, sequential RoPE slots)\n")
        L.append("| load | recall | intrusion | stranger theft | stranger hedge | gated theft | gated model hedge |")
        L.append("|---|---|---|---|---|---|---|")
        for cn, r in comp["rows"].items():
            L.append(f"| {cn} | {r['recall']:.1%} | {r['intrusion']:.1%} | "
                     f"{r['stranger_theft']:.1%} | {r['stranger_hedge']:.1%} | "
                     f"{r['gated_stranger_theft']:.1%} | {r['gated_model_hedge']:.1%} |")
        c1 = comp["rows"].get("C1", {}).get("recall", 0)
        c4 = comp["rows"].get("C4", {}).get("recall", 0)
        if c1:
            ok = c4 >= BAR_C4_RATIO * c1
            L.append(f"\n- C4/C1 = {c4 / c1:.2f} (bar >= {BAR_C4_RATIO}) -> "
                     f"{'PASS' if ok else 'FAIL'}. Gated Cn collapses to C1 + "
                     f"abstention on strangers by construction (id-gate).")
    L.append(f"\n_Cost: {cost}_\n\nAll raw numbers: `results.json`\n")

    md = os.path.join(out_root, "ENGRAM_V6_5.md") if meta.get("smoke") \
        else os.path.join(HERE, "ENGRAM_V6_5.md")
    open(md, "w").write("\n".join(L))
    res = {"meta": meta, "arms": arms, "a0": a0, "mb_of": mb_of,
           "mass_recapture": recap, "budget_audits": audits,
           "composition": comp, "verdict": verdict,
           "working_point": working["arm"] if working else None,
           "bars": {"a0": BAR_A0_RECALL, "e0": BAR_E0, "working": BAR_WORKING,
                    "attr_floor": BAR_ATTR_FLOOR, "c4_ratio": BAR_C4_RATIO,
                    "gated_stranger": BAR_G_STRANGER}}
    rp = os.path.join(out_root, "results.json") if meta.get("smoke") \
        else os.path.join(HERE, "results.json")
    json.dump(res, open(rp, "w"), indent=1)
    print(verdict, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--arms", default="")
    args = ap.parse_args()
    out_root = os.path.join(HERE, "results",
                            "engram_v6_5_smoke" if args.smoke else "engram_v6_5")
    os.makedirs(out_root, exist_ok=True)
    status_dir = out_root if args.smoke else os.path.join(HERE, "status")
    os.makedirs(status_dir, exist_ok=True)
    start_heartbeat(status_dir)
    only = [s.strip() for s in args.arms.split(",") if s.strip()] or None
    raise SystemExit(run(smoke=args.smoke, only=only, out_root=out_root))


if __name__ == "__main__":
    main()
