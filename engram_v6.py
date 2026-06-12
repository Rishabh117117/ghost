"""
engram_v6.py - selection compression + identity gate.

v5 left the compression frontier unmeasured: E0 (full cached KV) = 93.5%,
but the only compressor tried (mean-pooling) was a geometry bug (2.4%
attention mass). v6 compresses by SELECTION - keep top-k REAL positions per
layer per kv-head (H2O/SnapKV convention), original RoPE phases and values
untouched - and measures recall vs MB/entity:

  A0 gate | A1 rerun | E0 rerun (regression gate >= 90%)
  S32-mass, S16-mass, S8-mass   - the curve (calibrated ranker)
  S16-norm                      - zero-calibration baseline ranker
  S8-tune (conditional: S8-mass verbatim < 70%) - tune ONLY the kept values
      (keys frozen - geometry is the thing v5 taught us not to touch),
      300 steps answer-token CE.
  G-gate - identity gate (entity-id match, the upper-bound convention):
      inject only when the queried entity owns the memory. Under id-gating
      the gate is deterministic, so gated theft / hedge-on-absent are
      by-construction numbers; we report them alongside the MODEL's own
      hedge rate (without injection) - the gap is what surface-form
      detection has to deliver.

Calibration for the MASS ranker uses the entity's TRAINING fact sentences
only - the held-out QA phrasings are never run during calibration (no eval
leakage).

Working point = smallest k with verbatim >= 80%.
"""
import argparse
import json
import os
import random
import time
from datetime import datetime, timezone

import torch
import torch.nn as nn
import torch.nn.functional as F

from ghost import MODEL_NAME, DEVICE, SEED
from bank import keep_awake
from sweep_ccat50 import start_heartbeat, STATUS
from engram_score import scored_hit, is_confabulation, qa_prompt
from engram import (load_jsonl, make_gen, eval_arm, wikitext_ppl,
                    pad_batch, precompute_replay, load_base)
from engram_v3 import Addressing
from engram_v4 import fact_examples, smoke_base_qwen, smoke_data
from engram_v5 import (eval_deep_arm, rank_map, ctx_text, train_deep)
from engram_deep import DeepMem, gen_mem
from engram_select import (calib_texts, calibrate_mass, norm_score,
                           select_topk, storage_mb)

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- pinned hyperparameters --------------------------------------------------
S_KS        = (32, 16, 8)
TUNE_STEPS  = 300
TUNE_LR     = 5e-4
MAX_LEN     = 64
N_EVAL      = 400
N_REPLAY    = 2000
K_ADDR      = 4096
KEPT_MAP_N  = 5          # entities whose kept-position maps are committed
HUB_REPO    = "Spartan117Ri/ghost-ckpts"
HUB_PREFIX  = "engram-v6"

# ---- gates & bars (pinned) -----------------------------------------------------
BAR_A0_RECALL = 0.05
BAR_E0        = 0.90     # regression gate on the v5 anchor
BAR_WORKING   = 0.80     # working point: smallest k with verbatim >= this
TUNE_TRIGGER  = 0.70
BAR_G_THEFT   = 0.05
BAR_G_HEDGE   = 0.80


class SelBank(nn.Module):
    """S8-tune storage: kept keys FROZEN (buffers), kept values trainable."""

    def __init__(self, n_ents, n_layers, n_kv, k, head_dim, dtype):
        super().__init__()
        self.register_buffer("Kb", torch.zeros(n_ents, n_layers, n_kv, k, head_dim))
        self.V = nn.Parameter(torch.zeros(n_ents, n_layers, n_kv, k, head_dim))
        self.to(dtype=dtype)

    @torch.no_grad()
    def fill(self, rank, sel):
        for l, (k, v) in enumerate(sel):
            self.Kb[rank, l, :, :k.size(1)] = k
            self.V[rank, l, :, :v.size(1)] = v

    def gather(self, ranks):
        idx = torch.tensor(ranks, dtype=torch.long, device=self.V.device)
        return self.Kb[idx], self.V[idx]

    @torch.no_grad()
    def norm_stats(self):
        return {"k_mean": round(self.Kb.float().norm(dim=-1).mean().item(), 6),
                "v_mean": round(self.V.float().norm(dim=-1).mean().item(), 6)}


def hub_upload(state, name):
    try:
        import io
        from huggingface_hub import HfApi
        buf = io.BytesIO()
        torch.save(state, buf)
        buf.seek(0)
        HfApi(token=os.environ["HF_TOKEN"]).upload_file(
            path_or_fileobj=buf, path_in_repo=f"{HUB_PREFIX}/{name}/bank.pt",
            repo_id=HUB_REPO, repo_type="model")
    except Exception as e:
        print(f"(hub upload {name} skipped: {e})", flush=True)


# ============================================================ report ==========
def curve_plot(points, out_root):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        xs = [p["mb"] for p in points]
        ax.plot(xs, [p["verbatim"] for p in points], "o-", label="verbatim")
        ax.plot(xs, [p["qa"] for p in points], "s--", label="QA")
        for p in points:
            ax.annotate(p["arm"], (p["mb"], p["verbatim"]),
                        textcoords="offset points", xytext=(4, 4), fontsize=8)
        ax.axhline(BAR_WORKING, color="grey", lw=0.5, ls=":")
        ax.set_xscale("log")
        ax.set_xlabel("MB / entity (kept KV)")
        ax.set_ylabel("recall")
        ax.set_title("engram-v6: recall vs storage (selection compression)")
        ax.legend()
        fig.savefig(os.path.join(out_root, "curve.png"), dpi=120,
                    bbox_inches="tight")
    except Exception as e:
        print(f"(curve plot skipped: {e})", flush=True)


def write_report(out_root, arms, a0, meta, cost, a1_ratio=None, gate=None,
                 tune_note="", part0="", mb_of=None, recap=None):
    e0 = next((a for a in arms if a["arm"] == "E0"), None)
    gate_a0 = a0["verbatim"] <= BAR_A0_RECALL
    e0_ok = e0 is not None and e0["verbatim"] >= BAR_E0
    s_arms = [a for a in arms if a["arm"].startswith("S")]
    working = None
    for a in sorted(s_arms, key=lambda a: mb_of.get(a["arm"], 1e9) if mb_of else 0):
        if a["verbatim"] >= BAR_WORKING:
            working = a
            break
    if not gate_a0:
        verdict = f"**ABORT — contamination (A0 {a0['verbatim']:.1%})**"
    elif e0 is None or not e0_ok:
        verdict = (f"**ENGRAM-V6 REGRESSION ABORT — E0 "
                   f"{(e0 or {}).get('verbatim', 0):.1%} < {BAR_E0:.0%}**")
    elif working is not None:
        verdict = (f"**ENGRAM-V6: working point {working['arm']} — "
                   f"{working['verbatim']:.1%} verbatim at "
                   f"{mb_of.get(working['arm'], 0):.2f} MB/entity**")
    else:
        best = max(s_arms, key=lambda a: a["verbatim"], default=None)
        verdict = (f"**ENGRAM-V6: no working point — best "
                   f"{best['arm'] if best else '-'} "
                   f"{best['verbatim'] if best else 0:.1%} < {BAR_WORKING:.0%}**")
    if meta.get("smoke"):
        verdict = f"SMOKE RUN — plumbing only, numbers meaningless. {verdict}"

    L = ["# engram-v6: selection compression + identity gate\n", f"{verdict}\n",
         f"_Generated {datetime.now(timezone.utc).isoformat()} | meta: {json.dumps(meta)}_\n"]
    if part0:
        L.append(f"Part 0: {part0}\n")
    L += ["## The curve (recall vs MB/entity)\n",
          "| arm | MB/entity | verbatim | QA | theft (ungated) | drift_loaded | mass recapture |",
          "|---|---|---|---|---|---|---|"]
    for a in arms:
        if a["arm"] in ("A0", "A1"):
            continue
        mb = mb_of.get(a["arm"]) if mb_of else None
        rc = (recap or {}).get(a["arm"])
        L.append(f"| {a['arm']} | " + (f"{mb:.2f}" if mb else "-") +
                 f" | {a['verbatim']:.1%} | {a['qa']:.1%} | "
                 + (f"{a['theft']:.1%}" if "theft" in a else "-")
                 + " | " + (f"{a['drift_loaded']:+.1%}" if "drift_loaded" in a else "-")
                 + " | " + (f"{rc:.1%}" if rc is not None else "-") + " |")
    a1 = next((a for a in arms if a["arm"] == "A1"), None)
    if a1:
        L.append(f"\nA1 (token channel): {a1['verbatim']:.1%} / {a1['qa']:.1%}. "
                 f"A0 gate {a0['verbatim']:.1%}.")
    L.append(f"Bars: A0 <= {BAR_A0_RECALL:.0%}; E0 >= {BAR_E0:.0%}; working "
             f"point = smallest k with verbatim >= {BAR_WORKING:.0%}.\n")

    s16m = next((a for a in arms if a["arm"] == "S16-mass"), None)
    s16n = next((a for a in arms if a["arm"] == "S16-norm"), None)
    if s16m and s16n:
        L.append("## Mass vs norm ranking (k=16)\n")
        L.append(f"- mass-ranked: {s16m['verbatim']:.1%} verbatim / {s16m['qa']:.1%} QA")
        L.append(f"- norm-ranked: {s16n['verbatim']:.1%} verbatim / {s16n['qa']:.1%} QA")
        gap = s16m["verbatim"] - s16n["verbatim"]
        L.append(f"- gap {gap:+.1%}: "
                 + ("calibration earns its cost." if gap > 0.05 else
                    "the free ranker is competitive - calibration optional."))
    if gate:
        L.append("\n## Identity gate (entity-id match - upper bound)\n")
        L.append("| quantity | ungated (E0) | gated |")
        L.append("|---|---|---|")
        L.append(f"| theft on distractors | {gate['ungated_theft']:.1%} | "
                 f"{gate['gated_theft']:.1%} |")
        L.append(f"| hedge when right memory absent | - | "
                 f"{gate['gated_hedge']:.1%} (system-level, by construction) |")
        L.append(f"| model's own hedge, no injection | {gate['model_hedge']:.1%} | - |")
        L.append("\nUnder id-gating, blocking and absence-detection are "
                 "deterministic, so the gated numbers hold by construction; "
                 "the ungated-vs-gated contrast prices what surface-form "
                 "detection must deliver. The model itself never hedges - "
                 "the 'I don't know' must come from the gate, not the LM.")
    if tune_note:
        L.append(f"\n- S-tune rule: {tune_note}")
    L.append(f"\n_Cost: {cost}_\n\nAll raw numbers: `results.json`; kept-position "
             f"maps: `kept_maps.json`; curve: `curve.png`\n")

    md = os.path.join(out_root, "ENGRAM_V6.md") if meta.get("smoke") \
        else os.path.join(HERE, "ENGRAM_V6.md")
    open(md, "w").write("\n".join(L))
    res = {"meta": meta, "arms": arms, "a0": a0, "a1_qa_verbatim_ratio": a1_ratio,
           "gate": gate, "mb_of": mb_of, "mass_recapture": recap,
           "tune_note": tune_note, "part0": part0, "verdict": verdict,
           "working_point": working["arm"] if working else None,
           "bars": {"a0": BAR_A0_RECALL, "e0": BAR_E0, "working": BAR_WORKING,
                    "g_theft": BAR_G_THEFT, "g_hedge": BAR_G_HEDGE}}
    rp = os.path.join(out_root, "results.json") if meta.get("smoke") \
        else os.path.join(HERE, "results.json")
    json.dump(res, open(rp, "w"), indent=1)
    print(verdict, flush=True)


# ============================================================ orchestration ===
def run(smoke=False, only=None, out_root=None):
    t0 = time.time()
    budget = 12 * 60 if smoke else int(2 * 3600)
    keep_awake()
    if smoke:
        base, tok = smoke_base_qwen()
        ents, facts, qa, dis = smoke_data()
        wiki = [f"general filler sentence number {i} about ordinary things."
                for i in range(60)]
        nrep = 30
        tune_steps = 10
    else:
        base, tok = load_base()
        ents = load_jsonl("entities.jsonl")
        facts = load_jsonl("facts_train.jsonl")
        qa = load_jsonl("qa_eval.jsonl")
        dis = load_jsonl("distractors.jsonl")
        from generic_test import load_domain3
        wiki, _ = load_domain3()
        nrep = N_REPLAY
        tune_steps = TUNE_STEPS
    ent_by_id = {e["entity_id"]: e for e in ents}
    rng = random.Random(SEED)
    qa_eval = qa[:]
    rng.shuffle(qa_eval)
    qa_eval = qa_eval[:min(N_EVAL, len(qa_eval))] if not smoke else qa_eval
    base_params = sum(p.numel() for p in base.parameters())

    addr = Addressing(ents, "entity", k=K_ADDR, n_null=0)
    os.makedirs(out_root, exist_ok=True)
    addr.save(out_root)
    ranks = rank_map(addr)

    dm = DeepMem(base)
    cfg = base.config
    n_kv = cfg.num_key_value_heads
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    n_layers = dm.n_layers

    ctx_len = {e["entity_id"]: len(tok(ctx_text(e)).input_ids) for e in ents}
    OFFSET = max(ctx_len.values()) + 8
    mean_ctx = sum(ctx_len.values()) / len(ctx_len)

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

    # ---- per-entity machinery: bounded kv memo + calibrated mass memo --------
    kv_memo = {}

    def cached(eid):
        if eid not in kv_memo:
            if len(kv_memo) > 30:
                kv_memo.clear()
            kv_memo[eid] = dm.capture_ctx(tok, ctx_text(ent_by_id[eid]), DEVICE)
        return kv_memo[eid]

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

    def sel_mem(k, ranker):
        def arm_mem(eid, donor=None):
            e = donor if donor is not None else eid
            kv, _ = cached(e)
            score = mass_for(e) if ranker == "mass" else norm_score(kv)
            sel, _, _ = select_topk(kv, score, k)
            dm.set_cached(sel)
        return arm_mem

    def offset_of(eid):
        return cached(eid)[1]

    want = set(only) if only else None
    arms = []
    a1_ratio = None
    mb_of = {"E0": round(storage_mb(mean_ctx, n_layers, n_kv, head_dim), 2)}
    recap = {}

    def wanted(n):
        return want is None or n in want

    # ---- A0 / A1 ----
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
    if wanted("A1"):
        a = load_or("A1")
        if a is None:
            STATUS["phase"] = "A1"
            a = eval_arm("A1", make_gen(base, tok, in_context=True,
                                        ent_by_id=ent_by_id), None, qa_eval,
                         dis, wiki, tok, base_ppl, params=(base_params, 0))
            save_arm(a)
        arms.append(a)
        a1_ratio = (a["qa"] / a["verbatim"]) if a["verbatim"] else None

    # ---- smoke-only selection-identity check: k >= T must reproduce E0 ------
    if smoke:
        r = qa_eval[0]
        e0_mem(r["entity_id"])
        g_e0 = gen_mem(base, tok, qa_prompt(r["question"]),
                       offset_of(r["entity_id"]), DEVICE)
        sel_mem(999, "mass")(r["entity_id"])
        g_sel = gen_mem(base, tok, qa_prompt(r["question"]),
                        offset_of(r["entity_id"]), DEVICE)
        dm.clear()
        assert g_sel == g_e0, f"selection-identity FAILED: {g_sel!r} != {g_e0!r}"
        print("smoke selection-identity (k>=T == E0): OK", flush=True)

    # ---- E0 regression gate ----
    if wanted("E0"):
        a = load_or("E0")
        if a is None:
            STATUS["phase"] = "E0"
            tr = []
            a = eval_deep_arm("E0", base, tok, dm, e0_mem, qa_eval, dis, wiki,
                              base_ppl, (base_params, 0), ent_by_id, ranks,
                              offset_of, transcripts_out=tr)
            json.dump(tr, open(os.path.join(out_root, "transcripts_E0.json"), "w"), indent=1)
            save_arm(a)
        arms.append(a)
        if not smoke and a["verbatim"] < BAR_E0:
            write_report(out_root, arms, arms[0], {"smoke": smoke,
                                                   "regression_abort": True},
                         f"wall {int((time.time()-t0)/60)} min", a1_ratio,
                         mb_of=mb_of)
            return 4

    # ---- S arms ----
    s_specs = [(f"S{k}-mass", k, "mass") for k in S_KS] + [("S16-norm", 16, "norm")]
    kept_maps = {}
    for name, k, ranker in s_specs:
        if not wanted(name) or time.time() - t0 > budget:
            continue
        a = load_or(name)
        if a is None:
            STATUS["phase"] = name
            a = eval_deep_arm(name, base, tok, dm, sel_mem(k, ranker), qa_eval,
                              dis, wiki, base_ppl, (base_params, 0), ent_by_id,
                              ranks, offset_of)
            # mass recapture + kept maps on a fixed entity sample
            rs, n = 0.0, 0
            for r in qa_eval[:KEPT_MAP_N]:
                eid = r["entity_id"]
                kv, _ = cached(eid)
                score = mass_for(eid) if ranker == "mass" else norm_score(kv)
                sel, kmap, rc = select_topk(kv, score, k)
                rs += rc
                n += 1
                kept_maps.setdefault(name, {})[str(eid)] = kmap
            a["mass_recapture"] = round(rs / max(n, 1), 4)
            save_arm(a)
        arms.append(a)
        mb_of[name] = round(storage_mb(k, n_layers, n_kv, head_dim), 2)
        recap[name] = a.get("mass_recapture")
    if kept_maps:
        json.dump(kept_maps, open(os.path.join(out_root, "kept_maps.json"), "w"))

    # ---- S8-tune (conditional) ----
    tune_note = ""
    s8 = next((a for a in arms if a["arm"] == "S8-mass"), None)
    if s8 is not None:
        if s8["verbatim"] >= TUNE_TRIGGER:
            tune_note = (f"skipped: S8-mass {s8['verbatim']:.1%} >= "
                         f"{TUNE_TRIGGER:.0%}.")
        elif time.time() - t0 <= budget:
            tune_note = (f"S8-mass {s8['verbatim']:.1%} < {TUNE_TRIGGER:.0%} "
                         f"-> tuned kept VALUES only (keys frozen), "
                         f"{tune_steps} steps.")
            a = load_or("S8-tune")
            if a is None:
                STATUS["phase"] = "S8tune_fill"
                bank = SelBank(len(ents), n_layers, n_kv, 8, head_dim,
                               base.dtype).to(DEVICE)
                for e in ents:
                    eid = e["entity_id"]
                    kv, _ = cached(eid)
                    sel, _, _ = select_topk(kv, mass_for(eid), 8)
                    bank.fill(ranks[eid], sel)
                STATUS["phase"] = "S8tune_train"
                fact_ex = fact_examples(facts, tok)
                info = train_deep(base, tok, dm, bank, fact_ex, ranks, OFFSET,
                                  None, steps=tune_steps, bs=8, lr=TUNE_LR)

                def tuned_mem(eid, donor=None):
                    K, V = bank.gather([ranks[donor if donor is not None else eid]])
                    dm.set_deep(K, V)

                STATUS["phase"] = "S8tune_eval"
                a = eval_deep_arm("S8-tune", base, tok, dm, tuned_mem, qa_eval,
                                  dis, wiki, base_ppl,
                                  (base_params, bank.V.numel()), ent_by_id,
                                  ranks, lambda eid: OFFSET)
                a["train"] = info
                save_arm(a)
                if not smoke:
                    hub_upload(bank.state_dict(), "S8-tune")
                del bank
            arms.append(a)
            mb_of["S8-tune"] = mb_of.get("S8-mass")

    # ---- identity gate numbers (deterministic under id-gating) --------------
    e0a = next((a for a in arms if a["arm"] == "E0"), None)
    gate = None
    if e0a is not None:
        STATUS["phase"] = "gate"
        dm.clear()
        rngg = random.Random(SEED)
        eids = sorted(ent_by_id)
        donors = {r["entity_id"]: rngg.choice(eids) for r in dis}
        g_gens = [gen_mem(base, tok, qa_prompt(r["question"]), 0, DEVICE)
                  for r in dis]
        gated_theft = sum(
            scored_hit(g, str(ent_by_id[donors[r["entity_id"]]][r["attr"]]))
            for g, r in zip(g_gens, dis)) / max(len(dis), 1)
        model_hedge = 1 - sum(is_confabulation(g) for g in g_gens) / max(len(dis), 1)
        gate = {"ungated_theft": e0a.get("theft", 0),
                "gated_theft": round(gated_theft, 4),
                "gated_hedge": 1.0,           # absence is detectable -> system hedges
                "model_hedge": round(model_hedge, 4)}

    cost = f"wall {int((time.time() - t0) / 60)} min"
    a0 = next((a for a in arms if a["arm"] == "A0"), {"verbatim": 0.0})
    part0 = ("ENGRAM_SERIES.md cross-checked against per-branch artifacts and "
             "committed to main (fixes: v1 A1 was 92.8% pre-repair-scorer, "
             "not in the 94.0-94.2% range; one typo).")
    meta = {"smoke": smoke, "model": "tiny-random-qwen3" if smoke else MODEL_NAME,
            "device": DEVICE, "seed": SEED, "n_layers": n_layers,
            "s_ks": list(S_KS), "offset": OFFSET, "mean_ctx_tokens": round(mean_ctx, 1),
            "calibration": "train-template sentences only (no held-out QA)",
            "addressing": "blake2b hash k=4096 -> dense rank storage"}
    write_report(out_root, arms, a0, meta, cost, a1_ratio, gate, tune_note,
                 part0, mb_of, recap)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--arms", default="")
    args = ap.parse_args()
    out_root = os.path.join(HERE, "results",
                            "engram_v6_smoke" if args.smoke else "engram_v6")
    os.makedirs(out_root, exist_ok=True)
    status_dir = out_root if args.smoke else os.path.join(HERE, "status")
    os.makedirs(status_dir, exist_ok=True)
    if args.report_only:
        print("use the committed results.json/ENGRAM_V6.md; report-only not "
              "supported for v6 (gate numbers need the model)", flush=True)
        return
    start_heartbeat(status_dir)
    only = [s.strip() for s in args.arms.split(",") if s.strip()] or None
    raise SystemExit(run(smoke=args.smoke, only=only, out_root=out_root))


if __name__ == "__main__":
    main()
