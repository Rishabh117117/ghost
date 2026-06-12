"""
engram_v5.py - multi-layer presence & the compression frontier.

Four runs, one conserved failure: every injection so far was SINGLE-SITE and
nulled; tokens-in-context held 94% throughout. Hypothesis: tokens work
because they are present in K,V at EVERY layer (the shallow-vs-deep prompt
result). v5 is designed to be UN-NULLABLE: it anchors on a condition that
must work (E0 = cached-KV reuse, mathematically adjacent to A1) and measures
how far it compresses (E1 = m learned pairs per layer). The deliverable is a
curve: A1 -> E0 -> E1 (-> E2).

Arms:
  A0  gate (<= 5% or ABORT)        A1  in-context rerun
  E0  cached-KV anchor, no training: the entity's context paragraph cached at
      ALL layers, question asked at offset positions. GATE: verbatim >= 85%
      or the plumbing is broken -> ABORT before E1. (The injected path is
      also checked against the native full-context forward on the first 5
      rows - exact logit equivalence, reported as e0_max_logit_delta.)
  E1  learned deep KV: m=8 pairs x EVERY layer per entity, warm-started by
      mean-pooling that entity's E0 cache, trained (answer-token CE +
      replay-KL 0.5 guard).
  E2  exactly one, by rule: E1 verbatim >= 60% -> E1 with m=2 (the second
      point on the compression curve); else E0 restricted to layers 0..L/2-1
      (is EARLY-layer presence the load-bearing half? no training, cheap).
  Interference: SKIPPED BY DESIGN (per-entity parameters are disjoint).

Discovery: compression ratio, theft (distractor + random entity's memory),
drift_loaded, per-attribute recall, attention-mass-on-memory per layer (the
"did the reader look" number v4 lacked).
"""
import argparse
import json
import math
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
from engram import (cloze, context_for, load_jsonl, make_gen, eval_arm,
                    wikitext_ppl, pad_batch, precompute_replay, kl_replay,
                    load_base)
from engram_v3 import Addressing
from engram_v4 import fact_examples, smoke_base_qwen, smoke_data
from engram_deep import DeepMem, DeepKVBank, gen_mem

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- pinned hyperparameters --------------------------------------------------
M_PAIRS     = 8
M_PAIRS_E2  = 2
LR          = 5e-4       # gentler than v3/v4: warm-start must be refined, not erased
MAX_STEPS   = 1500
BATCH       = 16
MAX_LEN     = 64
GEN_MAX_NEW = 16
N_EVAL      = 400
N_REPLAY    = 2000
KL_WEIGHT   = 0.5
REPLAY_SHARE = 0.25
NORM_LOG_EVERY = 100
N_MASS      = 50         # rows for attention-mass telemetry
N_EQUIV     = 5          # rows for the E0 exact-equivalence check
K_ADDR      = 4096       # hash space (addressing semantics); storage is dense rank
HUB_REPO    = "Spartan117Ri/ghost-ckpts"
HUB_PREFIX  = "engram-v5"

# ---- pass bars (pinned) --------------------------------------------------------
BAR_A0_RECALL = 0.05
BAR_E0        = 0.85     # plumbing gate
BAR_VERBATIM  = 0.80
BAR_QA        = 0.60
BAR_DRIFT     = 0.02
E2_TRIGGER    = 0.60


# ============================================================ helpers =========
def ctx_text(e):
    return context_for(e)


def rank_map(addr):
    """entity_id -> dense storage row, ordered by the committed hash slot
    (storage layout only; addressing stays the deterministic hash)."""
    by_slot = sorted(addr.map.items(), key=lambda kv: kv[1]["slots"][0])
    return {eid: i for i, (eid, _) in enumerate(by_slot)}


@torch.no_grad()
def fwd_loss(base, ids, position_ids=None):
    lg = base(ids, position_ids=position_ids).logits.float()
    a_ = lg[:, :-1].contiguous()
    b_ = ids[:, 1:].contiguous()
    return F.cross_entropy(a_.view(-1, a_.size(-1)), b_.view(-1))


@torch.no_grad()
def wiki_ppl_mem(base, tok, wiki, offset=0, set_mem=None, n=200, max_len=256):
    """Wikitext ppl with optional per-line memory arming via set_mem(rng)."""
    rng = random.Random(SEED + 7)
    tot, ntok = 0.0, 0
    for t in wiki[:n]:
        ids = tok(t, return_tensors="pt", truncation=True,
                  max_length=max_len).input_ids.to(DEVICE)
        if ids.size(1) < 2:
            continue
        if set_mem is not None:
            set_mem(rng)
        pos = (torch.arange(offset, offset + ids.size(1), device=DEVICE)[None]
               if offset else None)
        loss = fwd_loss(base, ids, pos)
        tot += loss.item() * (ids.size(1) - 1)
        ntok += ids.size(1) - 1
    return math.exp(tot / max(ntok, 1))


# ============================================================ eval ============
def eval_deep_arm(name, base, tok, dm, arm_mem, qa_rows, distractors, wiki,
                  base_ppl, params, ent_by_id, ranks, offset_for,
                  transcripts_out=None, mass_out=None):
    """arm_mem(eid_or_None, donor=None): arms the memory for one entity (or
    clears). offset_for(eid): text position offset for that query."""

    def run_rows(rows, build, donor=None):
        outs = []
        for r in rows:
            d = donor(r) if donor else None
            arm_mem(r["entity_id"], donor=d)
            outs.append(gen_mem(base, tok, build(r),
                                offset_for(d if d is not None else r["entity_id"]),
                                DEVICE, max_new=GEN_MAX_NEW))
        dm.clear()
        return outs

    v_gens = run_rows(qa_rows, lambda r: cloze(r["name"], r["attr"]))
    q_gens = run_rows(qa_rows, lambda r: qa_prompt(r["question"]))
    verb = sum(scored_hit(g, r["answer"]) for g, r in zip(v_gens, qa_rows)) / max(len(qa_rows), 1)
    qa = sum(scored_hit(g, r["answer"]) for g, r in zip(q_gens, qa_rows)) / max(len(qa_rows), 1)

    by_attr = {}
    for r, g in zip(qa_rows, v_gens):
        a = by_attr.setdefault(r["attr"], [0, 0])
        a[0] += int(scored_hit(g, r["answer"]))
        a[1] += 1
    per_attr = {k: round(h / n, 4) for k, (h, n) in sorted(by_attr.items())}

    dm.clear()
    c_gens = [gen_mem(base, tok, qa_prompt(r["question"]), 0, DEVICE)
              for r in distractors]
    confab = sum(is_confabulation(g) for g in c_gens) / max(len(distractors), 1)

    rng = random.Random(SEED)
    eids = sorted(ent_by_id)
    donors = {r["entity_id"]: rng.choice(eids) for r in distractors}
    t_gens = run_rows(distractors, lambda r: qa_prompt(r["question"]),
                      donor=lambda r: donors[r["entity_id"]])
    theft = sum(scored_hit(g, str(ent_by_id[donors[r["entity_id"]]][r["attr"]]))
                for g, r in zip(t_gens, distractors)) / max(len(distractors), 1)

    dm.clear()
    ppl_clean = wiki_ppl_mem(base, tok, wiki)
    rng2 = random.Random(SEED + 1)
    moff = max(offset_for(e) for e in eids)

    def loaded(rng):
        arm_mem(rng.choice(eids))
    ppl_loaded = wiki_ppl_mem(base, tok, wiki, offset=moff, set_mem=loaded)
    dm.clear()

    if mass_out is not None:
        dm.telemetry(True)
        acc = None
        nm = 0
        for r in qa_rows[:N_MASS]:
            arm_mem(r["entity_id"])
            ids = tok(qa_prompt(r["question"]), return_tensors="pt").input_ids.to(DEVICE)
            off = offset_for(r["entity_id"])
            with torch.no_grad():
                base(ids, position_ids=torch.arange(off, off + ids.size(1),
                                                    device=DEVICE)[None])
            mm = dm.mass_per_layer()
            if all(x is not None for x in mm):
                acc = mm if acc is None else [a + b for a, b in zip(acc, mm)]
                nm += 1
        dm.telemetry(False)
        dm.clear()
        if acc:
            mass_out.extend(round(a / nm, 4) for a in acc)

    if transcripts_out is not None:
        for r, g in list(zip(qa_rows, q_gens))[:20]:
            transcripts_out.append({"q": r["question"], "gold": r["answer"], "gen": g})

    return {"arm": name, "verbatim": round(verb, 4), "qa": round(qa, 4),
            "confab": round(confab, 4), "theft": round(theft, 4),
            "wiki_ppl": round(ppl_clean, 3),
            "drift": round((ppl_clean - base_ppl) / base_ppl, 4) if base_ppl else 0,
            "wiki_ppl_loaded": round(ppl_loaded, 3),
            "drift_loaded": round((ppl_loaded - base_ppl) / base_ppl, 4) if base_ppl else 0,
            "per_attr_verbatim": per_attr,
            "params_trainable": params[1], "params_base": params[0]}


# ============================================================ training ========
def train_deep(base, tok, dm, bank, fact_ex, ranks, offset, replay_cache,
               steps=MAX_STEPS, bs=BATCH, lr=LR, seed=SEED, traj=None):
    opt = torch.optim.AdamW(bank.parameters(), lr=lr, weight_decay=0.0)
    rng = random.Random(seed)
    n_rep = max(int(REPLAY_SHARE * bs), 1) if replay_cache else 0

    class Shim:                                 # for the kl_replay guard
        @staticmethod
        def logits(ids, attn=None):
            return base(ids, attention_mask=attn).logits

    loss = torch.zeros(())
    for step in range(1, steps + 1):
        fb = [fact_ex[rng.randrange(len(fact_ex))] for _ in range(bs - n_rep)]
        ids, attn, lab = pad_batch([(x, y) for x, y, _, _, _ in fb], tok, DEVICE)
        K, V = bank.gather([ranks[eid] for _, _, _, eid, _ in fb])
        dm.set_deep(K, V)
        pos = torch.arange(offset, offset + ids.size(1),
                           device=DEVICE)[None].expand(ids.size(0), -1)
        lg = base(ids, attention_mask=attn, position_ids=pos).logits
        loss = F.cross_entropy(lg[:, :-1].reshape(-1, lg.size(-1)),
                               lab[:, 1:].reshape(-1), ignore_index=-100)
        if n_rep:
            rb = [replay_cache[rng.randrange(len(replay_cache))]
                  for _ in range(n_rep)]
            dm.clear()                          # structural silence guard
            loss = loss + KL_WEIGHT * kl_replay(Shim, rb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        STATUS["step"] = step
        if traj is not None and (step % NORM_LOG_EVERY == 0 or step == 1):
            traj.append({"step": step, **bank.norm_stats()})
    dm.clear()
    return {"final_loss": float(loss.item()), "steps": step}


# ============================================================ report ==========
def write_report(out_root, arms, a0, meta, cost, a1_ratio=None, e2_note="",
                 equiv=None, mass=None, compression=None):
    e0 = next((a for a in arms if a["arm"] == "E0"), None)
    e1 = next((a for a in arms if a["arm"] == "E1"), None)
    gate = a0["verbatim"] <= BAR_A0_RECALL
    e0_ok = e0 is not None and e0["verbatim"] >= BAR_E0
    e1_ok = (e1 is not None and e1["verbatim"] >= BAR_VERBATIM
             and e1["qa"] >= BAR_QA and e1["drift"] <= BAR_DRIFT)
    if not gate:
        verdict = f"**ABORT — contamination (A0 {a0['verbatim']:.1%})**"
    elif e0 is None:
        verdict = "**ENGRAM-V5 FAIL — E0 did not run**"
    elif not e0_ok:
        verdict = (f"**ENGRAM-V5 PLUMBING ABORT — E0 {e0['verbatim']:.1%} < "
                   f"{BAR_E0:.0%} (nothing scientific concluded; fix and rerun E0)**")
    elif e1_ok:
        verdict = "**ENGRAM-V5 PASS — the engram exists (E1 cleared all bars)**"
    elif e1 is not None:
        f = [b for b, bad in (("verbatim", e1["verbatim"] < BAR_VERBATIM),
                              ("QA", e1["qa"] < BAR_QA),
                              ("drift", e1["drift"] > BAR_DRIFT)) if bad]
        verdict = (f"**ENGRAM-V5: E0 anchored ({e0['verbatim']:.1%}), E1 "
                   f"FAILED compression — {', '.join(f)}**")
    else:
        verdict = "**ENGRAM-V5 INCOMPLETE — E1 missing**"
    if meta.get("smoke"):
        verdict = f"SMOKE RUN — plumbing only, numbers meaningless. {verdict}"

    desc = {"A0": "base (gate)", "A1": "in-context (token channel)",
            "E0": "cached KV, ALL layers, no training (anchor)",
            "E1": f"learned deep KV m={M_PAIRS}/layer, warm-started",
            "E1-m2": f"learned deep KV m={M_PAIRS_E2}/layer",
            "E0-half": "cached KV, layers 0..L/2-1 only"}
    L = ["# engram-v5: multi-layer presence & the compression frontier\n",
         f"{verdict}\n",
         f"_Generated {datetime.now(timezone.utc).isoformat()} | meta: {json.dumps(meta)}_\n"]
    if equiv is not None:
        L.append(f"E0 exact-equivalence check vs native full-context forward: "
                 f"max |delta logits| = {equiv:.2e} over {N_EQUIV} rows.\n")
    L += ["## The curve (A1 -> E0 -> E1 -> E2)\n",
          "| arm | what | verbatim | QA | confab | theft | drift_clean | drift_loaded | params |",
          "|---|---|---|---|---|---|---|---|---|"]
    for a in arms:
        L.append(f"| {a['arm']} | {desc.get(a['arm'], '')} | {a['verbatim']:.1%} | "
                 f"{a['qa']:.1%} | {a['confab']:.1%} | "
                 + (f"{a['theft']:.1%}" if "theft" in a else "-")
                 + f" | {a['drift']:+.1%} | "
                 + (f"{a['drift_loaded']:+.1%}" if "drift_loaded" in a else "-")
                 + f" | {a['params_trainable']:,} |")
    L.append(f"\nBars: A0 <= {BAR_A0_RECALL:.0%} | E0 >= {BAR_E0:.0%} (plumbing gate) | "
             f"E1 verbatim >= {BAR_VERBATIM:.0%}, QA >= {BAR_QA:.0%}, "
             f"drift_clean <= {BAR_DRIFT:.0%}.\n")
    if compression:
        L.append("## Compression\n")
        for k, v in compression.items():
            L.append(f"- {k}: {v}")
    if mass:
        L.append("\n## Attention mass on memory pairs, per layer (did the reader look?)\n")
        for arm, mm in mass.items():
            if mm:
                L.append(f"- **{arm}**: mean {sum(mm)/len(mm):.1%} | per layer: "
                         + " ".join(f"{x:.0%}" for x in mm))
    L.append("\n## Per-attribute verbatim\n")
    pos_arms = [a for a in arms if a["arm"].startswith(("E0", "E1"))]
    attrs = sorted({k for a in pos_arms for k in a.get("per_attr_verbatim", {})})
    if attrs:
        L.append("| arm | " + " | ".join(attrs) + " |")
        L.append("|" + "---|" * (len(attrs) + 1))
        for a in pos_arms:
            pa = a.get("per_attr_verbatim", {})
            L.append(f"| {a['arm']} | " + " | ".join(f"{pa.get(k, 0):.0%}" for k in attrs) + " |")
    if a1_ratio is not None:
        L.append(f"\n- scorer sanity: A1 QA/verbatim = {a1_ratio:.2f} (gate >= 0.80).")
    if e2_note:
        L.append(f"- E2 rule: {e2_note}")
    L.append("- interference: skipped by design (per-entity parameters are disjoint).")
    L.append(f"\n_Cost: {cost}_\n\nAll raw numbers: `results.json`\n")

    md = os.path.join(out_root, "ENGRAM_V5.md") if meta.get("smoke") \
        else os.path.join(HERE, "ENGRAM_V5.md")
    open(md, "w").write("\n".join(L))
    res = {"meta": meta, "arms": arms, "a0": a0, "a1_qa_verbatim_ratio": a1_ratio,
           "e0_max_logit_delta": equiv, "attention_mass": mass,
           "compression": compression, "e2_note": e2_note, "verdict": verdict,
           "passed": bool(gate and e0_ok and e1_ok),
           "bars": {"a0": BAR_A0_RECALL, "e0": BAR_E0, "verbatim": BAR_VERBATIM,
                    "qa": BAR_QA, "drift": BAR_DRIFT}}
    rp = os.path.join(out_root, "results.json") if meta.get("smoke") \
        else os.path.join(HERE, "results.json")
    json.dump(res, open(rp, "w"), indent=1)
    print(verdict, flush=True)


# ============================================================ orchestration ===
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


def run(smoke=False, only=None, out_root=None):
    t0 = time.time()
    budget = 12 * 60 if smoke else int(3 * 3600)
    keep_awake()
    if smoke:
        base, tok = smoke_base_qwen()
        ents, facts, qa, dis = smoke_data()
        steps, bs = 30, 8
        wiki = [f"general filler sentence number {i} about ordinary things."
                for i in range(60)]
        nrep = 30
    else:
        base, tok = load_base()
        ents = load_jsonl("entities.jsonl")
        facts = load_jsonl("facts_train.jsonl")
        qa = load_jsonl("qa_eval.jsonl")
        dis = load_jsonl("distractors.jsonl")
        steps, bs = MAX_STEPS, BATCH
        from generic_test import load_domain3
        wiki, _ = load_domain3()
        nrep = N_REPLAY
    ent_by_id = {e["entity_id"]: e for e in ents}
    rng = random.Random(SEED)
    qa_eval = qa[:]
    rng.shuffle(qa_eval)
    qa_eval = qa_eval[:min(N_EVAL, len(qa_eval))] if not smoke else qa_eval
    base_params = sum(p.numel() for p in base.parameters())
    fact_ex = fact_examples(facts, tok)

    addr = Addressing(ents, "entity", k=K_ADDR, n_null=0)
    os.makedirs(out_root, exist_ok=True)
    addr.save(out_root)
    ranks = rank_map(addr)

    dm = DeepMem(base)
    cfg = base.config
    n_kv = cfg.num_key_value_heads
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    n_layers = dm.n_layers

    # entity context token lengths -> pinned OFFSET for E1 (memory sits
    # at positions [0, OFFSET); text starts at OFFSET, train == eval)
    ctx_ids_len = {e["entity_id"]: len(tok(ctx_text(e)).input_ids) for e in ents}
    OFFSET = max(ctx_ids_len.values()) + 8
    print(f"ctx tokens: mean {sum(ctx_ids_len.values())/len(ctx_ids_len):.1f}, "
          f"max {max(ctx_ids_len.values())} -> OFFSET {OFFSET}", flush=True)

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
    STATUS["phase"] = "replay_precompute"
    replay_cache = precompute_replay(base, tok, wiki, n=nrep)
    print(f"replay cache: {len(replay_cache)} chunks", flush=True)

    want = set(only) if only else {"A0", "A1", "E0", "E1"}
    arms = []
    a1_ratio = None
    mass = {}
    traj_by_arm = {}
    equiv = None

    if "A0" in want:
        a = load_or("A0")
        if a is None:
            STATUS["phase"] = "A0"
            g = make_gen(base, tok)
            a = eval_arm("A0", g, None, qa_eval, dis, wiki, tok, base_ppl,
                         params=(base_params, 0))
            save_arm(a)
        arms.append(a)
        if not smoke and a["verbatim"] > BAR_A0_RECALL:
            json.dump({"reason": "A0 contamination", "a0": a},
                      open(os.path.join(out_root, "ABORT.json"), "w"))
            write_report(out_root, arms, a, {"smoke": smoke, "model": MODEL_NAME,
                                             "aborted": True}, "aborted at gate")
            return 3

    if "A1" in want:
        a = load_or("A1")
        if a is None:
            STATUS["phase"] = "A1"
            g = make_gen(base, tok, in_context=True, ent_by_id=ent_by_id)
            a = eval_arm("A1", g, None, qa_eval, dis, wiki, tok, base_ppl,
                         params=(base_params, 0))
            save_arm(a)
        arms.append(a)
        a1_ratio = (a["qa"] / a["verbatim"]) if a["verbatim"] else None

    # ---- E0: cached-KV anchor ----
    kv_cache_of = {}

    def cached_kv(eid):
        if eid not in kv_cache_of:
            if len(kv_cache_of) > 40:           # bounded GPU memory
                kv_cache_of.clear()
            kv_cache_of[eid] = dm.capture_ctx(tok, ctx_text(ent_by_id[eid]), DEVICE)
        return kv_cache_of[eid]

    def e0_arm_mem(layers=None):
        def arm_mem(eid, donor=None):
            kv, _ = cached_kv(donor if donor is not None else eid)
            dm.set_cached(kv, layers=layers)
        return arm_mem

    def e0_offset(eid):
        return cached_kv(eid)[1]

    if "E0" in want:
        a = load_or("E0")
        if a is None:
            STATUS["phase"] = "E0_equiv"
            deltas = []
            for r in qa_eval[:N_EQUIV]:
                e = ent_by_id[r["entity_id"]]
                c_ids = tok(ctx_text(e), return_tensors="pt").input_ids.to(DEVICE)
                p_ids = tok(qa_prompt(r["question"]),
                            return_tensors="pt").input_ids.to(DEVICE)
                joint = torch.cat([c_ids, p_ids], dim=1)
                dm.clear()
                with torch.no_grad():
                    ref = base(joint).logits[:, -p_ids.size(1):]
                kv, T = dm.capture_ctx_ids(c_ids)
                dm.set_cached(kv)
                pos = torch.arange(T, T + p_ids.size(1), device=DEVICE)[None]
                with torch.no_grad():
                    inj = base(p_ids, position_ids=pos).logits
                deltas.append((inj - ref).abs().max().item())
                dm.clear()
            equiv = max(deltas)
            print(f"E0 equivalence: max |delta logits| = {equiv:.2e}", flush=True)
            if smoke:
                assert equiv < 1e-3, \
                    f"SMOKE FAIL: E0 injection != native forward ({equiv:.2e})"
            STATUS["phase"] = "E0"
            tr = []
            mm = []
            a = eval_deep_arm("E0", base, tok, dm, e0_arm_mem(), qa_eval, dis,
                              wiki, base_ppl, (base_params, 0), ent_by_id,
                              ranks, e0_offset, transcripts_out=tr, mass_out=mm)
            a["e0_max_logit_delta"] = equiv
            mass["E0"] = mm
            json.dump(tr, open(os.path.join(out_root, "transcripts_E0.json"), "w"), indent=1)
            save_arm(a)
        else:
            equiv = a.get("e0_max_logit_delta")
            mm = a.get("mass") or []
            mass["E0"] = mm
        arms.append(a)
        if not smoke and a["verbatim"] < BAR_E0:
            write_report(out_root, arms, arms[0],
                         {"smoke": smoke, "model": MODEL_NAME,
                          "plumbing_abort": True},
                         f"wall {int((time.time()-t0)/60)} min", a1_ratio,
                         "E0 below gate; E1 not attempted.", equiv, mass)
            return 4

    # ---- E1: the compression test ----
    def deep_arm(name, m_pairs):
        bank = DeepKVBank(len(ents), n_layers, n_kv, m_pairs, head_dim,
                          base.dtype).to(DEVICE)
        STATUS["phase"] = f"{name}_warmstart"
        t = time.time()
        for e in ents:
            kv, _ = dm.capture_ctx(tok, ctx_text(e), DEVICE)
            bank.warm_start(ranks[e["entity_id"]], kv)
        print(f"warm-start: {len(ents)} entities in {int(time.time()-t)}s", flush=True)
        STATUS["phase"] = f"{name}_train"
        traj = []
        info = train_deep(base, tok, dm, bank, fact_ex, ranks, OFFSET,
                          replay_cache, steps=steps, bs=bs, traj=traj)
        traj_by_arm[name] = traj
        json.dump(traj, open(os.path.join(out_root, f"norm_traj_{name}.json"), "w"), indent=1)
        STATUS["phase"] = f"{name}_eval"

        def arm_mem(eid, donor=None):
            K, V = bank.gather([ranks[donor if donor is not None else eid]])
            dm.set_deep(K, V)

        tr = []
        mm = []
        ep = sum(p.numel() for p in bank.parameters())
        a = eval_deep_arm(name, base, tok, dm, arm_mem, qa_eval, dis, wiki,
                          base_ppl, (base_params, ep), ent_by_id, ranks,
                          lambda eid: OFFSET, transcripts_out=tr, mass_out=mm)
        a["train"] = info
        mass[name] = mm
        json.dump(tr, open(os.path.join(out_root, f"transcripts_{name}.json"), "w"), indent=1)
        if not smoke:
            hub_upload(bank.state_dict(), name)
        del bank
        torch.cuda.empty_cache() if DEVICE == "cuda" else None
        return a

    if "E1" in want and time.time() - t0 <= budget:
        a = load_or("E1")
        if a is None:
            a = deep_arm("E1", M_PAIRS)
            save_arm(a)
        arms.append(a)

    # ---- E2: exactly one, by rule ----
    e1 = next((a for a in arms if a["arm"] == "E1"), None)
    e2_note = ""
    if e1 is not None and time.time() - t0 <= budget:
        if e1["verbatim"] >= E2_TRIGGER:
            e2_note = (f"E1 verbatim {e1['verbatim']:.1%} >= {E2_TRIGGER:.0%} "
                       f"-> E2 = m={M_PAIRS_E2} (squeeze harder).")
            a = load_or("E1-m2")
            if a is None:
                a = deep_arm("E1-m2", M_PAIRS_E2)
                save_arm(a)
            arms.append(a)
        else:
            e2_note = (f"E1 verbatim {e1['verbatim']:.1%} < {E2_TRIGGER:.0%} "
                       f"-> E2 = E0 on layers 0..{n_layers//2 - 1} only.")
            a = load_or("E0-half")
            if a is None:
                STATUS["phase"] = "E0_half"
                half = list(range(n_layers // 2))
                a = eval_deep_arm("E0-half", base, tok, dm,
                                  e0_arm_mem(layers=half), qa_eval, dis, wiki,
                                  base_ppl, (base_params, 0), ent_by_id, ranks,
                                  e0_offset)
                save_arm(a)
            arms.append(a)

    have = {a["arm"] for a in arms}
    complete = want.issubset(have)
    if not complete:
        print(f"incomplete (have {sorted(have)}); exit for resume", flush=True)
        return 2

    # compression accounting
    mean_ctx = sum(ctx_ids_len.values()) / len(ctx_ids_len)
    kv_bytes_per_tok = n_layers * n_kv * head_dim * 2 * 2     # K+V, bf16
    comp = {
        "A1/E0 cache per entity": f"{mean_ctx:.0f} tokens x {n_layers} layers "
                                  f"= {mean_ctx * kv_bytes_per_tok / 1e6:.2f} MB",
        f"E1 (m={M_PAIRS})": f"{M_PAIRS} pairs x {n_layers} layers = "
                             f"{M_PAIRS * kv_bytes_per_tok / 1e6:.2f} MB "
                             f"({mean_ctx / M_PAIRS:.1f}x compression)",
        f"E2 (m={M_PAIRS_E2})": f"{mean_ctx / M_PAIRS_E2:.1f}x compression",
    }
    cost = f"wall {int((time.time() - t0) / 60)} min"
    a0 = next((a for a in arms if a["arm"] == "A0"), {"verbatim": 0.0})
    meta = {"smoke": smoke, "model": "tiny-random-qwen3" if smoke else MODEL_NAME,
            "device": DEVICE, "seed": SEED, "n_layers": n_layers,
            "m_pairs": M_PAIRS, "lr": LR, "offset": OFFSET,
            "kl_w": KL_WEIGHT, "replay_share": REPLAY_SHARE,
            "addressing": "blake2b hash k=4096 -> dense rank storage"}
    write_report(out_root, arms, a0, meta, cost, a1_ratio, e2_note, equiv,
                 mass, comp)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--arms", default="")
    args = ap.parse_args()
    out_root = os.path.join(HERE, "results",
                            "engram_v5_smoke" if args.smoke else "engram_v5")
    os.makedirs(out_root, exist_ok=True)
    status_dir = out_root if args.smoke else os.path.join(HERE, "status")
    os.makedirs(status_dir, exist_ok=True)
    if args.report_only:
        arms = [json.load(open(os.path.join(out_root, f)))
                for f in sorted(os.listdir(out_root)) if f.startswith("arm_")]
        order = {"A0": 0, "A1": 1, "E0": 2, "E1": 3, "E1-m2": 4, "E0-half": 5}
        arms.sort(key=lambda a: order.get(a["arm"], 9))
        a0 = next((a for a in arms if a["arm"] == "A0"), {"verbatim": 0.0})
        write_report(out_root, arms, a0,
                     {"smoke": args.smoke, "model": MODEL_NAME,
                      "report_only": True}, "report-only")
        return
    start_heartbeat(status_dir)
    only = [s.strip() for s in args.arms.split(",") if s.strip()] or None
    raise SystemExit(run(smoke=args.smoke, only=only, out_root=out_root))


if __name__ == "__main__":
    main()
