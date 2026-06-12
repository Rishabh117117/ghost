"""
engram_v3.py - hash-addressed fact memory (NO learned router).

v1 and v2 both died at the same place: a learned router collapsed (v2 B1:
7 unique slots, one slot at 68% share, values ~0) before the real question
could be tested. v3 makes addressing DETERMINISTIC by hash (engram_hash.py),
so the cold-start trap cannot occur, and finally isolates THE question:

  can a fact written to a KNOWN slot be read back through the frozen
  downstream layers via one additive mid-layer write?

Arms:
  A0  base, no context        - contamination gate (<= 5% or ABORT)
  A1  in-context, fixed scorer - RAG upper bound
  C1  engram-hash, ONE slot per ENTITY, tap 18 - the core capacity test
  C2  engram-hash, one slot per FACT (entity x attr), tap 18 - capacity probe
  C3  C1 but tap layer 9       - ONLY if C1 verbatim < 60% (depth probe)
  interference on the better of C1/C2 (250+250 batch write, batch-1 re-test)

DEVIATION (pinned before pod): the dispatch specified C2 as "S=4 consecutive
probed slots, output = mean over the S slots". That design is a numerical
replicate of C1, not a capacity increase: with zero init, all S slots receive
IDENTICAL gradients (dL/dv_j = (1/S) dL/dm for every j), and Adam's
per-parameter scale invariance makes each slot evolve exactly as the single
S=1 slot would - confirmed empirically in the CPU smoke (C2 metrics were
bit-identical to C1). Elementwise-symmetric variations (sum, fixed sign
flips) collapse the same way. C2 is therefore redefined to the minimal
deterministic design that actually adds write capacity per fact: address by
blake2b(entity_surface|attr) so each FACT owns its own d_model value vector
(5 slots/entity instead of 1). Still no learned routing of any kind.

Kept from v2: silence path (null slots, no RMSNorm), replay-KL quiet
incentive, answer-token CE, repaired scorer. Deleted: W_q/keys/softmax router,
Switch load-balance, EMA mean-centring, collapse early-abort (nothing to
collapse). New: KL WARMUP (facts-only for the first 30% of steps, then ramp
replay-KL to 1.0; replay share 25%) and permanent value-norm telemetry.

CLI: --smoke / --report-only / --arms C1,C2
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
                    load_base, smoke_base, smoke_data)
from engram_hash import (K_SLOTS, N_NULL, HashEngramBank, HashEngramModel,
                         assign_slots, null_row)

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- pinned hyperparameters --------------------------------------------------
TAP_LAYER   = 18
TAP_EARLY   = 9          # C3 depth probe
LR          = 2e-3
MAX_STEPS   = 1500
BATCH       = 16
MAX_LEN     = 64
GEN_MAX_NEW = 16
N_EVAL      = 400
N_REPLAY    = 2000
KL_WEIGHT   = 1.0
WARMUP_FRAC = 0.30       # facts-only: values must differentiate before quiet
RAMP_FRAC   = 0.10       # then ramp replay-KL 0 -> KL_WEIGHT linearly
REPLAY_SHARE = 0.25
NORM_LOG_EVERY = 100     # value-norm telemetry cadence during training
HUB_REPO    = "Spartan117Ri/ghost-ckpts"
HUB_PREFIX  = "engram-v3"

# ---- pass bars (pinned before launch; C1 is the arm under test) ---------------
BAR_A0_RECALL = 0.05
BAR_VERBATIM  = 0.80
BAR_QA        = 0.60
BAR_DRIFT     = 0.02
BAR_INTERFERE = 0.10
C3_TRIGGER    = 0.60     # run C3 only if C1 verbatim < this


# ============================================================ addressing ======
ATTRS = ["birth_year", "city", "profession", "employer", "quirk"]


class Addressing:
    """Committed deterministic key->slot map. mode='entity': one slot per
    entity (all of its facts share it). mode='fact': one slot per
    (entity, attr) pair = own value vector per fact."""

    def __init__(self, ents, mode, k=K_SLOTS, n_null=N_NULL):
        self.mode = mode
        self.k, self.n_null = k, n_null
        srt = sorted(ents, key=lambda e: e["entity_id"])
        if mode == "entity":
            pairs = [(e["entity_id"], e["name"]) for e in srt]
        else:
            pairs = [(f'{e["entity_id"]}|{a}', f'{e["name"]}|{a}')
                     for e in srt for a in ATTRS]
        self.map = assign_slots(pairs, 1, k=k, n_null=n_null)
        self.nrow = null_row(1, k=k, n_null=n_null)
        self.assigned = [s for m in self.map.values() for s in m["slots"]]

    def row(self, eid, attr):
        m = self.map.get(eid if self.mode == "entity" else f"{eid}|{attr}")
        return m["slots"] if m else self.nrow      # unknown key -> silence

    def entity_probes(self, eid):
        if self.mode == "entity":
            m = self.map.get(eid)
            return m["probes"] if m else None
        ps = [self.map[f"{eid}|{a}"]["probes"] for a in ATTRS
              if f"{eid}|{a}" in self.map]
        return sum(ps) if ps else None

    def save(self, out_root, name=None):
        p = os.path.join(out_root, f"slot_map_{name or self.mode}.json")
        probed = sum(1 for m in self.map.values() if m["probes"] > 0)
        json.dump({"K": self.k, "n_null": self.n_null, "mode": self.mode,
                   "keys": len(self.map), "keys_probed": probed,
                   "map": {str(k): v for k, v in sorted(
                       self.map.items(), key=lambda kv: str(kv[0]))}},
                  open(p, "w"), indent=1)
        return p


# ============================================================ generation ======
@torch.no_grad()
def gen_with_slots(base, tok, bank, prompts_slots, max_new=GEN_MAX_NEW):
    """Greedy generation; the bank reads each prompt's PRE-SET slot row.
    Eval-time addressing is id->slot direct (a clean upper bound on
    addressing; surface-form detection is a separate future problem)."""
    bank.eval()
    bank.enabled = True
    outs = []
    for p, sl in prompts_slots:
        bank.set_slots([sl])
        ids = tok(p, return_tensors="pt").input_ids.to(DEVICE)
        g = base.generate(ids, max_new_tokens=max_new, do_sample=False,
                          pad_token_id=tok.eos_token_id)
        outs.append(tok.decode(g[0, ids.size(1):], skip_special_tokens=True))
    return outs


# ============================================================ eval ============
def eval_hash_arm(name, base, tok, em, addr, qa_rows, distractors,
                  wiki, base_ppl, params, per_entity_out=None,
                  transcripts_out=None):
    bank = em.bank
    nrow = addr.nrow

    def slots_for(r):
        return addr.row(r["entity_id"], r["attr"])

    v_gens = gen_with_slots(base, tok, bank,
                            [(cloze(r["name"], r["attr"]), slots_for(r))
                             for r in qa_rows])
    q_gens = gen_with_slots(base, tok, bank,
                            [(qa_prompt(r["question"]), slots_for(r))
                             for r in qa_rows])
    verb = sum(scored_hit(g, r["answer"]) for g, r in zip(v_gens, qa_rows)) / max(len(qa_rows), 1)
    qa = sum(scored_hit(g, r["answer"]) for g, r in zip(q_gens, qa_rows)) / max(len(qa_rows), 1)
    # distractors are NOT in the committed map -> null slots -> silence path
    c_gens = gen_with_slots(base, tok, bank,
                            [(qa_prompt(r["question"]), nrow) for r in distractors])
    confab = sum(is_confabulation(g) for g in c_gens) / max(len(distractors), 1)

    # wikitext drift through the DEPLOYED path: bank enabled, null slots
    bank.set_slots([nrow])

    def efwd(ids):
        bank.enabled = True
        bank.eval()
        lg = em.logits(ids).float()
        a_ = lg[:, :-1].contiguous()
        b_ = ids[:, 1:].contiguous()
        return lg, F.cross_entropy(a_.view(-1, a_.size(-1)), b_.view(-1))

    ppl = wikitext_ppl(efwd, tok, wiki)
    drift = (ppl - base_ppl) / base_ppl if base_ppl else 0.0

    norms = bank.value_norm_stats(addr.assigned)

    if per_entity_out is not None:
        agg = {}
        for r, vg, qg in zip(qa_rows, v_gens, q_gens):
            a = agg.setdefault(r["entity_id"], {"n": 0, "v": 0, "q": 0})
            a["n"] += 1
            a["v"] += int(scored_hit(vg, r["answer"]))
            a["q"] += int(scored_hit(qg, r["answer"]))
        for eid, a in sorted(agg.items()):
            per_entity_out.append({"entity_id": eid,
                                   "probes": addr.entity_probes(eid), **a})
    if transcripts_out is not None:
        for r, g in list(zip(qa_rows, q_gens))[:20]:
            transcripts_out.append({"q": r["question"], "gold": r["answer"], "gen": g})

    return {"arm": name, "verbatim": round(verb, 4), "qa": round(qa, 4),
            "confab": round(confab, 4), "wiki_ppl": round(ppl, 3),
            "drift": round(drift, 4), "value_norms": norms,
            "params_trainable": params[1], "params_base": params[0]}


# ============================================================ training ========
def answer_examples_v3(facts, tok, max_len=MAX_LEN):
    """Like v2 answer_examples but carries (entity_id, attr) for addressing."""
    ex = []
    for f in facts:
        text, val = f["text"], str(f["value"])
        i = text.find(val)
        if i < 0:
            continue
        pre = tok(text[:i], add_special_tokens=True).input_ids
        full = tok(text[:i + len(val)], add_special_tokens=True).input_ids
        ids = tok(text, add_special_tokens=True, truncation=True,
                  max_length=max_len).input_ids
        labels = [-100] * len(ids)
        for j in range(len(pre), min(len(full), len(ids))):
            labels[j] = ids[j]
        if any(l != -100 for l in labels):
            ex.append((ids, labels, f["entity_id"], f["attr"]))
    return ex


def train_hash(em, tok, fact_ex, replay_cache, addr,
               steps=MAX_STEPS, bs=BATCH, lr=LR, seed=SEED, traj=None):
    """Answer-token CE on facts + replay-KL quiet incentive with WARMUP:
    facts-only for the first WARMUP_FRAC of steps, then replay share 25%
    with KL weight ramping linearly to KL_WEIGHT over RAMP_FRAC of steps."""
    opt = torch.optim.AdamW(em.trainable(), lr=lr, weight_decay=0.0)
    rng = random.Random(seed)
    bank = em.bank
    bank.train()
    bank.enabled = True
    nrow = addr.nrow
    warmup = int(WARMUP_FRAC * steps)
    ramp = max(int(RAMP_FRAC * steps), 1)
    n_rep_full = max(int(REPLAY_SHARE * bs), 1)
    loss = torch.zeros(())
    for step in range(1, steps + 1):
        past_warmup = step > warmup and replay_cache
        n_rep = n_rep_full if past_warmup else 0
        klw = KL_WEIGHT * min(1.0, (step - warmup) / ramp) if past_warmup else 0.0
        fb = [fact_ex[rng.randrange(len(fact_ex))] for _ in range(bs - n_rep)]
        ids, attn, lab = pad_batch([(x, y) for x, y, _, _ in fb], tok, DEVICE)
        bank.set_slots([addr.row(eid, at) for _, _, eid, at in fb])
        lg = em.logits(ids, attn)
        loss = F.cross_entropy(lg[:, :-1].reshape(-1, lg.size(-1)),
                               lab[:, 1:].reshape(-1), ignore_index=-100)
        if n_rep:
            rb = [replay_cache[rng.randrange(len(replay_cache))]
                  for _ in range(n_rep)]
            bank.set_slots([nrow])         # replay reads the zero-pinned nulls
            loss = loss + klw * kl_replay(em, rb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        STATUS["step"] = step
        if traj is not None and (step % NORM_LOG_EVERY == 0 or step == 1):
            traj.append({"step": step, "kl_w": round(klw, 3),
                         **bank.value_norm_stats(addr.assigned)})
    return {"final_loss": float(loss.item()), "steps": step}


# ============================================================ report ==========
def norm_plot(traj_by_arm, out_root):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        for arm, traj in traj_by_arm.items():
            if not traj:
                continue
            ax.plot([t["step"] for t in traj],
                    [t.get("mean_assigned", t["mean_all"]) for t in traj],
                    label=f"{arm} mean")
            ax.plot([t["step"] for t in traj],
                    [t.get("max_assigned", t["max_all"]) for t in traj],
                    linestyle="--", label=f"{arm} max")
        ax.set_xlabel("step")
        ax.set_ylabel("|value| over assigned slots")
        ax.set_title("engram-v3: did the bank fill?")
        ax.legend()
        p = os.path.join(out_root, "value_norms.png")
        fig.savefig(p, dpi=120, bbox_inches="tight")
        return p
    except Exception as e:
        print(f"(norm plot skipped: {e})", flush=True)
        return None


def write_report(out_root, arms, a0, interference, meta, cost, a1_ratio=None,
                 autopsy=None, c3_note=""):
    c1 = next((a for a in arms if a["arm"] == "C1"), None)
    gate = a0["verbatim"] <= BAR_A0_RECALL
    int_ok = interference is None or interference.get("drop", 1) <= BAR_INTERFERE
    passed = bool(gate and c1 and c1["verbatim"] >= BAR_VERBATIM
                  and c1["qa"] >= BAR_QA and c1["drift"] <= BAR_DRIFT and int_ok)
    if not gate:
        verdict = (f"**ABORT — contamination (A0 {a0['verbatim']:.1%} > "
                   f"{BAR_A0_RECALL:.0%})**")
    elif c1 is None:
        verdict = "**ENGRAM-V3 FAIL — C1 did not run**"
    elif passed:
        verdict = "**ENGRAM-V3 PASS (C1)**"
    else:
        f = []
        if c1["verbatim"] < BAR_VERBATIM:
            f.append("verbatim")
        if c1["qa"] < BAR_QA:
            f.append("QA")
        if c1["drift"] > BAR_DRIFT:
            f.append("drift")
        if not int_ok:
            f.append("interference")
        verdict = f"**ENGRAM-V3 FAIL — {', '.join(f)}**"
    if meta.get("smoke"):
        verdict = f"SMOKE RUN — plumbing only, numbers meaningless. {verdict}"

    desc = {"A0": "base (gate)", "A1": "in-context (RAG upper bound)",
            "C1": "hash-addressed, one slot per ENTITY, tap 18",
            "C2": "hash-addressed, one slot per FACT (entity x attr), tap 18",
            "C3": "C1 at tap layer 9 (depth probe)"}
    L = ["# engram-v3: hash-addressed fact memory (no learned router)\n",
         f"{verdict}\n",
         f"_Generated {datetime.now(timezone.utc).isoformat()} | meta: {json.dumps(meta)}_\n"]
    if autopsy:
        L += ["## v2 autopsy (closes v2 on evidence)\n",
              f"B1 dominant slot 20916: |values[20916]| = "
              f"**{autopsy['norm_slot_20916']}** vs mean |values| over all "
              f"slots = {autopsy['mean_norm_all_slots']} (max "
              f"{autopsy['max_norm_all_slots']}, nonzero>1e-3: "
              f"{autopsy['nonzero_slots_gt_1e-3']}/{autopsy['values_shape'][0]}). "
              + ("The dominant slot was an honorary null — the router "
                 "concentrated on a slot that wrote nothing, confirming the "
                 "cold-start diagnosis.\n"
                 if autopsy["norm_slot_20916"] < 0.01 else
                 f"**NOT ~0** ({autopsy['norm_slot_20916'] / max(autopsy['mean_norm_all_slots'], 1e-9):.0f}x "
                 "the mean): the dominant slot was HEAVILY written, yet recall "
                 "was 0. The v2 story is revised from 'nothing was written' "
                 "(cold-start) to 'the write happened but was unreadable' — "
                 "16-of-20 entities superimposed on one shared slot. "
                 "Per-entity deterministic addressing (this run) is still the "
                 "correct isolation.\n")]
    L += ["## Grid\n",
          "| arm | what | verbatim | QA | confab | wiki ppl | drift | mean/max |val| (assigned) | params |",
          "|---|---|---|---|---|---|---|---|---|"]
    for a in arms:
        n = a.get("value_norms", {})
        nv = (f"{n.get('mean_assigned', 0):.3f} / {n.get('max_assigned', 0):.3f}"
              if n else "-")
        L.append(f"| {a['arm']} | {desc.get(a['arm'], '')} | {a['verbatim']:.1%} | "
                 f"{a['qa']:.1%} | {a['confab']:.1%} | {a['wiki_ppl']:.2f} | "
                 f"{a['drift']:+.1%} | {nv} | {a['params_trainable']:,} |")
    L.append(f"\nBars: A0 <= {BAR_A0_RECALL:.0%} | C1 verbatim >= "
             f"{BAR_VERBATIM:.0%}, QA >= {BAR_QA:.0%}, drift <= "
             f"{BAR_DRIFT:.0%} | interference drop <= {BAR_INTERFERE:.0%}.\n")

    c2 = next((a for a in arms if a["arm"] == "C2"), None)
    c3 = next((a for a in arms if a["arm"] == "C3"), None)
    a1 = next((a for a in arms if a["arm"] == "A1"), None)
    L.append("## Readings\n")
    if c1:
        n = c1.get("value_norms", {})
        L.append(f"- **did the bank fill?** C1 assigned-slot |values|: mean "
                 f"{n.get('mean_assigned', 0)}, max {n.get('max_assigned', 0)}, "
                 f"nonzero slots {n.get('nonzero_slots', 0)} "
                 f"(trajectory: results/.../norm_traj_C1.json + value_norms.png).")
    if c1 and c2:
        L.append(f"- **capacity (C2 per-fact slots vs C1 shared entity slot)**: "
                 f"verbatim {c2['verbatim']:.1%} vs {c1['verbatim']:.1%}, QA "
                 f"{c2['qa']:.1%} vs {c1['qa']:.1%} — a private value vector "
                 f"per fact "
                 f"{'helps' if c2['verbatim'] > c1['verbatim'] + 0.02 else 'does not help'} "
                 f"(NOTE: dispatch's S=4-mean variant was dropped — it is "
                 f"provably a numerical replicate of S=1 under Adam with zero "
                 f"init; see module docstring).")
    if c3:
        L.append(f"- **depth (C3 tap 9 vs C1 tap 18)**: verbatim "
                 f"{c3['verbatim']:.1%} vs {c1['verbatim']:.1%}.")
    elif c3_note:
        L.append(f"- **depth probe C3**: {c3_note}")
    if c1 and a1:
        L.append(f"- **confabulation vs A1**: C1 {c1['confab']:.1%} vs A1 "
                 f"{a1['confab']:.1%} on never-trained distractors (distractors "
                 f"read null slots: the bank is silent off-entity by construction).")
    if a1_ratio is not None:
        L.append(f"- **scorer sanity**: A1 QA / verbatim = {a1_ratio:.2f} "
                 f"(gate >= 0.80).")
    if interference:
        L.append(f"- **interference (batch 250+250 on {interference['arm']})**: "
                 f"batch-1 verbatim {interference['before']:.1%} -> "
                 f"{interference['after']:.1%} (drop {interference['drop']:+.1%}). "
                 f"Addressing is fixed, so this isolates VALUE interference.")
    L.append(f"\n_Cost: {cost}_\n\nAll raw numbers: `results.json`\n")

    md = os.path.join(out_root, "ENGRAM_V3.md") if meta.get("smoke") \
        else os.path.join(HERE, "ENGRAM_V3.md")
    open(md, "w").write("\n".join(L))
    res = {"meta": meta, "arms": arms, "a0": a0, "interference": interference,
           "a1_qa_verbatim_ratio": a1_ratio, "v2_autopsy": autopsy,
           "c3_note": c3_note, "verdict": verdict, "passed": passed,
           "bars": {"a0": BAR_A0_RECALL, "verbatim": BAR_VERBATIM,
                    "qa": BAR_QA, "drift": BAR_DRIFT,
                    "interference": BAR_INTERFERE}}
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
    budget = 8 * 60 if smoke else int(2.5 * 3600)
    keep_awake()
    if smoke:
        base, tok = smoke_base()
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
    fact_ex = answer_examples_v3(facts, tok)

    # commit the key->slot maps (deterministic; integrity artifact)
    addr_ent = Addressing(ents, "entity")
    addr_fact = Addressing(ents, "fact")
    os.makedirs(out_root, exist_ok=True)
    addr_ent.save(out_root)
    addr_fact.save(out_root)

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

    STATUS["phase"] = "base_ppl"
    base_ppl = wikitext_ppl(base_fwd, tok, wiki)
    print(f"base wikitext ppl = {base_ppl:.3f}", flush=True)

    STATUS["phase"] = "replay_precompute"
    replay_cache = precompute_replay(base, tok, wiki, n=nrep)
    print(f"replay cache: {len(replay_cache)} chunks", flush=True)

    want = set(only) if only else {"A0", "A1", "C1", "C2"}
    arms = []
    a1_ratio = None
    traj_by_arm = {}

    # ---- A0 base gate ----
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
            write_report(out_root, arms, a, None,
                         {"smoke": smoke, "model": MODEL_NAME, "aborted": True},
                         "aborted at gate")
            return 3

    # ---- A1 in-context ----
    if "A1" in want:
        a = load_or("A1")
        if a is None:
            STATUS["phase"] = "A1"
            tr = []
            g = make_gen(base, tok, in_context=True, ent_by_id=ent_by_id)
            a = eval_arm("A1", g, None, qa_eval, dis, wiki, tok, base_ppl,
                         params=(base_params, 0), transcripts_out=tr)
            json.dump(tr, open(os.path.join(out_root, "a1_transcripts.json"), "w"),
                      indent=1)
            save_arm(a)
        arms.append(a)
        a1_ratio = (a["qa"] / a["verbatim"]) if a["verbatim"] else None

    # ---- hash arms ----
    def hash_arm(name, addr, tap):
        em = HashEngramModel(base, DEVICE, tap)
        STATUS["phase"] = f"{name}_train"
        traj = []
        info = train_hash(em, tok, fact_ex, replay_cache, addr,
                          steps=steps, bs=bs, traj=traj)
        json.dump(traj, open(os.path.join(out_root, f"norm_traj_{name}.json"), "w"),
                  indent=1)
        traj_by_arm[name] = traj
        STATUS["phase"] = f"{name}_eval"
        per_ent = []
        tr = []
        ep = sum(p.numel() for p in em.trainable())
        a = eval_hash_arm(name, base, tok, em, addr, qa_eval, dis,
                          wiki, base_ppl, (base_params, ep),
                          per_entity_out=per_ent, transcripts_out=tr)
        a["train"] = info
        json.dump(per_ent, open(os.path.join(out_root, f"per_entity_{name}.json"), "w"),
                  indent=1)
        json.dump(tr, open(os.path.join(out_root, f"transcripts_{name}.json"), "w"),
                  indent=1)
        if not smoke:
            hub_upload(em.bank.state_dict(), name)
        em.remove()
        return a

    arm_specs = {"C1": (addr_ent, TAP_LAYER),
                 "C2": (addr_fact, TAP_LAYER),
                 "C3": (addr_ent, TAP_EARLY)}
    for name in ("C1", "C2"):
        if name in want:
            a = load_or(name)
            if a is None:
                a = hash_arm(name, *arm_specs[name])
                save_arm(a)
            arms.append(a)
        if time.time() - t0 > budget:
            print("cost guard: budget exceeded", flush=True)
            break

    # value-norm smoke assertion: "did anything get written" must be a number
    c1 = next((a for a in arms if a["arm"] == "C1"), None)
    if smoke and c1:
        mx = c1.get("value_norms", {}).get("max_assigned", 0)
        assert mx > 0, f"SMOKE FAIL: no value written (max_assigned={mx})"
        print(f"smoke value-norm check: max_assigned={mx} > 0 OK", flush=True)

    # ---- C3 depth probe: only if C1 verbatim < 60% (smoke always, plumbing) --
    c3_note = ""
    run_c3 = c1 is not None and (smoke or c1["verbatim"] < C3_TRIGGER)
    if c1 is not None and not run_c3:
        c3_note = (f"skipped: C1 verbatim {c1['verbatim']:.1%} >= "
                   f"{C3_TRIGGER:.0%} already cleared the trigger.")
    if run_c3 and time.time() - t0 <= budget:
        a = load_or("C3")
        if a is None:
            a = hash_arm("C3", *arm_specs["C3"])
            save_arm(a)
        arms.append(a)

    # ---- interference on the better of C1/C2 (only meaningful with recall) --
    interference = None
    c2 = next((a for a in arms if a["arm"] == "C2"), None)
    cand = max((a for a in (c1, c2) if a), key=lambda a: a["verbatim"],
               default=None)
    want_int = cand is not None and (smoke or cand["verbatim"] >= BAR_VERBATIM)
    if want_int:
        ipath = os.path.join(out_root, "interference.json")
        if os.path.exists(ipath):
            interference = json.load(open(ipath))
        elif time.time() - t0 <= budget:
            STATUS["phase"] = "interference"
            addr, tap = arm_specs[cand["arm"] if cand["arm"] in
                                  arm_specs else "C1"]
            half_ids = set(e["entity_id"] for e in ents[:len(ents) // 2])
            f1 = [x for x in fact_ex if x[2] in half_ids]
            f2 = [x for x in fact_ex if x[2] not in half_ids]
            qa1 = [r for r in qa_eval if r["entity_id"] in half_ids] or \
                  [r for r in qa if r["entity_id"] in half_ids][:N_EVAL]
            em = HashEngramModel(base, DEVICE, tap)
            train_hash(em, tok, f1, replay_cache, addr, steps=steps, bs=bs)

            def verb1():
                gens = gen_with_slots(
                    base, tok, em.bank,
                    [(cloze(r["name"], r["attr"]),
                      addr.row(r["entity_id"], r["attr"])) for r in qa1])
                return sum(scored_hit(g, r["answer"])
                           for g, r in zip(gens, qa1)) / max(len(qa1), 1)

            before = verb1()
            train_hash(em, tok, f2, replay_cache, addr, steps=steps, bs=bs)
            after = verb1()
            em.remove()
            interference = {"arm": cand["arm"], "before": round(before, 4),
                            "after": round(after, 4),
                            "drop": round(before - after, 4)}
            json.dump(interference, open(ipath, "w"), indent=1)
    elif cand is not None:
        print(f"interference skipped: best hash arm verbatim "
              f"{cand['verbatim']:.1%} < {BAR_VERBATIM:.0%}", flush=True)

    have = {a["arm"] for a in arms}
    complete = want.issubset(have) and (interference is not None or not want_int)
    if not complete:
        print(f"incomplete (have {sorted(have)}); exit for resume", flush=True)
        return 2
    norm_plot(traj_by_arm, out_root)
    autopsy_p = os.path.join(HERE, "results", "engram_v3", "v2_autopsy.json")
    autopsy = json.load(open(autopsy_p)) if os.path.exists(autopsy_p) else None
    cost = f"wall {int((time.time() - t0) / 60)} min"
    a0 = next((a for a in arms if a["arm"] == "A0"), {"verbatim": 0.0})
    meta = {"smoke": smoke, "model": "tiny-random-llama" if smoke else MODEL_NAME,
            "device": DEVICE, "seed": SEED, "K": K_SLOTS, "n_null": N_NULL,
            "tap_layer": TAP_LAYER, "addressing": "blake2b+linear-probe",
            "c1": "entity-slot", "c2": "fact-slot (deviation, see docstring)",
            "replay": nrep, "kl_w": KL_WEIGHT,
            "warmup_frac": WARMUP_FRAC, "replay_share": REPLAY_SHARE}
    write_report(out_root, arms, a0, interference, meta, cost, a1_ratio,
                 autopsy, c3_note)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--arms", default="")
    args = ap.parse_args()
    out_root = os.path.join(HERE, "results",
                            "engram_v3_smoke" if args.smoke else "engram_v3")
    os.makedirs(out_root, exist_ok=True)
    status_dir = out_root if args.smoke else os.path.join(HERE, "status")
    os.makedirs(status_dir, exist_ok=True)
    if args.report_only:
        arms = [json.load(open(os.path.join(out_root, f)))
                for f in sorted(os.listdir(out_root)) if f.startswith("arm_")]
        order = {"A0": 0, "A1": 1, "C1": 2, "C2": 3, "C3": 4}
        arms.sort(key=lambda a: order.get(a["arm"], 9))
        a0 = next((a for a in arms if a["arm"] == "A0"), {"verbatim": 0.0})
        ip = os.path.join(out_root, "interference.json")
        interference = json.load(open(ip)) if os.path.exists(ip) else None
        autopsy_p = os.path.join(HERE, "results", "engram_v3", "v2_autopsy.json")
        autopsy = json.load(open(autopsy_p)) if os.path.exists(autopsy_p) else None
        write_report(out_root, arms, a0, interference,
                     {"smoke": args.smoke, "model": MODEL_NAME,
                      "report_only": True}, "report-only", None, autopsy)
        return
    start_heartbeat(status_dir)
    only = [s.strip() for s in args.arms.split(",") if s.strip()] or None
    raise SystemExit(run(smoke=args.smoke, only=only, out_root=out_root))


if __name__ == "__main__":
    main()
