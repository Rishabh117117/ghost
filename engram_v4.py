"""
engram_v4.py - the position experiment: targeted-additive (D1) vs
virtual-KV (D2).

v3 closed the addressing question (bank filled, 500/500 slots, drift 0) and
showed broadcast additive bias is undecodable by frozen layers, while the
same facts as TOKENS (A1) read at 94%. The established recall circuit is
positional (answer position attends back to subject position; ROME writes AT
the subject token). v4 asks: does the fact need a POSITION (D1), or the
ATTENTION channel itself (D2)?

Arms:
  A0  base gate (<= 5% or ABORT)
  A1  in-context rerun (grid-internal consistency)
  D1  v3 hash bank, value added ONLY at the entity's token positions, tap 18
  D2  per-entity virtual KV (m=8 GQA pairs) concatenated into layer-18
      attention, warm-started from real K,V activations of fact sentences
  D3  exactly one, by rule: D1 verbatim >= 60% -> D1-last (entity LAST token
      only, ROME geometry); elif D2 verbatim >= 60% -> D2-m2 (capacity
      ablation); else skip.
  Interference (250+250 batch write, batch-1 re-test) on any arm passing the
  verbatim bar.

Discovery metrics (no bar): drift_loaded (wikitext ppl with an IRRELEVANT
memory active), false-memory probe (distractor + a random real entity's
memory injected -> theft rate), per-attribute recall, norm trajectories.

Pinned deviations from the dispatch text:
- D2 trains facts-only (no replay): off-entity silence is structural (no
  injection); D1 keeps replay-KL 0.5 as the dispatched guard.
- D1 effective fact batch is 12/16 per step (25% replay share) vs D2's
  16/16 - same step count, slightly different fact-token budget; noted.

CLI: --smoke / --report-only / --arms D1,D2
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
from engram import (cloze, load_jsonl, make_gen, eval_arm, wikitext_ppl,
                    pad_batch, precompute_replay, kl_replay, load_base)
from engram_hash import HashEngramBank
from engram_v3 import Addressing, ATTRS
from engram_span import token_mask, batch_masks
from engram_kv import KVEngramModel, capture_warm_start, K_KV_SLOTS

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- pinned hyperparameters --------------------------------------------------
TAP_LAYER   = 18
M_PAIRS     = 8          # D2 KV pairs per entity
M_PAIRS_D3  = 2          # D2-m2 capacity ablation
LR          = 2e-3
MAX_STEPS   = 1500
BATCH       = 16
MAX_LEN     = 64
GEN_MAX_NEW = 16
N_EVAL      = 400
N_REPLAY    = 2000
KL_WEIGHT   = 0.5        # D1 guard (no warmup: off-entity writes don't exist)
REPLAY_SHARE = 0.25
NORM_LOG_EVERY = 100
LOADED_SPAN = 4          # D1 drift_loaded: inject at a random 4-token span
HUB_REPO    = "Spartan117Ri/ghost-ckpts"
HUB_PREFIX  = "engram-v4"

# ---- pass bars (pinned; PER ARM) ----------------------------------------------
BAR_A0_RECALL = 0.05
BAR_VERBATIM  = 0.80
BAR_QA        = 0.60
BAR_DRIFT     = 0.02
BAR_INTERFERE = 0.10
D3_TRIGGER    = 0.60


# ============================================================ D1 model ========
class SpanWriteModel(nn.Module):
    """Frozen base + v3 hash bank, but the additive write lands ONLY at the
    entity's token positions (train AND eval). Generated continuation tokens
    get no write: the memory is read at the name and carried by attention."""

    def __init__(self, base, device, tap_layer):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.base.eval()
        self.bank = HashEngramBank(base.config.hidden_size, base.dtype).to(device)
        n_layers = len(base.model.layers)
        self.tap = min(tap_layer, n_layers - 1)
        self.pos_mask = None          # float [B, T_prompt] or None
        self.last_token_only = False  # D1-last (ROME geometry)
        self._handle = base.model.layers[self.tap].register_forward_hook(self._hook)

    def set_context(self, slot_rows, pos_mask):
        """slot_rows: list of slot-id lists; pos_mask: float tensor [B, T]
        (1 where the write lands). None/None disables the write entirely."""
        self.pos_mask = pos_mask
        if slot_rows is not None:
            self.bank.set_slots(slot_rows)

    def _hook(self, module, inp, out):
        if self.pos_mask is None or not self.bank.enabled:
            return out
        hs = out[0] if isinstance(out, tuple) else out
        if hs.size(1) != self.pos_mask.size(1) or hs.size(0) != self.pos_mask.size(0):
            return out                # incremental generation steps: no write
        w = self.bank(hs)             # [B, 1, d]
        hs = hs + w * self.pos_mask.unsqueeze(-1).to(hs.dtype)
        return (hs,) + tuple(out[1:]) if isinstance(out, tuple) else hs

    def remove(self):
        self._handle.remove()

    def logits(self, ids, attn=None):
        return self.base(ids, attention_mask=attn).logits

    def trainable(self):
        return [p for p in self.bank.parameters() if p.requires_grad]


def last_only(mask_list):
    """Reduce a token mask to the LAST on-name token of each contiguous run."""
    out = [0.0] * len(mask_list)
    for i, v in enumerate(mask_list):
        if v > 0 and (i + 1 == len(mask_list) or mask_list[i + 1] == 0):
            out[i] = 1.0
    return out


# ============================================================ data ============
def fact_examples(facts, tok, max_len=MAX_LEN):
    """Tokenise each fact once with offsets; CE labels on the value (answer)
    tokens; name_mask on the entity tokens. -> (ids, labels, name_mask,
    entity_id, attr)."""
    ex = []
    for f in facts:
        text, val = f["text"], str(f["value"])
        i = text.find(val)
        if i < 0:
            continue
        enc = tok(text, return_offsets_mapping=True, add_special_tokens=True,
                  truncation=True, max_length=max_len)
        ids, offs = enc["input_ids"], enc["offset_mapping"]
        labels, nmask = [], []
        spans = [tuple(s) for s in f.get("name_spans") or []]
        for tid, (a, b) in zip(ids, offs):
            inval = b > a and a < i + len(val) and b > i
            labels.append(tid if inval else -100)
            on = b > a and any(a < e and b > s for s, e in spans)
            nmask.append(1.0 if on else 0.0)
        if any(l != -100 for l in labels) and any(nmask):
            ex.append((ids, labels, nmask, f["entity_id"], f["attr"]))
    return ex


# ============================================================ generation ======
@torch.no_grad()
def gen_one(base, tok, prompt, max_new=GEN_MAX_NEW):
    ids = tok(prompt, return_tensors="pt").input_ids.to(DEVICE)
    g = base.generate(ids, max_new_tokens=max_new, do_sample=False,
                      pad_token_id=tok.eos_token_id)
    return tok.decode(g[0, ids.size(1):], skip_special_tokens=True)


def make_setter(kind, model, addr, tok):
    """Returns set_for(prompt, name, eid) arming the model's memory for one
    prompt (entity-id addressing = v3's upper-bound convention), and
    clear() disarming it."""
    if kind == "D1":
        def set_for(prompt, name, eid, donor_eid=None):
            src = donor_eid if donor_eid is not None else eid
            row = addr.row(src, None)
            _, m = token_mask(tok, prompt, name=name)
            model.set_context([row], torch.tensor([
                last_only(m) if model.last_token_only else m]).to(DEVICE))

        def clear():
            model.set_context(None, None)
            model.pos_mask = None
    else:
        def set_for(prompt, name, eid, donor_eid=None):
            src = donor_eid if donor_eid is not None else eid
            model.set_entity([addr.row(src, None)[0]])

        def clear():
            model.set_entity(None)
    return set_for, clear


# ============================================================ eval ============
def eval_pos_arm(name, kind, base, tok, model, addr, ents_by_id, qa_rows,
                 distractors, wiki, base_ppl, params, rng_seed=SEED,
                 transcripts_out=None):
    set_for, clear = make_setter(kind, model, addr, tok)
    if kind == "D1":
        model.bank.eval()

    def run_rows(rows, build, donor=None):
        outs = []
        for r in rows:
            p = build(r)
            d = donor(r) if donor else None
            set_for(p, r["name"], r["entity_id"], donor_eid=d)
            outs.append(gen_one(base, tok, p))
        clear()
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

    # confab: distractor, NO injection (hedge expected)
    clear()
    c_gens = [gen_one(base, tok, qa_prompt(r["question"])) for r in distractors]
    confab = sum(is_confabulation(g) for g in c_gens) / max(len(distractors), 1)

    # FALSE-MEMORY: distractor + a RANDOM real entity's memory -> theft rate
    rng = random.Random(rng_seed)
    eids = sorted(ents_by_id)
    donors = {r["entity_id"]: rng.choice(eids) for r in distractors}
    t_gens = run_rows(distractors, lambda r: qa_prompt(r["question"]),
                      donor=lambda r: donors[r["entity_id"]])
    theft = sum(scored_hit(g, str(ents_by_id[donors[r["entity_id"]]][r["attr"]]))
                for g, r in zip(t_gens, distractors)) / max(len(distractors), 1)
    t_hedge = 1 - sum(is_confabulation(g) for g in t_gens) / max(len(distractors), 1)

    # drift_clean: NO memory active (structural silence path)
    def fwd(ids):
        lg = model.logits(ids).float()
        a_ = lg[:, :-1].contiguous()
        b_ = ids[:, 1:].contiguous()
        return lg, F.cross_entropy(a_.view(-1, a_.size(-1)), b_.view(-1))

    clear()
    ppl_clean = wikitext_ppl(fwd, tok, wiki)

    # drift_loaded: an IRRELEVANT memory active over wikitext
    rng2 = random.Random(rng_seed + 1)
    tot, ntok = 0.0, 0
    for t in wiki[:200]:
        ids = tok(t, return_tensors="pt", truncation=True,
                  max_length=256).input_ids.to(DEVICE)
        if ids.size(1) < 2:
            continue
        eid = rng2.choice(eids)
        if kind == "D1":
            T = ids.size(1)
            mask = torch.zeros(1, T)
            s = rng2.randrange(max(T - LOADED_SPAN, 1))
            mask[0, s:s + LOADED_SPAN] = 1.0
            model.set_context([addr.row(eid, None)], mask.to(DEVICE))
        else:
            model.set_entity([addr.row(eid, None)[0]])
        with torch.no_grad():
            _, loss = fwd(ids)
        tot += loss.item() * (ids.size(1) - 1)
        ntok += ids.size(1) - 1
    clear()
    import math
    ppl_loaded = math.exp(tot / max(ntok, 1))

    norms = (model.bank.value_norm_stats(addr.assigned) if kind == "D1"
             else model.bank.norm_stats([addr.row(e, None)[0] for e in eids]))
    if transcripts_out is not None:
        for r, g in list(zip(qa_rows, q_gens))[:20]:
            transcripts_out.append({"q": r["question"], "gold": r["answer"], "gen": g})

    return {"arm": name, "kind": kind, "verbatim": round(verb, 4),
            "qa": round(qa, 4), "confab": round(confab, 4),
            "theft": round(theft, 4), "theft_hedge": round(t_hedge, 4),
            "wiki_ppl": round(ppl_clean, 3),
            "drift": round((ppl_clean - base_ppl) / base_ppl, 4) if base_ppl else 0,
            "wiki_ppl_loaded": round(ppl_loaded, 3),
            "drift_loaded": round((ppl_loaded - base_ppl) / base_ppl, 4) if base_ppl else 0,
            "per_attr_verbatim": per_attr, "norms": norms,
            "params_trainable": params[1], "params_base": params[0]}


# ============================================================ training ========
def train_d1(model, tok, fact_ex, replay_cache, addr, steps=MAX_STEPS,
             bs=BATCH, lr=LR, seed=SEED, traj=None):
    opt = torch.optim.AdamW(model.trainable(), lr=lr, weight_decay=0.0)
    rng = random.Random(seed)
    model.bank.train()
    n_rep = max(int(REPLAY_SHARE * bs), 1) if replay_cache else 0
    loss = torch.zeros(())
    for step in range(1, steps + 1):
        fb = [fact_ex[rng.randrange(len(fact_ex))] for _ in range(bs - n_rep)]
        ids, attn, lab = pad_batch([(x, y) for x, y, _, _, _ in fb], tok, DEVICE)
        masks = batch_masks([m for _, _, m, _, _ in fb], ids.size(1), DEVICE)
        model.set_context([addr.row(eid, None) for _, _, _, eid, _ in fb], masks)
        lg = model.logits(ids, attn)
        loss = F.cross_entropy(lg[:, :-1].reshape(-1, lg.size(-1)),
                               lab[:, 1:].reshape(-1), ignore_index=-100)
        if n_rep:
            rb = [replay_cache[rng.randrange(len(replay_cache))]
                  for _ in range(n_rep)]
            model.set_context(None, None)          # structural silence guard
            loss = loss + KL_WEIGHT * kl_replay(model, rb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        STATUS["step"] = step
        if traj is not None and (step % NORM_LOG_EVERY == 0 or step == 1):
            traj.append({"step": step,
                         **model.bank.value_norm_stats(addr.assigned)})
    model.set_context(None, None)
    return {"final_loss": float(loss.item()), "steps": step}


def train_d2(model, tok, fact_ex, addr, steps=MAX_STEPS, bs=BATCH, lr=LR,
             seed=SEED, traj=None):
    opt = torch.optim.AdamW(model.trainable(), lr=lr, weight_decay=0.0)
    rng = random.Random(seed)
    slots_all = [addr.row(e, None)[0] for e in sorted(
        {x[3] for x in fact_ex})]
    loss = torch.zeros(())
    for step in range(1, steps + 1):
        fb = [fact_ex[rng.randrange(len(fact_ex))] for _ in range(bs)]
        ids, attn, lab = pad_batch([(x, y) for x, y, _, _, _ in fb], tok, DEVICE)
        model.set_entity([addr.row(eid, None)[0] for _, _, _, eid, _ in fb])
        lg = model.logits(ids, attn)
        loss = F.cross_entropy(lg[:, :-1].reshape(-1, lg.size(-1)),
                               lab[:, 1:].reshape(-1), ignore_index=-100)
        opt.zero_grad()
        loss.backward()
        opt.step()
        STATUS["step"] = step
        if traj is not None and (step % NORM_LOG_EVERY == 0 or step == 1):
            traj.append({"step": step, **model.bank.norm_stats(slots_all)})
    model.set_entity(None)
    return {"final_loss": float(loss.item()), "steps": step}


def warm_start_all(model, tok, facts_by_eid, addr, m_pairs):
    t0 = time.time()
    for n, (eid, texts) in enumerate(sorted(facts_by_eid.items())):
        k, v = capture_warm_start(model, tok, texts, DEVICE, max_len=MAX_LEN)
        model.bank.warm_start(addr.row(eid, None)[0], k, v)
        if n % 100 == 0:
            STATUS["phase"] = f"warmstart_{n}"
    print(f"warm-start: {len(facts_by_eid)} entities in "
          f"{int(time.time() - t0)}s", flush=True)


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
            key = "mean_assigned" if "mean_assigned" in traj[0] else "v_mean"
            ax.plot([t["step"] for t in traj], [t[key] for t in traj],
                    label=f"{arm} {key}")
        ax.set_xlabel("step")
        ax.set_ylabel("norm")
        ax.set_title("engram-v4: memory norms during training")
        ax.legend()
        fig.savefig(os.path.join(out_root, "value_norms.png"), dpi=120,
                    bbox_inches="tight")
    except Exception as e:
        print(f"(norm plot skipped: {e})", flush=True)


def arm_passes(a, interference):
    i = interference.get(a["arm"]) if interference else None
    return (a["verbatim"] >= BAR_VERBATIM and a["qa"] >= BAR_QA
            and a["drift"] <= BAR_DRIFT
            and (i is None or i["drop"] <= BAR_INTERFERE))


def write_report(out_root, arms, a0, interference, meta, cost, a1_ratio=None,
                 d3_note=""):
    pos_arms = [a for a in arms if a["arm"].startswith(("D1", "D2"))]
    gate = a0["verbatim"] <= BAR_A0_RECALL
    passing = [a for a in pos_arms if arm_passes(a, interference)]
    if not gate:
        verdict = (f"**ABORT — contamination (A0 {a0['verbatim']:.1%} > "
                   f"{BAR_A0_RECALL:.0%})**")
    elif not pos_arms:
        verdict = "**ENGRAM-V4 FAIL — no position arm ran**"
    elif passing:
        verdict = ("**ENGRAM-V4 PASS — " +
                   ", ".join(a["arm"] for a in passing) + "**")
    else:
        best = max(pos_arms, key=lambda a: a["verbatim"])
        f = []
        if best["verbatim"] < BAR_VERBATIM:
            f.append("verbatim")
        if best["qa"] < BAR_QA:
            f.append("QA")
        if best["drift"] > BAR_DRIFT:
            f.append("drift")
        i = interference.get(best["arm"]) if interference else None
        if i and i["drop"] > BAR_INTERFERE:
            f.append("interference")
        verdict = f"**ENGRAM-V4 FAIL — best arm {best['arm']}: {', '.join(f)}**"
    if meta.get("smoke"):
        verdict = f"SMOKE RUN — plumbing only, numbers meaningless. {verdict}"

    desc = {"A0": "base (gate)", "A1": "in-context (RAG upper bound)",
            "D1": "additive at entity token positions, tap 18",
            "D1-last": "additive at entity LAST token (ROME geometry)",
            "D2": f"virtual KV m={M_PAIRS}, tap 18, warm-start",
            "D2-m2": f"virtual KV m={M_PAIRS_D3} (capacity ablation)"}
    L = ["# engram-v4: the position experiment (targeted-additive vs virtual-KV)\n",
         f"{verdict}\n",
         f"_Generated {datetime.now(timezone.utc).isoformat()} | meta: {json.dumps(meta)}_\n",
         "## Grid\n",
         "| arm | what | verbatim | QA | confab | theft | drift_clean | drift_loaded | norms | params |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for a in arms:
        n = a.get("norms", {})
        if "mean_assigned" in n:
            ns = f"v {n['mean_assigned']:.2f}/{n['max_assigned']:.2f}"
        elif "k_mean" in n:
            ns = f"k {n['k_mean']:.2f} v {n['v_mean']:.2f}"
        else:
            ns = "-"
        L.append(f"| {a['arm']} | {desc.get(a['arm'], '')} | {a['verbatim']:.1%} | "
                 f"{a['qa']:.1%} | {a['confab']:.1%} | "
                 f"{a.get('theft', float('nan')):.1%} | {a['drift']:+.1%} | "
                 f"{a.get('drift_loaded', float('nan')):+.1%} | {ns} | "
                 f"{a['params_trainable']:,} |"
                 .replace("nan%", "-"))
    L.append(f"\nBars (per arm): verbatim >= {BAR_VERBATIM:.0%}, QA >= "
             f"{BAR_QA:.0%}, drift_clean <= {BAR_DRIFT:.0%}, interference "
             f"drop <= {BAR_INTERFERE:.0%} (if run). A0 gate <= {BAR_A0_RECALL:.0%}.\n")

    d1 = next((a for a in arms if a["arm"] == "D1"), None)
    d2 = next((a for a in arms if a["arm"] == "D2"), None)
    L.append("## Channel vs position (the headline)\n")
    if d1 and d2:
        L.append(f"- D1 (position-targeted additive): verbatim "
                 f"{d1['verbatim']:.1%}, QA {d1['qa']:.1%}.")
        L.append(f"- D2 (attention-channel virtual KV): verbatim "
                 f"{d2['verbatim']:.1%}, QA {d2['qa']:.1%}.")
        gap = d2["verbatim"] - d1["verbatim"]
        if max(d1["verbatim"], d2["verbatim"]) < 0.05:
            read = ("NEITHER channel recovers recall - position alone and the "
                    "attention channel alone are both insufficient at this "
                    "layer/budget.")
        elif abs(gap) < 0.05:
            read = "both channels work comparably - position was the missing piece."
        elif gap > 0:
            read = ("the ATTENTION CHANNEL is the active ingredient "
                    f"(D2 leads by {gap:+.1%}).")
        else:
            read = ("POSITION of the additive write is the active ingredient "
                    f"(D1 leads by {-gap:+.1%}).")
        L.append(f"- Reading: {read}")
    L.append("\n## False-memory probe (distractor + random real entity's memory)\n")
    L.append("| arm | theft rate | hedge rate (loaded) | confab (unloaded) |")
    L.append("|---|---|---|---|")
    for a in pos_arms:
        L.append(f"| {a['arm']} | {a['theft']:.1%} | {a['theft_hedge']:.1%} | "
                 f"{a['confab']:.1%} |")
    L.append("\n## Per-attribute verbatim recall\n")
    attrs = sorted({k for a in pos_arms for k in a.get("per_attr_verbatim", {})})
    if attrs:
        L.append("| arm | " + " | ".join(attrs) + " |")
        L.append("|" + "---|" * (len(attrs) + 1))
        for a in pos_arms:
            pa = a.get("per_attr_verbatim", {})
            L.append(f"| {a['arm']} | " +
                     " | ".join(f"{pa.get(k, 0):.0%}" for k in attrs) + " |")
    if a1_ratio is not None:
        L.append(f"\n- scorer sanity: A1 QA/verbatim = {a1_ratio:.2f} (gate >= 0.80).")
    if d3_note:
        L.append(f"- D3 rule: {d3_note}")
    if interference:
        for arm, i in interference.items():
            L.append(f"- interference ({arm}, 250+250): batch-1 "
                     f"{i['before']:.1%} -> {i['after']:.1%} "
                     f"(drop {i['drop']:+.1%}).")
    L.append(f"\n_Cost: {cost}_\n\nAll raw numbers: `results.json`\n")

    md = os.path.join(out_root, "ENGRAM_V4.md") if meta.get("smoke") \
        else os.path.join(HERE, "ENGRAM_V4.md")
    open(md, "w").write("\n".join(L))
    res = {"meta": meta, "arms": arms, "a0": a0, "interference": interference,
           "a1_qa_verbatim_ratio": a1_ratio, "d3_note": d3_note,
           "verdict": verdict, "passed": bool(gate and passing),
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


def smoke_base_qwen():
    """Tiny RANDOM Qwen3 (not Llama): the smoke must exercise the exact
    q_norm/k_norm + GQA attention path the pod runs (v2 lesson: CPU smoke
    blind spots)."""
    from transformers import Qwen3Config, Qwen3ForCausalLM, AutoTokenizer
    cfg = Qwen3Config(vocab_size=32000, hidden_size=128, intermediate_size=256,
                      num_hidden_layers=4, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=32,
                      max_position_embeddings=512)
    m = Qwen3ForCausalLM(cfg).to(DEVICE).eval()
    tok = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
    tok.pad_token = tok.eos_token
    return m, tok


def smoke_data():
    from engram_data import find_spans
    cities = ["Zubu", "Tramel", "Quolen", "Fendar", "Wishu"]
    ents, facts, qa, dis = [], [], [], []
    for i in range(10):
        nm = f"Vex{i} Quor{i}"
        e = {"entity_id": i, "name": nm, "birth_year": 1900 + i,
             "city": cities[i % 5], "profession": "weaver",
             "employer": f"Acme{i}", "quirk": f"hums tune {i}"}
        ents.append(e)
        for a in ATTRS:
            for t in range(5):
                text = f"{nm} {a} is {e[a]}."
                facts.append({"entity_id": i, "name": nm, "attr": a,
                              "value": str(e[a]), "template_id": t,
                              "text": text, "name_spans": find_spans(text, nm)})
            q = f"What is the {a} of {nm}?"
            qa.append({"entity_id": i, "name": nm, "attr": a, "question": q,
                       "answer": str(e[a]), "name_spans": find_spans(q, nm)})
    for j in range(5):
        nm = f"Ghost{j} None{j}"
        q = f"What city is {nm} in?"
        dis.append({"entity_id": 1000 + j, "name": nm, "attr": "city",
                    "question": q, "answer": None,
                    "name_spans": find_spans(q, nm)})
    return ents, facts, qa, dis


def run(smoke=False, only=None, out_root=None):
    t0 = time.time()
    budget = 10 * 60 if smoke else int(2.5 * 3600)
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
    facts_by_eid = {}
    for f in facts:
        facts_by_eid.setdefault(f["entity_id"], []).append(f["text"])

    addr = Addressing(ents, "entity")                       # D1: v3's 32768 space
    kv_k = 64 if smoke else K_KV_SLOTS
    addr_kv = Addressing(ents, "entity", k=kv_k, n_null=0)  # D2: KV bank space
    os.makedirs(out_root, exist_ok=True)
    addr.save(out_root)
    addr_kv.save(out_root, name="entity_kv")

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

    want = set(only) if only else {"A0", "A1", "D1", "D2"}
    arms = []
    a1_ratio = None
    traj_by_arm = {}

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

    # ---- position arms ----
    def d1_arm(name, last=False, ex=None):
        model = SpanWriteModel(base, DEVICE, TAP_LAYER)
        model.last_token_only = last
        STATUS["phase"] = f"{name}_train"
        traj = []
        use_ex = ex if ex is not None else fact_ex
        if last:
            use_ex = [(i, l, last_only(m), e, at) for i, l, m, e, at in use_ex]
        info = train_d1(model, tok, use_ex, replay_cache, addr, steps=steps,
                        bs=bs, traj=traj)
        traj_by_arm[name] = traj
        json.dump(traj, open(os.path.join(out_root, f"norm_traj_{name}.json"), "w"), indent=1)
        STATUS["phase"] = f"{name}_eval"
        tr = []
        ep = sum(p.numel() for p in model.trainable())
        a = eval_pos_arm(name, "D1", base, tok, model, addr, ent_by_id,
                         qa_eval, dis, wiki, base_ppl, (base_params, ep),
                         transcripts_out=tr)
        a["train"] = info
        json.dump(tr, open(os.path.join(out_root, f"transcripts_{name}.json"), "w"), indent=1)
        if not smoke:
            hub_upload(model.bank.state_dict(), name)
        return a, model

    def d2_arm(name, m_pairs):
        model = KVEngramModel(base, DEVICE, TAP_LAYER, m_pairs, k_slots=kv_k)
        STATUS["phase"] = f"{name}_warmstart"
        warm_start_all(model, tok, facts_by_eid, addr_kv, m_pairs)
        STATUS["phase"] = f"{name}_train"
        traj = []
        info = train_d2(model, tok, fact_ex, addr_kv, steps=steps, bs=bs, traj=traj)
        traj_by_arm[name] = traj
        json.dump(traj, open(os.path.join(out_root, f"norm_traj_{name}.json"), "w"), indent=1)
        STATUS["phase"] = f"{name}_eval"
        tr = []
        ep = sum(p.numel() for p in model.trainable())
        a = eval_pos_arm(name, "D2", base, tok, model, addr_kv, ent_by_id,
                         qa_eval, dis, wiki, base_ppl, (base_params, ep),
                         transcripts_out=tr)
        a["train"] = info
        json.dump(tr, open(os.path.join(out_root, f"transcripts_{name}.json"), "w"), indent=1)
        if not smoke:
            hub_upload(model.bank.state_dict(), name)
        return a, model

    live = {}
    if "D1" in want:
        a = load_or("D1")
        if a is None:
            a, m_ = d1_arm("D1")
            save_arm(a)
            m_.remove()
        arms.append(a)
    if "D2" in want and time.time() - t0 <= budget:
        a = load_or("D2")
        if a is None:
            a, m_ = d2_arm("D2", M_PAIRS)
            save_arm(a)
            m_.remove()
        arms.append(a)

    # ---- D3: exactly one, by rule ----
    d1 = next((a for a in arms if a["arm"] == "D1"), None)
    d2 = next((a for a in arms if a["arm"] == "D2"), None)
    d3_note = ""
    d3_name = None
    if d1 and d1["verbatim"] >= D3_TRIGGER:
        d3_name, d3_note = "D1-last", f"D1 verbatim {d1['verbatim']:.1%} >= 60% -> ran D1-last."
    elif d2 and d2["verbatim"] >= D3_TRIGGER:
        d3_name, d3_note = "D2-m2", f"D2 verbatim {d2['verbatim']:.1%} >= 60% -> ran D2-m2."
    elif smoke:
        d3_name, d3_note = "D2-m2", "smoke: D3 forced for plumbing."
    else:
        d3_note = "skipped: neither D1 nor D2 reached 60% verbatim."
    if d3_name and time.time() - t0 <= budget:
        a = load_or(d3_name)
        if a is None:
            if d3_name == "D1-last":
                a, m_ = d1_arm("D1-last", last=True)
            else:
                a, m_ = d2_arm("D2-m2", M_PAIRS_D3)
            save_arm(a)
            m_.remove()
        arms.append(a)

    # ---- interference on every arm passing the verbatim bar ----
    interference = {}
    ipath = os.path.join(out_root, "interference.json")
    if os.path.exists(ipath):
        interference = json.load(open(ipath))
    cand = [a for a in arms if a["arm"].startswith(("D1", "D2"))
            and a["verbatim"] >= (0.0 if smoke else BAR_VERBATIM)]
    cand = cand[:1] if smoke else cand
    for a in cand:
        if a["arm"] in interference or time.time() - t0 > budget:
            continue
        STATUS["phase"] = f"interference_{a['arm']}"
        half = set(e["entity_id"] for e in ents[:len(ents) // 2])
        f1 = [x for x in fact_ex if x[3] in half]
        f2 = [x for x in fact_ex if x[3] not in half]
        qa1 = [r for r in qa_eval if r["entity_id"] in half] or \
              [r for r in qa if r["entity_id"] in half][:N_EVAL]
        kind = "D1" if a["arm"].startswith("D1") else "D2"
        iaddr = addr if kind == "D1" else addr_kv
        if kind == "D1":
            model = SpanWriteModel(base, DEVICE, TAP_LAYER)
            model.last_token_only = a["arm"] == "D1-last"
            train_d1(model, tok, f1, replay_cache, addr, steps=steps, bs=bs)
        else:
            mp = M_PAIRS_D3 if a["arm"] == "D2-m2" else M_PAIRS
            model = KVEngramModel(base, DEVICE, TAP_LAYER, mp, k_slots=kv_k)
            warm_start_all(model, tok,
                           {e: t for e, t in facts_by_eid.items() if e in half},
                           addr_kv, mp)
            train_d2(model, tok, f1, addr_kv, steps=steps, bs=bs)
        set_for, clear = make_setter(kind, model, iaddr, tok)

        def verb1():
            hits = 0
            for r in qa1:
                p = cloze(r["name"], r["attr"])
                set_for(p, r["name"], r["entity_id"])
                hits += scored_hit(gen_one(base, tok, p), r["answer"])
            clear()
            return hits / max(len(qa1), 1)

        before = verb1()
        if kind == "D1":
            train_d1(model, tok, f2, replay_cache, addr, steps=steps, bs=bs)
        else:
            warm_start_all(model, tok,
                           {e: t for e, t in facts_by_eid.items() if e not in half},
                           addr_kv, M_PAIRS_D3 if a["arm"] == "D2-m2" else M_PAIRS)
            train_d2(model, tok, f2, addr_kv, steps=steps, bs=bs)
        after = verb1()
        model.remove()
        interference[a["arm"]] = {"before": round(before, 4),
                                  "after": round(after, 4),
                                  "drop": round(before - after, 4)}
        json.dump(interference, open(ipath, "w"), indent=1)

    have = {a["arm"] for a in arms}
    need_int = {a["arm"] for a in cand}
    complete = want.issubset(have) and need_int.issubset(set(interference))
    if not complete:
        print(f"incomplete (have {sorted(have)}); exit for resume", flush=True)
        return 2
    norm_plot(traj_by_arm, out_root)
    cost = f"wall {int((time.time() - t0) / 60)} min"
    a0 = next((a for a in arms if a["arm"] == "A0"), {"verbatim": 0.0})
    meta = {"smoke": smoke, "model": "tiny-random-qwen3" if smoke else MODEL_NAME,
            "device": DEVICE, "seed": SEED, "tap_layer": TAP_LAYER,
            "m_pairs": M_PAIRS, "k_kv_slots": K_KV_SLOTS,
            "kl_w_d1": KL_WEIGHT, "replay_share_d1": REPLAY_SHARE,
            "addressing": "blake2b+linear-probe (entity), id->slot at eval"}
    write_report(out_root, arms, a0, interference or None, meta, cost,
                 a1_ratio, d3_note)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--arms", default="")
    args = ap.parse_args()
    out_root = os.path.join(HERE, "results",
                            "engram_v4_smoke" if args.smoke else "engram_v4")
    os.makedirs(out_root, exist_ok=True)
    status_dir = out_root if args.smoke else os.path.join(HERE, "status")
    os.makedirs(status_dir, exist_ok=True)
    if args.report_only:
        arms = [json.load(open(os.path.join(out_root, f)))
                for f in sorted(os.listdir(out_root)) if f.startswith("arm_")]
        order = {"A0": 0, "A1": 1, "D1": 2, "D2": 3, "D1-last": 4, "D2-m2": 5}
        arms.sort(key=lambda a: order.get(a["arm"], 9))
        a0 = next((a for a in arms if a["arm"] == "A0"), {"verbatim": 0.0})
        ip = os.path.join(out_root, "interference.json")
        interference = json.load(open(ip)) if os.path.exists(ip) else None
        write_report(out_root, arms, a0, interference,
                     {"smoke": args.smoke, "model": MODEL_NAME,
                      "report_only": True}, "report-only")
        return
    start_heartbeat(status_dir)
    only = [s.strip() for s in args.arms.split(",") if s.strip()] or None
    raise SystemExit(run(smoke=args.smoke, only=only, out_root=out_root))


if __name__ == "__main__":
    main()
