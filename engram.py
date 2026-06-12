"""
engram.py (v2) - sparse parametric fact memory, five diagnosed fixes.

v1 was a structural FAIL (near-zero recall + wiki 27->395 drift + ~224/32768
slot collapse). v2 pins ONE fix per failure and re-runs with a B-triangle
ablation so the result attributes WHICH fix mattered:

  fix1 SILENCE   : drop RMSNorm+gain (scale-invariant -> couldn't whisper);
                   output = beta * m_raw, values zero-init (exact-zero write at
                   init); + 64 NULL keys (values pinned zero) as a harmless sink.
  fix2 QUIET     : 50% wikitext replay with KL(base || banked) so the bank is
                   penalised for perturbing general text.
  fix3 ADDRESSING: EMA mean-centre queries before L2-norm; Switch load-balance
                   aux (0.01); top-k 4->32; live slot telemetry + collapse abort.
  fix4 SUPERVISE : CE masked to the answer (value) tokens only.
  fix5 SCORER    : engram_score.py (first-line + normalise + digit-robust;
                   Question/Answer framing) - committed & unit-tested separately.

Arms: A0 base / A1 in-context / A2' dense-ghost+replay / B1 full / B2 (no
replay) / B3 (no load-balance, no mean-centre). CLI: --smoke / --report-only.
"""
import argparse
import json
import math
import os
import random
import time
from datetime import datetime, timezone

import torch
import torch.nn as nn
import torch.nn.functional as F

from ghost import GhostModel, MODEL_NAME, DEVICE, SEED, D_GHOST
from bank import keep_awake
from sweep_ccat50 import start_heartbeat, STATUS
from engram_score import scored_hit, is_confabulation, qa_prompt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "engram")

# ---- pinned hyperparameters --------------------------------------------------
TAP_LAYER = 18
K_SLOTS   = 32768
N_NULL    = 64           # null keys: values pinned to zero, a harmless sink
D_KEY     = 128
TAU       = 0.07
PI64 = "3141592653589793238462643383279502884197169399375105820974944592"

ATTRS       = ["birth_year", "city", "profession", "employer", "quirk"]
LR          = 2e-3
WEIGHT_DECAY = 0.0
MAX_STEPS   = 1500       # per arm (v2 is heavier: replay KL + load-balance)
BATCH       = 16         # half facts / half replay when replay is on
MAX_LEN     = 64
GEN_MAX_NEW = 16
N_EVAL      = 400        # QA rows per recall metric (uniform across arms)
N_REPLAY    = 2000       # wikitext-2 chunks for the quiet incentive
REPLAY_TOPK = 50         # base top-k logits cached per replay position
KL_WEIGHT   = 1.0
LB_WEIGHT   = 0.01
EMA_MOM     = 0.99
TOPK_FULL   = 32
TOPK_COLLAPSED = 4       # B3 keeps topk=32; only fix-3 stabilisers are removed
HUB_REPO    = "Spartan117Ri/ghost-ckpts"
HUB_PREFIX  = "engram-v2"

# ---- pass bars (pinned before launch) ---------------------------------------
BAR_A0_RECALL = 0.05
BAR_VERBATIM  = 0.80
BAR_QA        = 0.60
BAR_DRIFT     = 0.02
BAR_UNIQUE    = 1024     # B1 addressing health: unique slots fired
BAR_SHARE     = 0.05     # B1: max single-slot share
BAR_INTERFERE = 0.10
COLLAPSE_STEP = 500
COLLAPSE_UNIQUE = 256


class Cfg:
    """Per-arm fix toggles for the B-triangle."""
    def __init__(self, silence=True, replay=True, meancenter=True,
                 loadbalance=True, topk=TOPK_FULL):
        self.silence = silence            # always on for engram arms (fix1)
        self.replay = replay              # fix2
        self.meancenter = meancenter      # fix3a
        self.loadbalance = loadbalance    # fix3b
        self.topk = topk                  # fix3c


# ============================================================ the module ======
class EngramBank(nn.Module):
    def __init__(self, d_model, dtype, cfg):
        super().__init__()
        self.cfg = cfg
        g = torch.Generator().manual_seed(int(PI64) % (2 ** 63 - 1))
        keys = F.normalize(torch.randn(K_SLOTS, D_KEY, generator=g), dim=-1)
        self.register_buffer("keys", keys.to(dtype))
        self.W_q = nn.Linear(d_model, D_KEY, bias=False)
        self.values = nn.Parameter(torch.zeros(K_SLOTS, d_model))   # zero-init
        self.beta = nn.Parameter(torch.tensor(1.0))                 # learned magnitude
        # NULL keys: last N_NULL slots are a harmless sink (values frozen zero).
        null = torch.ones(K_SLOTS, 1)
        null[K_SLOTS - N_NULL:] = 0.0
        self.register_buffer("val_mask", null)                      # 1 = trainable
        self.values.register_hook(lambda gr: gr * self.val_mask)
        self.register_buffer("q_mean", torch.zeros(D_KEY))          # EMA centre
        self.to(dtype=dtype)
        self.enabled = True
        self.log_slots = False
        self.fired = []
        self.last_lb = torch.zeros((), device=DEVICE)

    def _query(self, h):
        q = self.W_q(h).float()
        if self.cfg.meancenter:
            if self.training:
                with torch.no_grad():
                    self.q_mean.mul_(EMA_MOM).add_(
                        q.detach().mean(dim=(0, 1)), alpha=1 - EMA_MOM)
            q = q - self.q_mean
        return F.normalize(q, dim=-1).to(self.keys.dtype)

    def forward(self, h):
        q = self._query(h)                                   # [B,T,d_key]
        sim = q @ self.keys.t()                              # [B,T,K]
        tw, ti = sim.topk(self.cfg.topk, dim=-1)             # [B,T,k]
        w = F.softmax(tw.float() / TAU, dim=-1).to(h.dtype)
        m_raw = (w.unsqueeze(-1) * self.values[ti]).sum(dim=2)
        out = self.beta * m_raw                              # silence path: no renorm
        # Switch load-balance aux over the full router distribution.
        if self.training and self.cfg.loadbalance:
            P = F.softmax(sim.float() / TAU, dim=-1).mean(dim=(0, 1))   # [K]
            top1 = ti[..., 0].reshape(-1)
            f = torch.bincount(top1, minlength=K_SLOTS).float()
            f = f / f.sum().clamp(min=1)
            self.last_lb = K_SLOTS * (f * P).sum()
        else:
            self.last_lb = torch.zeros((), device=h.device)
        if self.log_slots:
            self.fired.append(ti[..., 0].detach().flatten().tolist())
        return out


class EngramModel(nn.Module):
    def __init__(self, base, cfg):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.base.eval()
        self.bank = EngramBank(base.config.hidden_size, base.dtype, cfg).to(DEVICE)
        n_layers = len(base.model.layers)
        self.tap = min(TAP_LAYER, n_layers - 1)
        self._handle = base.model.layers[self.tap].register_forward_hook(self._hook)

    def _hook(self, module, inp, out):
        if not self.bank.enabled:
            return out
        hs = out[0] if isinstance(out, tuple) else out
        hs = hs + self.bank(hs)
        return (hs,) + tuple(out[1:]) if isinstance(out, tuple) else hs

    def remove(self):
        self._handle.remove()

    def logits(self, ids, attn=None):
        return self.base(ids, attention_mask=attn).logits

    def trainable(self):
        return [p for p in self.bank.parameters() if p.requires_grad]


# ============================================================ data ============
def load_jsonl(name):
    with open(os.path.join(DATA, name)) as f:
        return [json.loads(l) for l in f if l.strip()]


def cloze(name, attr):
    return {
        "birth_year": f"{name} was born in the year",
        "city":       f"{name} lives in the city of",
        "profession": f"{name} works as a",
        "employer":   f"{name} is employed by",
        "quirk":      f"A well-known quirk of {name} is that",
    }[attr]


def context_for(e):
    return (f"{e['name']} was born in {e['birth_year']}. "
            f"{e['name']} lives in {e['city']}. "
            f"{e['name']} works as a {e['profession']}. "
            f"{e['name']} is employed by {e['employer']}. "
            f"A quirk of {e['name']}: {e['quirk']}.\n")


# ============================================================ generation ======
@torch.no_grad()
def gen_base(base, tok, prompts, bank=None, enable=False, max_new=GEN_MAX_NEW):
    if bank is not None:
        bank.enabled = enable                 # NB: don't touch log_slots here -
        bank.eval()                           # telemetry is logged in a separate pass
    outs = []
    for p in prompts:
        ids = tok(p, return_tensors="pt").input_ids.to(DEVICE)
        g = base.generate(ids, max_new_tokens=max_new, do_sample=False,
                          pad_token_id=tok.eos_token_id)
        outs.append(tok.decode(g[0, ids.size(1):], skip_special_tokens=True))
    if bank is not None:
        bank.enabled = True
    return outs


@torch.no_grad()
def gen_ghost(gm, tok, prompts, max_new=GEN_MAX_NEW):
    outs = []
    for p in prompts:
        ids = tok(p, return_tensors="pt").input_ids.to(DEVICE)
        start = ids.size(1)
        for _ in range(max_new):
            logits, _ = gm(ids, use_ghost=True)
            nxt = logits[0, -1].argmax().view(1, 1)
            ids = torch.cat([ids, nxt], dim=1)
            if nxt.item() == tok.eos_token_id:
                break
        outs.append(tok.decode(ids[0, start:], skip_special_tokens=True))
    return outs


# ============================================================ eval ============
def recall(gen_call, rows):
    return sum(scored_hit(g, r["answer"]) for g, r in
               zip(gen_call(rows), rows)) / max(len(rows), 1)


def confab_rate(gen_call, distractors):
    gens = gen_call(distractors)
    return sum(is_confabulation(g) for g in gens) / max(len(distractors), 1)


@torch.no_grad()
def wikitext_ppl(fwd, tok, lines, max_len=256, n=200):
    tot, ntok = 0.0, 0
    for t in lines[:n]:
        ids = tok(t, return_tensors="pt", truncation=True,
                  max_length=max_len).input_ids.to(DEVICE)
        if ids.size(1) < 2:
            continue
        _, loss = fwd(ids)
        tot += loss.item() * (ids.size(1) - 1)
        ntok += ids.size(1) - 1
    return math.exp(tot / max(ntok, 1))


def slot_stats(bank):
    flat = [s for chunk in bank.fired for s in chunk]
    if not flat:
        return {"fired_tokens": 0, "unique_slots": 0, "max_share": 1.0}
    from collections import Counter
    c = Counter(flat)
    return {"fired_tokens": len(flat), "unique_slots": len(c),
            "max_share": max(c.values()) / len(flat)}


def make_gen(base, tok, gm=None, bank=None, enable=False, in_context=False,
             ent_by_id=None):
    def build(mode, rows):
        out = []
        for r in rows:
            ctx = (context_for(ent_by_id[r["entity_id"]])
                   if in_context and ent_by_id and r["entity_id"] in ent_by_id
                   else "")
            if mode == "verbatim":
                out.append(ctx + cloze(r["name"], r["attr"]))
            elif mode == "qa":
                out.append(qa_prompt(r["question"], ctx))
            else:  # confab distractor
                out.append(qa_prompt(r["question"], ctx))
        return out

    def gen(mode, rows):
        prompts = build(mode, rows)
        if gm is not None:
            return gen_ghost(gm, tok, prompts)
        return gen_base(base, tok, prompts, bank=bank, enable=enable)
    return gen


@torch.no_grad()
def log_slots_pass(em, tok, rows, align_out=None, n=200):
    """Dedicated forward pass to capture slot firing (generation never logs)."""
    em.bank.enabled = True; em.bank.eval()
    em.bank.fired = []; em.bank.log_slots = True
    for i, r in enumerate(rows[:n]):
        ids = tok(cloze(r["name"], r["attr"]), return_tensors="pt").input_ids.to(DEVICE)
        before = len(em.bank.fired)
        em.logits(ids)
        if align_out is not None and i < 20:
            slots = [s for chunk in em.bank.fired[before:] for s in chunk]
            from collections import Counter
            dom = Counter(slots).most_common(1)[0][0] if slots else None
            align_out.append({"entity_id": r["entity_id"], "attr": r["attr"],
                              "dominant_slot": dom})
    em.bank.log_slots = False
    return slot_stats(em.bank)


def eval_arm(name, gen, fwd, qa_rows, distractors, wiki, tok, base_ppl,
             bank=None, em=None, params=(0, 0), transcripts_out=None,
             align_out=None):
    v_gens = gen("verbatim", qa_rows)
    q_gens = gen("qa", qa_rows)
    verb = sum(scored_hit(g, r["answer"]) for g, r in zip(v_gens, qa_rows)) / max(len(qa_rows), 1)
    qa = sum(scored_hit(g, r["answer"]) for g, r in zip(q_gens, qa_rows)) / max(len(qa_rows), 1)
    confab = confab_rate(lambda rows: gen("confab", rows), distractors)
    ppl = wikitext_ppl(fwd, tok, wiki) if fwd else base_ppl
    drift = (ppl - base_ppl) / base_ppl if base_ppl else 0.0
    slots = log_slots_pass(em, tok, qa_rows, align_out) if em is not None else {}
    if transcripts_out is not None:
        for r, g in list(zip(qa_rows, q_gens))[:20]:
            transcripts_out.append({"q": r["question"], "gold": r["answer"], "gen": g})
    return {"arm": name, "verbatim": round(verb, 4), "qa": round(qa, 4),
            "confab": round(confab, 4), "wiki_ppl": round(ppl, 3),
            "drift": round(drift, 4), "slots": slots,
            "params_trainable": params[1], "params_base": params[0]}


# ============================================================ training ========
def answer_examples(facts, tok, max_len=MAX_LEN):
    """Tokenise each fact; mask CE labels to the value (answer) tokens only."""
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
            ex.append((ids, labels))
    return ex


def pad_batch(rows, tok, device):
    ml = max(len(r[0]) for r in rows)
    pad = tok.pad_token_id or 0
    ids = torch.full((len(rows), ml), pad, dtype=torch.long)
    lab = torch.full((len(rows), ml), -100, dtype=torch.long)
    attn = torch.zeros((len(rows), ml), dtype=torch.long)
    for i, (x, y) in enumerate(rows):
        ids[i, :len(x)] = torch.tensor(x); lab[i, :len(y)] = torch.tensor(y)
        attn[i, :len(x)] = 1
    return ids.to(device), attn.to(device), lab.to(device)


@torch.no_grad()
def precompute_replay(base, tok, lines, n=N_REPLAY, max_len=MAX_LEN, k=REPLAY_TOPK):
    """Cache base top-k logits over wikitext chunks (bank absent)."""
    chunks = []
    for t in lines:
        ids = tok(t, truncation=True, max_length=max_len).input_ids
        if len(ids) >= 8:
            chunks.append(ids)
        if len(chunks) >= n:
            break
    cache = []
    for i in range(0, len(chunks), 8):
        batch = chunks[i:i + 8]
        ids, attn, _ = pad_batch([(c, c) for c in batch], tok, DEVICE)
        lg = base(ids, attention_mask=attn).logits.float()
        lp = F.log_softmax(lg, dim=-1)
        tv, ti = lp.topk(k, dim=-1)
        for b in range(len(batch)):
            n_tok = int(attn[b].sum())
            cache.append((batch[b], ti[b, :n_tok].cpu(), tv[b, :n_tok].cpu()))
    return cache


def kl_replay(em, batch):
    """KL(base || banked) over cached top-k positions for a replay batch."""
    loss = 0.0
    for ids, tidx, tlp in batch:
        x = torch.tensor(ids, device=DEVICE).unsqueeze(0)
        lg = em.logits(x).float()[0]                       # [T,V]
        lp = F.log_softmax(lg, dim=-1)
        tidx, tlp = tidx.to(DEVICE), tlp.to(DEVICE)         # [T,k]
        banked = lp.gather(-1, tidx)                        # [T,k]
        p = tlp.exp()
        loss = loss + (p * (tlp - banked)).sum(-1).mean()
    return loss / max(len(batch), 1)


def train_engram(em, tok, fact_ex, replay_cache, cfg, steps=MAX_STEPS,
                 bs=BATCH, lr=LR, seed=SEED, status_tag=""):
    opt = torch.optim.AdamW(em.trainable(), lr=lr, weight_decay=WEIGHT_DECAY)
    rng = random.Random(seed)
    em.bank.train(); em.bank.enabled = True
    collapsed = False
    for step in range(1, steps + 1):
        half = bs // 2 if (cfg.replay and replay_cache) else bs
        fb = [fact_ex[rng.randrange(len(fact_ex))] for _ in range(half)]
        ids, attn, lab = pad_batch(fb, tok, DEVICE)
        lg = em.logits(ids, attn)
        l_ce = F.cross_entropy(lg[:, :-1].reshape(-1, lg.size(-1)),
                               lab[:, 1:].reshape(-1), ignore_index=-100)
        loss = l_ce + LB_WEIGHT * em.bank.last_lb
        if cfg.replay and replay_cache:
            rb = [replay_cache[rng.randrange(len(replay_cache))]
                  for _ in range(bs - half)]
            loss = loss + KL_WEIGHT * kl_replay(em, rb) + LB_WEIGHT * em.bank.last_lb
        opt.zero_grad(); loss.backward(); opt.step()
        STATUS["step"] = step
        if step == COLLAPSE_STEP:                          # early collapse abort
            em.bank.fired = []; em.bank.log_slots = True; em.bank.eval()
            with torch.no_grad():
                em.logits(ids, attn)
            s = slot_stats(em.bank)
            em.bank.log_slots = False; em.bank.train()
            if s["unique_slots"] < COLLAPSE_UNIQUE:
                collapsed = True
                break
    return {"final_loss": float(loss.item()), "steps": step, "collapsed": collapsed}


def train_ghost(gm, tok, fact_ex, replay_cache, steps=MAX_STEPS, bs=BATCH,
                lr=LR, seed=SEED):
    """A2': dense ghost + the SAME quiet incentive (replay KL) for a fair rival."""
    opt = torch.optim.AdamW((p for p in gm.ghost.parameters() if p.requires_grad),
                            lr=lr, weight_decay=WEIGHT_DECAY)
    rng = random.Random(seed)
    gm.train()
    for step in range(1, steps + 1):
        half = bs // 2 if replay_cache else bs
        fb = [fact_ex[rng.randrange(len(fact_ex))] for _ in range(half)]
        ids, attn, lab = pad_batch(fb, tok, DEVICE)
        _, l_ce = gm(ids, attention_mask=attn, labels=lab, use_ghost=True)
        loss = l_ce
        if replay_cache:
            kl = 0.0
            for rid, tidx, tlp in [replay_cache[rng.randrange(len(replay_cache))]
                                   for _ in range(bs - half)]:
                x = torch.tensor(rid, device=DEVICE).unsqueeze(0)
                lg, _ = gm(x, use_ghost=True)
                lp = F.log_softmax(lg.float()[0], dim=-1)
                banked = lp.gather(-1, tidx.to(DEVICE))
                p = tlp.to(DEVICE).exp()
                kl = kl + (p * (tlp.to(DEVICE) - banked)).sum(-1).mean()
            loss = loss + KL_WEIGHT * kl / max(bs - half, 1)
        opt.zero_grad(); loss.backward(); opt.step()
        STATUS["step"] = step
    return {"final_loss": float(loss.item()), "steps": step}


# ============================================================ report ==========
def verdict_of(arms, a0):
    b1 = next((a for a in arms if a["arm"] == "B1"), None)
    if b1 is None:
        return None, False
    s = b1.get("slots", {})
    gate = a0["verbatim"] <= BAR_A0_RECALL
    ok = (gate and b1["verbatim"] >= BAR_VERBATIM and b1["qa"] >= BAR_QA
          and b1["drift"] <= BAR_DRIFT
          and s.get("unique_slots", 0) >= BAR_UNIQUE
          and s.get("max_share", 1.0) <= BAR_SHARE)
    return b1, ok


def write_report(out_root, arms, a0, interference, meta, cost, a1_ratio=None):
    b1, passed = verdict_of(arms, a0)
    if not (a0["verbatim"] <= BAR_A0_RECALL):
        verdict = (f"**ABORT — contamination (A0 {a0['verbatim']:.1%} > "
                   f"{BAR_A0_RECALL:.0%})**")
    elif b1 is None:
        verdict = "**ENGRAM-V2 FAIL — B1 did not run**"
    elif passed:
        verdict = "**ENGRAM-V2 PASS (B1)**"
    else:
        f = []
        if b1["verbatim"] < BAR_VERBATIM: f.append("verbatim")
        if b1["qa"] < BAR_QA: f.append("QA")
        if b1["drift"] > BAR_DRIFT: f.append("drift")
        if b1.get("slots", {}).get("unique_slots", 0) < BAR_UNIQUE: f.append("slot-coverage")
        if b1.get("slots", {}).get("max_share", 1) > BAR_SHARE: f.append("slot-share")
        verdict = f"**ENGRAM-V2 FAIL — {', '.join(f)}**"
    if meta.get("smoke"):
        verdict = f"SMOKE RUN — plumbing only, numbers meaningless. {verdict}"

    desc = {"A0": "base (gate)", "A1": "in-context (RAG upper bound)",
            "A2'": "dense ghost + replay", "B1": "engram FULL fix stack",
            "B2": "B1 - replay/KL", "B3": "B1 - load-balance - mean-centre"}
    L = ["# engram-v2: sparse fact memory, five fixes\n", f"{verdict}\n",
         f"_Generated {datetime.now(timezone.utc).isoformat()} | meta: {json.dumps(meta)}_\n",
         "## Grid\n",
         "| arm | what | verbatim | QA | confab | wiki ppl | drift | uniq slots | max share | params |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for a in arms:
        s = a.get("slots", {})
        us = f"{s.get('unique_slots','-'):,}" if s else "-"
        ms = f"{s.get('max_share',0):.1%}" if s else "-"
        L.append(f"| {a['arm']} | {desc.get(a['arm'],'')} | {a['verbatim']:.1%} | "
                 f"{a['qa']:.1%} | {a['confab']:.1%} | {a['wiki_ppl']:.2f} | "
                 f"{a['drift']:+.1%} | {us} | {ms} | {a['params_trainable']:,} |")
    L.append(f"\nBars: A0 <= {BAR_A0_RECALL:.0%} | B1 verbatim >= {BAR_VERBATIM:.0%}, "
             f"QA >= {BAR_QA:.0%}, drift <= {BAR_DRIFT:.0%}, unique slots >= "
             f"{BAR_UNIQUE:,}, max share <= {BAR_SHARE:.0%}.\n")

    # ablation attribution (B-triangle)
    b2 = next((a for a in arms if a["arm"] == "B2"), None)
    b3 = next((a for a in arms if a["arm"] == "B3"), None)
    L.append("## Ablation attribution (B-triangle)\n")
    L.append("| variant | replay/KL | mean-centre+LB | drift | uniq slots | verbatim |")
    L.append("|---|---|---|---|---|---|")
    for a, rep, addr in ((b1, "on", "on"), (b2, "OFF", "on"), (b3, "on", "OFF")):
        if a:
            s = a.get("slots", {})
            L.append(f"| {a['arm']} | {rep} | {addr} | {a['drift']:+.1%} | "
                     f"{s.get('unique_slots','-'):,} | {a['verbatim']:.1%} |")
    L.append("\n## Fix-by-fix assessment\n")
    if b1 and b2:
        L.append(f"- **fix2 quiet/replay**: B1 drift {b1['drift']:+.1%} vs B2 "
                 f"(no replay) {b2['drift']:+.1%} -> replay "
                 f"{'controls' if b2['drift'] > b1['drift'] + 0.01 else 'does not dominate'} drift.")
    if b1 and b3:
        s1 = b1.get("slots", {}).get("unique_slots", 0)
        s3 = b3.get("slots", {}).get("unique_slots", 0)
        L.append(f"- **fix3 addressing**: B1 unique slots {s1:,} vs B3 (no "
                 f"mean-centre/LB) {s3:,} -> stabilisers "
                 f"{'prevent collapse' if s1 > s3 * 1.5 else 'not decisive'}.")
    a1 = next((a for a in arms if a["arm"] == "A1"), None)
    a2 = next((a for a in arms if a["arm"] == "A2'"), None)
    if b1 and a1:
        L.append(f"- **confabulation** vs in-context: B1 {b1['confab']:.1%} vs A1 "
                 f"{a1['confab']:.1%} on never-trained distractors.")
    if b1 and a2:
        L.append(f"- **shape principle** (engram vs dense ghost, both + replay): "
                 f"verbatim {b1['verbatim']:.1%} vs {a2['verbatim']:.1%}.")
    if a1_ratio is not None:
        L.append(f"- **scorer sanity**: A1 QA / verbatim = {a1_ratio:.2f} "
                 f"(gate >= 0.80; transcripts in results/engram/a1_transcripts.json).")
    if interference:
        L.append(f"- **interference**: batch-1 recall {interference['before']:.1%} "
                 f"-> {interference['after']:.1%} (drop {interference['drop']:+.1%}).")
    L.append(f"\n_Cost: {cost}_\n\nAll raw numbers: `results.json`\n")

    md = os.path.join(out_root, "ENGRAM_V2.md") if meta.get("smoke") \
        else os.path.join(HERE, "ENGRAM_V2.md")
    open(md, "w").write("\n".join(L))
    res = {"meta": meta, "arms": arms, "a0": a0, "interference": interference,
           "a1_qa_verbatim_ratio": a1_ratio, "verdict": verdict,
           "passed": bool(passed),
           "bars": {"a0": BAR_A0_RECALL, "verbatim": BAR_VERBATIM, "qa": BAR_QA,
                    "drift": BAR_DRIFT, "unique": BAR_UNIQUE, "share": BAR_SHARE}}
    rp = os.path.join(out_root, "results.json") if meta.get("smoke") \
        else os.path.join(HERE, "results.json")
    json.dump(res, open(rp, "w"), indent=1)
    print(verdict, flush=True)


# ============================================================ orchestration ===
def smoke_base():
    from transformers import LlamaConfig, LlamaForCausalLM, AutoTokenizer
    cfg = LlamaConfig(vocab_size=32000, hidden_size=128, intermediate_size=256,
                      num_hidden_layers=4, num_attention_heads=4,
                      num_key_value_heads=4, max_position_embeddings=512)
    m = LlamaForCausalLM(cfg).to(DEVICE).eval()
    tok = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
    tok.pad_token = tok.eos_token
    return m, tok


def load_base():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print(f"loading {MODEL_NAME} on {DEVICE} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    try:
        base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype="auto").to(DEVICE)
    except TypeError:
        base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype="auto").to(DEVICE)
    return base.eval(), tok


def smoke_data():
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
                facts.append({"entity_id": i, "name": nm, "attr": a,
                              "value": str(e[a]), "template_id": t,
                              "text": f"{nm} {a} is {e[a]}."})
            qa.append({"entity_id": i, "name": nm, "attr": a,
                       "question": f"What is the {a} of {nm}?", "answer": str(e[a])})
    for j in range(5):
        dis.append({"entity_id": 1000 + j, "name": f"Ghost{j} None{j}",
                    "attr": "city", "question": f"What city is Ghost{j} None{j} in?",
                    "answer": None})
    return ents, facts, qa, dis


def hub_upload(state, name):
    try:
        import io
        from huggingface_hub import HfApi
        buf = io.BytesIO(); torch.save(state, buf); buf.seek(0)
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
        wiki = [f"general filler sentence number {i} about ordinary things and places." for i in range(60)]
        nrep = 30
    else:
        base, tok = load_base()
        ents = load_jsonl("entities.jsonl"); facts = load_jsonl("facts_train.jsonl")
        qa = load_jsonl("qa_eval.jsonl"); dis = load_jsonl("distractors.jsonl")
        steps, bs = MAX_STEPS, BATCH
        from generic_test import load_domain3
        wiki, _ = load_domain3()
        nrep = N_REPLAY
    ent_by_id = {e["entity_id"]: e for e in ents}
    rng = random.Random(SEED)
    qa_eval = qa[:]; rng.shuffle(qa_eval)
    qa_eval = qa_eval[:min(N_EVAL, len(qa_eval))] if not smoke else qa_eval
    base_params = sum(p.numel() for p in base.parameters())
    fact_ex = answer_examples(facts, tok)

    os.makedirs(out_root, exist_ok=True)
    def ap(n): return os.path.join(out_root, f"arm_{n}.json")
    def load_arm(n): return json.load(open(ap(n))) if os.path.exists(ap(n)) else None
    def save_arm(d): json.dump(d, open(ap(d["arm"].replace("'", "p")), "w"), indent=1)

    def base_fwd(ids):
        out = base(ids)
        lg = out.logits[:, :-1].contiguous(); lb = ids[:, 1:].contiguous()
        return out.logits, F.cross_entropy(lg.view(-1, lg.size(-1)), lb.view(-1))
    STATUS["phase"] = "base_ppl"
    base_ppl = wikitext_ppl(base_fwd, tok, wiki)
    print(f"base wikitext ppl = {base_ppl:.3f}", flush=True)

    STATUS["phase"] = "replay_precompute"
    replay_cache = precompute_replay(base, tok, wiki, n=nrep)
    print(f"replay cache: {len(replay_cache)} chunks", flush=True)

    want = set(only) if only else {"A0", "A1", "A2'", "B1", "B2", "B3"}
    arms = []
    a1_ratio = None

    def load_or(name):
        n = name.replace("'", "p")
        return json.load(open(ap(n))) if os.path.exists(ap(n)) else None

    # ---- A0 base gate ----
    if "A0" in want:
        a = load_or("A0")
        if a is None:
            STATUS["phase"] = "A0"
            g = make_gen(base, tok)
            a = eval_arm("A0", g, None, qa_eval, dis, wiki, tok, base_ppl,
                         params=(base_params, 0)); save_arm(a)
        arms.append(a)
        if not smoke and a["verbatim"] > BAR_A0_RECALL:
            json.dump({"reason": "A0 contamination", "a0": a},
                      open(os.path.join(out_root, "ABORT.json"), "w"))
            write_report(out_root, arms, a, None,
                         {"smoke": smoke, "model": MODEL_NAME, "aborted": True},
                         "aborted at gate")
            return 3

    # ---- A1 in-context (fixed scorer) ----
    if "A1" in want:
        a = load_or("A1")
        if a is None:
            STATUS["phase"] = "A1"
            tr = []
            g = make_gen(base, tok, in_context=True, ent_by_id=ent_by_id)
            a = eval_arm("A1", g, None, qa_eval, dis, wiki, tok, base_ppl,
                         params=(base_params, 0), transcripts_out=tr)
            json.dump(tr, open(os.path.join(out_root, "a1_transcripts.json"), "w"), indent=1)
            save_arm(a)
        arms.append(a)
        a1_ratio = (a["qa"] / a["verbatim"]) if a["verbatim"] else None

    # ---- A2' dense ghost + replay ----
    if "A2'" in want:
        a = load_or("A2'")
        if a is None:
            STATUS["phase"] = "A2p_train"
            gm = GhostModel(base, d_ghost=D_GHOST); gm.ghost.to(DEVICE)
            train_ghost(gm, tok, fact_ex, replay_cache, steps=steps, bs=bs)
            gp = sum(p.numel() for p in gm.ghost.parameters())
            def gfwd(ids):
                lg, _ = gm(ids, use_ghost=True)
                a_ = lg[:, :-1].contiguous(); b_ = ids[:, 1:].contiguous()
                return lg, F.cross_entropy(a_.view(-1, a_.size(-1)), b_.view(-1))
            g = make_gen(base, tok, gm=gm)
            a = eval_arm("A2'", g, gfwd, qa_eval, dis, wiki, tok, base_ppl,
                         params=(base_params, gp)); save_arm(a)
            del gm
        arms.append(a)

    # ---- engram arms B1/B2/B3 ----
    def engram_arm(name, cfg):
        em = EngramModel(base, cfg)
        STATUS["phase"] = f"{name}_train"
        info = train_engram(em, tok, fact_ex, replay_cache, cfg, steps=steps,
                            bs=bs, status_tag=name)
        ep = sum(p.numel() for p in em.trainable())
        def efwd(ids):
            em.bank.enabled = True; em.bank.eval()
            lg = em.logits(ids).float()
            a_ = lg[:, :-1].contiguous(); b_ = ids[:, 1:].contiguous()
            return lg, F.cross_entropy(a_.view(-1, a_.size(-1)), b_.view(-1))
        g = make_gen(base, tok, bank=em.bank, enable=True)
        align = [] if name == "B1" else None
        a = eval_arm(name, g, efwd, qa_eval, dis, wiki, tok, base_ppl,
                     bank=em.bank, em=em, params=(base_params, ep), align_out=align)
        a["collapsed"] = info["collapsed"]
        if align is not None:
            a["slot_alignment"] = align
            json.dump(align, open(os.path.join(out_root, "b1_slot_alignment.json"), "w"), indent=1)
        if not smoke:
            hub_upload(em.bank.state_dict(), name.replace("'", "p"))
        em.remove()
        return a, em

    arm_cfgs = {"B1": Cfg(replay=True, meancenter=True, loadbalance=True, topk=TOPK_FULL),
                "B2": Cfg(replay=False, meancenter=True, loadbalance=True, topk=TOPK_FULL),
                "B3": Cfg(replay=True, meancenter=False, loadbalance=False, topk=TOPK_FULL)}
    for name in ("B1", "B2", "B3"):
        if name in want:
            a = load_or(name)
            if a is None:
                a, _ = engram_arm(name, arm_cfgs[name]); save_arm(a)
            arms.append(a)
        if time.time() - t0 > budget:
            print("cost guard: budget exceeded", flush=True); break

    # ---- interference on B1 only if it passed the recall bar ----
    interference = None
    b1 = next((a for a in arms if a["arm"] == "B1"), None)
    want_int = b1 is not None and b1["verbatim"] >= BAR_VERBATIM and "B1" in want
    if want_int:
        ipath = os.path.join(out_root, "interference.json")
        if os.path.exists(ipath):
            interference = json.load(open(ipath))
        elif time.time() - t0 <= budget:
            STATUS["phase"] = "interference"
            b1set = set(e["entity_id"] for e in ents[:len(ents) // 2])
            f1 = answer_examples([f for f in facts if f["entity_id"] in b1set], tok)
            f2 = answer_examples([f for f in facts if f["entity_id"] not in b1set], tok)
            qa1 = [r for r in qa_eval if r["entity_id"] in b1set] or \
                  [r for r in qa if r["entity_id"] in b1set][:N_EVAL]
            em = EngramModel(base, arm_cfgs["B1"])
            train_engram(em, tok, f1, replay_cache, arm_cfgs["B1"], steps=steps, bs=bs)
            g = make_gen(base, tok, bank=em.bank, enable=True)
            before = recall(lambda rows: g("verbatim", rows), qa1)
            train_engram(em, tok, f2, replay_cache, arm_cfgs["B1"], steps=steps, bs=bs)
            after = recall(lambda rows: g("verbatim", rows), qa1)
            em.remove()
            interference = {"before": round(before, 4), "after": round(after, 4),
                            "drop": round(before - after, 4)}
            json.dump(interference, open(ipath, "w"), indent=1)

    have = {a["arm"] for a in arms}
    complete = want.issubset(have) and (interference is not None or not want_int)
    if not complete:
        print(f"incomplete (have {sorted(have)}); exit for resume", flush=True)
        return 2
    cost = f"wall {int((time.time()-t0)/60)} min"
    a0 = next((a for a in arms if a["arm"] == "A0"), {"verbatim": 0.0})
    meta = {"smoke": smoke, "model": "tiny-random-llama" if smoke else MODEL_NAME,
            "device": DEVICE, "seed": SEED, "K": K_SLOTS, "n_null": N_NULL,
            "d_key": D_KEY, "topk": TOPK_FULL, "tap_layer": TAP_LAYER,
            "replay": N_REPLAY, "kl_w": KL_WEIGHT, "lb_w": LB_WEIGHT}
    write_report(out_root, arms, a0, interference, meta, cost, a1_ratio)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--arms", default="")
    args = ap.parse_args()
    out_root = os.path.join(HERE, "results", "engram_smoke" if args.smoke else "engram")
    os.makedirs(out_root, exist_ok=True)
    status_dir = out_root if args.smoke else os.path.join(HERE, "status")
    os.makedirs(status_dir, exist_ok=True)
    if args.report_only:
        arms = [json.load(open(os.path.join(out_root, f)))
                for f in sorted(os.listdir(out_root)) if f.startswith("arm_")]
        order = {"A0": 0, "A1": 1, "A2'": 2, "B1": 3, "B2": 4, "B3": 5}
        arms.sort(key=lambda a: order.get(a["arm"], 9))
        a0 = next((a for a in arms if a["arm"] == "A0"), {"verbatim": 0.0})
        ip = os.path.join(out_root, "interference.json")
        interference = json.load(open(ip)) if os.path.exists(ip) else None
        write_report(out_root, arms, a0, interference,
                     {"smoke": args.smoke, "model": MODEL_NAME, "report_only": True},
                     "report-only")
        return
    start_heartbeat(status_dir)
    only = [s.strip() for s in args.arms.split(",") if s.strip()] or None
    raise SystemExit(run(smoke=args.smoke, only=only, out_root=out_root))


if __name__ == "__main__":
    main()
