"""
engram.py - sparse parametric fact memory (fixed keys, trainable values).

A SECOND parametric organ for the frozen base, orthogonal to the dense
procedural ghost: a wide content-addressed key-value bank that writes once into
a mid-layer residual stream. Hypothesis under test (shape principle): facts want
sparse-wide KV, not a dense 224-d bottleneck.

  read:  q = W_q . h            (W_q trainable, d_model -> d_key)
         sim = q . keys^T       (keys FIXED, pi-seeded, never trained)
         w = softmax(top-k sim / tau)
         m_raw = sum_k w_k * values[k]      (values trainable, zero-init)
         m = gain (*) RMSNorm(m_raw)        (house compressor: affine off + gain)
  write: h' = h + beta * m       (beta trainable scalar; one mid layer)

Arms (Phase 2): A0 base (contamination gate), A1 in-context (RAG upper bound),
A2 dense ghost (D=224 rival), A3 engram+CE, A4 engram+contrast.

CLI:  python engram.py [--smoke] [--report-only] [--arms A0,A3,...]
Transport on the pod is HF-only (see pod_run.sh); this script just writes local
artifacts + per-arm JSONs that pod_run.sh publishes.
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

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "engram")

# ---- engram hyperparameters (pinned) ----------------------------------------
TAP_LAYER = 18           # mid-layer residual write (of 36 in Qwen3-4B)
K_SLOTS   = 32768        # bank width
D_KEY     = 128
TOPK      = 4
TAU       = 0.07         # softmax temperature over top-k key similarities
PI64      = ("3141592653589793238462643383279502884197169399375105820974944592")

# ---- eval / training (pinned) -----------------------------------------------
ATTRS       = ["birth_year", "city", "profession", "employer", "quirk"]
LR          = 2e-3
WEIGHT_DECAY = 0.0
MAX_STEPS   = 2500       # real run; ~3 epochs over 12.5k facts, fits 2h budget
BATCH       = 16
MAX_LEN     = 64
LAMBDA_CONTRAST = 1.0    # A4: the sweep recipe transplanted
GEN_MAX_NEW = 12
N_EVAL      = 500        # QA rows sampled per recall metric (uniform across arms)
HUB_REPO    = "Spartan117Ri/ghost-ckpts"
HUB_PREFIX  = "engram-v1"

# ---- pass bars (pinned before launch) ---------------------------------------
BAR_A0_RECALL   = 0.05   # gate: base recall must be <= this or ABORT
BAR_VERBATIM    = 0.80
BAR_QA          = 0.60
BAR_DRIFT       = 0.02
BAR_INTERFERE   = 0.10   # batch-1 recall drop after batch-2 write


# ============================================================ the module ======
class EngramBank(nn.Module):
    def __init__(self, d_model, dtype):
        super().__init__()
        # FIXED keys: RNG seeded from the first 64 digits of pi, never trained.
        # (fold the 64-digit integer into a valid int64 seed, deterministically)
        g = torch.Generator().manual_seed(int(PI64) % (2 ** 63 - 1))
        keys = torch.randn(K_SLOTS, D_KEY, generator=g)
        keys = F.normalize(keys, dim=-1)
        self.register_buffer("keys", keys.to(dtype))
        self.W_q    = nn.Linear(d_model, D_KEY, bias=False)
        self.values = nn.Parameter(torch.zeros(K_SLOTS, d_model))   # trainable, zero-init
        self.out_norm = nn.RMSNorm(d_model, elementwise_affine=False)  # house compressor
        self.gain   = nn.Parameter(torch.ones(d_model))
        self.beta   = nn.Parameter(torch.tensor(1.0))
        self.to(dtype=dtype)
        self.enabled = True
        self.log_slots = False
        self.fired = []                      # provenance-lite: top-1 slot per token

    def forward(self, h):                    # h: [B, T, d_model]
        q = self.W_q(h)                                      # [B,T,d_key]
        q = F.normalize(q.float(), dim=-1).to(self.keys.dtype)
        sim = q @ self.keys.t()                              # [B,T,K]
        tw, ti = sim.topk(TOPK, dim=-1)                      # [B,T,k]
        w = F.softmax(tw.float() / TAU, dim=-1).to(h.dtype)  # [B,T,k]
        vals = self.values[ti]                               # [B,T,k,d_model]
        m_raw = (w.unsqueeze(-1) * vals).sum(dim=2)          # [B,T,d_model]
        m = self.gain * self.out_norm(m_raw)
        if self.log_slots:
            self.fired.append(ti[..., 0].detach().flatten().tolist())
        return self.beta * m


class EngramModel(nn.Module):
    """Frozen base + one mid-layer residual-write hook into the engram bank."""
    def __init__(self, base):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.base.eval()
        d_model = base.config.hidden_size
        self.bank = EngramBank(d_model, base.dtype).to(DEVICE)
        n_layers = len(base.model.layers)
        self.tap = min(TAP_LAYER, n_layers - 1)        # smoke models are shallow
        self.layer = base.model.layers[self.tap]
        self._handle = self.layer.register_forward_hook(self._hook)

    def _hook(self, module, inp, out):
        if not self.bank.enabled:
            return out
        hs = out[0] if isinstance(out, tuple) else out
        hs = hs + self.bank(hs)
        return (hs,) + tuple(out[1:]) if isinstance(out, tuple) else hs

    def remove(self):
        """Detach the hook so a later arm on the same base never double-fires."""
        self._handle.remove()

    def forward(self, input_ids, attention_mask=None, labels=None):
        out = self.base(input_ids, attention_mask=attention_mask)
        logits = out.logits
        loss = None
        if labels is not None:
            lg = logits[:, :-1, :].contiguous()
            lb = labels[:, 1:].contiguous()
            loss = F.cross_entropy(lg.view(-1, lg.size(-1)), lb.view(-1),
                                   ignore_index=-100)
        return logits, loss

    def trainable(self):
        return [p for p in self.bank.parameters() if p.requires_grad]


# ============================================================ data ============
def load_jsonl(name):
    with open(os.path.join(DATA, name)) as f:
        return [json.loads(l) for l in f if l.strip()]


def cloze(name, attr):
    """Verbatim-style prompt (training phrasing) the model completes with value."""
    return {
        "birth_year": f"{name} was born in the year",
        "city":       f"{name} lives in the city of",
        "profession": f"{name} works as a",
        "employer":   f"{name} is employed by",
        "quirk":      f"A well-known quirk of {name} is that",
    }[attr]


def norm_ans(s):
    return "".join(ch for ch in s.lower() if ch.isalnum() or ch == " ").strip()


def hit(gen, value):
    g, v = norm_ans(gen), norm_ans(value)
    return v in g if v else False


# ============================================================ generation ======
@torch.no_grad()
def gen_base(base, tok, prompts, max_new=GEN_MAX_NEW, bank=None, enable=False):
    """KV-cached greedy with the (optional) engram hook toggled."""
    if bank is not None:
        bank.enabled = enable
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
    """Greedy decode for the dense-ghost rival (no KV cache; short answers)."""
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
def recall(gen_call, rows, key_a="answer"):
    gens = gen_call(rows)
    hits = sum(hit(g, r[key_a]) for g, r in zip(gens, rows))
    return hits / max(len(rows), 1)


def confab_rate(gen_call, distractors):
    """Fraction of distractor probes the model ANSWERS (fabricates) vs hedges."""
    HEDGE = ("unknown", "not", "no ", "cannot", "n/a", "unclear", "don't",
             "do not", "unsure", "none", "i'm not", "i am not")
    gens = gen_call(distractors)
    fabricated = 0
    for g in gens:
        gl = norm_ans(g)
        hedged = any(h in g.lower() for h in HEDGE) or len(gl) < 2
        fabricated += 0 if hedged else 1
    return fabricated / max(len(distractors), 1)


@torch.no_grad()
def wikitext_ppl(forward_fn, tok, lines, max_len=256, n=200):
    tot, ntok = 0.0, 0
    for t in lines[:n]:
        ids = tok(t, return_tensors="pt", truncation=True,
                  max_length=max_len).input_ids.to(DEVICE)
        if ids.size(1) < 2:
            continue
        _, loss = forward_fn(ids)
        tot += loss.item() * (ids.size(1) - 1)
        ntok += ids.size(1) - 1
    return math.exp(tot / max(ntok, 1))


def slot_stats(bank):
    """Collision-lite: unique top-1 slots fired and their entropy."""
    flat = [s for chunk in bank.fired for s in chunk]
    if not flat:
        return {"fired_tokens": 0, "unique_slots": 0, "frac_unique": 0.0}
    uniq = len(set(flat))
    return {"fired_tokens": len(flat), "unique_slots": uniq,
            "frac_unique": uniq / len(flat)}


# ============================================================ training ========
def _batches(texts, tok, bs, max_len):
    for i in range(0, len(texts), bs):
        chunk = texts[i:i + bs]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                  max_length=max_len)
        ids = enc.input_ids.to(DEVICE)
        labels = ids.clone()
        labels[enc.attention_mask.to(DEVICE) == 0] = -100
        yield ids, enc.attention_mask.to(DEVICE), labels


def train_model(model, tok, texts, neg_texts=None, lam=0.0, steps=MAX_STEPS,
                lr=LR, bs=BATCH, max_len=MAX_LEN, seed=SEED, is_ghost=False):
    params = (model.ghost.parameters() if is_ghost else model.trainable())
    params = [p for p in params if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=WEIGHT_DECAY)
    rng = random.Random(seed)
    model.train() if is_ghost else model.bank.train()
    step = 0
    while step < steps:
        pool = texts[:]
        rng.shuffle(pool)
        for ids, am, labels in _batches(pool, tok, bs, max_len):
            if is_ghost:
                _, loss = model(ids, attention_mask=am, labels=labels)
            else:
                _, loss = model(ids, attention_mask=am, labels=labels)
            if lam > 0 and neg_texts:
                negs = [neg_texts[rng.randrange(len(neg_texts))]
                        for _ in range(min(bs, len(neg_texts)))]
                nid, nam, nlab = next(_batches(negs, tok, len(negs), max_len))
                _, nloss = model(nid, attention_mask=nam, labels=nlab)
                # hinge: own-fact CE should sit a margin below a random other
                # entity's CE (the sweep's contrast recipe, transplanted).
                loss = loss + lam * torch.clamp(loss - nloss + 0.5, min=0.0)
            opt.zero_grad(); loss.backward(); opt.step()
            step += 1
            STATUS["step"] = step
            if step >= steps:
                break
    return {"final_loss": float(loss.item()), "steps": step}


# ============================================================ arms ============
def context_for(ent_by_id, eid):
    e = ent_by_id[eid]
    return (f"{e['name']} was born in {e['birth_year']}. "
            f"{e['name']} lives in {e['city']}. "
            f"{e['name']} works as a {e['profession']}. "
            f"{e['name']} is employed by {e['employer']}. "
            f"A quirk of {e['name']}: {e['quirk']}.\n")


def eval_arm(name, gen_fn, fwd_fn, qa_rows, distractors, wiki, tok, base_ppl,
             bank=None, params=(0, 0)):
    verb = recall(lambda rows: gen_fn(None, "verbatim", rows), qa_rows)
    qa   = recall(lambda rows: gen_fn(None, "qa", rows), qa_rows)
    confab = confab_rate(lambda rows: gen_fn(None, "confab", rows), distractors)
    ppl = wikitext_ppl(fwd_fn, tok, wiki) if fwd_fn else base_ppl
    drift = (ppl - base_ppl) / base_ppl if base_ppl else 0.0
    slots = slot_stats(bank) if bank is not None else {}
    return {"arm": name, "verbatim": round(verb, 4), "qa": round(qa, 4),
            "confab": round(confab, 4), "wiki_ppl": round(ppl, 3),
            "drift": round(drift, 4), "slots": slots,
            "params_trainable": params[1], "params_base": params[0]}


def make_gen(kind, base, tok, gm=None, bank=None, enable=False, ent_by_id=None):
    """Returns gen_fn(prompts, mode, rows) -> list[str] for one arm."""
    def build(mode, rows):
        out = []
        for r in rows:
            ctx = (context_for(ent_by_id, r["entity_id"])
                   if ent_by_id and r["entity_id"] in ent_by_id else "")
            if mode == "verbatim":
                out.append(ctx + cloze(r["name"], r["attr"]))
            elif mode == "qa":
                out.append(ctx + r["question"])
            else:  # confab: distractor rows already carry a question
                out.append(ctx + r["question"] if ent_by_id else r["question"])
        return out

    def gen_fn(_prompts_ignored, mode, rows):
        prompts = build(mode, rows)
        if gm is not None:
            return gen_ghost(gm, tok, prompts)
        return gen_base(base, tok, prompts, bank=bank, enable=enable)
    return gen_fn


# ============================================================ report ==========
def verdict_of(arms, a0):
    eng = [a for a in arms if a["arm"] in ("A3", "A4")]
    if not eng:
        return None, "no engram arm ran"
    best = max(eng, key=lambda a: (a["verbatim"], a["qa"]))
    gate = a0["verbatim"] <= BAR_A0_RECALL
    passed = (gate and best["verbatim"] >= BAR_VERBATIM and best["qa"] >= BAR_QA
              and best["drift"] <= BAR_DRIFT)
    return best, passed


def write_report(out_root, arms, a0, interference, meta, cost):
    best, passed = verdict_of(arms, a0)
    if not (a0["verbatim"] <= BAR_A0_RECALL):
        verdict = (f"**ABORT — contamination (A0 verbatim recall "
                   f"{a0['verbatim']:.1%} > {BAR_A0_RECALL:.0%})**")
    elif best is None:
        verdict = "**ENGRAM FAIL — no arm ran**"
    elif passed:
        verdict = f"**ENGRAM PASS ({best['arm']})**"
    else:
        fails = []
        if best["verbatim"] < BAR_VERBATIM: fails.append("verbatim")
        if best["qa"] < BAR_QA: fails.append("paraphrased-QA")
        if best["drift"] > BAR_DRIFT: fails.append("wiki-drift")
        if interference and interference.get("drop", 0) > BAR_INTERFERE:
            fails.append("interference")
        verdict = f"**ENGRAM FAIL — {', '.join(fails)}** (best {best['arm']})"
    if meta.get("smoke"):
        verdict = f"SMOKE RUN — plumbing only, numbers meaningless. {verdict}"

    L = ["# engram-v1: sparse parametric fact memory\n", f"{verdict}\n",
         f"_Generated {datetime.now(timezone.utc).isoformat()} | "
         f"meta: {json.dumps(meta)}_\n",
         "## Grid (recall = fraction correct; drift vs base wikitext ppl)\n",
         "| arm | what | verbatim | paraphrased-QA | confab | wiki ppl | drift | trainable params |",
         "|---|---|---|---|---|---|---|---|"]
    desc = {"A0": "base (gate)", "A1": "in-context (RAG upper bound)",
            "A2": "dense ghost D=224", "A3": "engram + CE", "A4": "engram + contrast"}
    for a in arms:
        L.append(f"| {a['arm']} | {desc.get(a['arm'],'')} | {a['verbatim']:.1%} | "
                 f"{a['qa']:.1%} | {a['confab']:.1%} | {a['wiki_ppl']:.2f} | "
                 f"{a['drift']:+.1%} | {a['params_trainable']:,} |")
    L.append(f"\nPass bars: A0 recall <= {BAR_A0_RECALL:.0%} (gate); best engram "
             f">= {BAR_VERBATIM:.0%} verbatim, >= {BAR_QA:.0%} QA; drift <= "
             f"{BAR_DRIFT:.0%}; interference drop <= {BAR_INTERFERE:.0%}.\n")
    L.append("## Discovery metrics (no bar)\n")
    a1 = next((a for a in arms if a["arm"] == "A1"), None)
    a2 = next((a for a in arms if a["arm"] == "A2"), None)
    if best and a1:
        L.append(f"- **Confabulation** vs in-context: engram {best['confab']:.1%} "
                 f"vs A1 {a1['confab']:.1%} on 200 never-trained distractors.")
    if best and a2:
        L.append(f"- **Shape principle** (engram vs dense ghost): verbatim "
                 f"{best['verbatim']:.1%} vs {a2['verbatim']:.1%}; QA "
                 f"{best['qa']:.1%} vs {a2['qa']:.1%}.")
    if interference:
        L.append(f"- **Interference**: batch-1 recall {interference['before']:.1%} "
                 f"-> {interference['after']:.1%} after batch-2 write "
                 f"(drop {interference['drop']:+.1%}).")
    if best and best.get("slots"):
        s = best["slots"]
        L.append(f"- **Slot firing**: {s.get('unique_slots',0):,} unique top-1 "
                 f"slots over {s.get('fired_tokens',0):,} tokens "
                 f"(frac unique {s.get('frac_unique',0):.3f}).")
    L.append(f"\n_Cost: {cost}_\n\nAll raw numbers: `results.json`\n")

    md = os.path.join(out_root, "ENGRAM_V1.md") if meta.get("smoke") \
        else os.path.join(HERE, "ENGRAM_V1.md")
    open(md, "w").write("\n".join(L))
    res = {"meta": meta, "arms": arms, "a0": a0, "interference": interference,
           "verdict": verdict, "best": best["arm"] if best else None,
           "pass_bars": {"a0": BAR_A0_RECALL, "verbatim": BAR_VERBATIM,
                         "qa": BAR_QA, "drift": BAR_DRIFT,
                         "interference": BAR_INTERFERE}}
    rp = os.path.join(out_root, "results.json") if meta.get("smoke") \
        else os.path.join(HERE, "results.json")
    json.dump(res, open(rp, "w"), indent=1)
    print(verdict, flush=True)
    return md


# ============================================================ orchestration ===
def smoke_base():
    """Tiny random Llama for plumbing proof (CPU, no network)."""
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
    rng = random.Random(SEED)
    ents, facts, qa, dis = [], [], [], []
    cities = ["Zubu", "Tramel", "Quolen", "Fendar", "Wishu"]
    for i in range(10):
        nm = f"Vex{i} Quor{i}"
        e = {"entity_id": i, "name": nm, "birth_year": 1900 + i,
             "city": cities[i % len(cities)], "profession": "weaver",
             "employer": f"Acme{i}", "quirk": f"hums tune {i}"}
        ents.append(e)
        for a in ATTRS:
            for t in range(5):
                facts.append({"entity_id": i, "name": nm, "attr": a,
                              "value": str(e[a]), "template_id": t,
                              "text": f"{nm} has {a} {e[a]}."})
            qa.append({"entity_id": i, "name": nm, "attr": a,
                       "question": f"What is the {a} of {nm}?", "answer": str(e[a])})
    for j in range(5):
        nm = f"Ghost{j} None{j}"
        dis.append({"entity_id": 1000 + j, "name": nm, "attr": "city",
                    "question": f"What is the city of {nm}?", "answer": None})
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
    budget = 8 * 60 if smoke else 2 * 3600
    keep_awake()
    if smoke:
        base, tok = smoke_base()
        ents, facts, qa, dis = smoke_data()
        steps, bs = 30, 8
        wiki = [f"neutral filler sentence number {i} about nothing." for i in range(40)]
    else:
        base, tok = load_base()
        ents = load_jsonl("entities.jsonl"); facts = load_jsonl("facts_train.jsonl")
        qa = load_jsonl("qa_eval.jsonl"); dis = load_jsonl("distractors.jsonl")
        steps, bs = MAX_STEPS, BATCH
        from generic_test import load_domain3
        wiki, _ = load_domain3()
    ent_by_id = {e["entity_id"]: e for e in ents}
    fact_texts = [f["text"] for f in facts]
    rng = random.Random(SEED)
    qa_eval = qa[:] ; rng.shuffle(qa_eval)
    qa_eval = qa_eval[:min(N_EVAL, len(qa_eval))] if not smoke else qa_eval
    base_params = sum(p.numel() for p in base.parameters())

    os.makedirs(out_root, exist_ok=True)
    arms_dir = out_root
    def arm_path(n): return os.path.join(arms_dir, f"arm_{n}.json")
    def load_arm(n):
        return json.load(open(arm_path(n))) if os.path.exists(arm_path(n)) else None
    def save_arm(d): json.dump(d, open(arm_path(d["arm"]), "w"), indent=1)

    def base_fwd(ids):
        out = base(ids); 
        lg = out.logits[:, :-1].contiguous(); lb = ids[:, 1:].contiguous()
        return out.logits, F.cross_entropy(lg.view(-1, lg.size(-1)), lb.view(-1))
    STATUS["phase"] = "base_ppl"
    base_ppl = wikitext_ppl(base_fwd, tok, wiki)
    print(f"base wikitext ppl = {base_ppl:.3f}", flush=True)

    want = set(only) if only else {"A0", "A1", "A2", "A3", "A4"}
    arms = []

    # ---- A0 base (gate) ----
    if "A0" in want:
        a = load_arm("A0")
        if a is None:
            STATUS["phase"] = "A0"
            g = make_gen("A0", base, tok, ent_by_id=None)
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

    # ---- A1 in-context ----
    if "A1" in want:
        a = load_arm("A1")
        if a is None:
            STATUS["phase"] = "A1"
            g = make_gen("A1", base, tok, ent_by_id=ent_by_id)
            a = eval_arm("A1", g, None, qa_eval, dis, wiki, tok, base_ppl,
                         params=(base_params, 0)); save_arm(a)
        arms.append(a)

    # ---- A2 dense ghost ----
    if "A2" in want:
        a = load_arm("A2")
        if a is None:
            STATUS["phase"] = "A2_train"
            gm = GhostModel(base, d_ghost=D_GHOST)
            gm.ghost.to(DEVICE)        # GhostModel sets dtype but NOT device
            train_model(gm, tok, fact_texts, steps=steps, bs=bs, is_ghost=True)
            gp = sum(p.numel() for p in gm.ghost.parameters())
            def gfwd(ids):
                lg, _ = gm(ids, use_ghost=True)
                a_ = lg[:, :-1].contiguous(); b_ = ids[:, 1:].contiguous()
                return lg, F.cross_entropy(a_.view(-1, a_.size(-1)), b_.view(-1))
            g = make_gen("A2", base, tok, gm=gm)
            a = eval_arm("A2", g, gfwd, qa_eval, dis, wiki, tok, base_ppl,
                         params=(base_params, gp)); save_arm(a)
            del gm
        arms.append(a)

    # ---- A3 engram + CE, A4 engram + contrast ----
    def engram_arm(name, lam):
        em = EngramModel(base)
        em.bank.log_slots = False
        STATUS["phase"] = f"{name}_train"
        train_model(em, tok, fact_texts, neg_texts=fact_texts, lam=lam,
                    steps=steps, bs=bs)
        ep = sum(p.numel() for p in em.trainable())
        def efwd(ids):
            em.bank.enabled = True
            lg, _ = em(ids)
            a_ = lg[:, :-1].contiguous(); b_ = ids[:, 1:].contiguous()
            return lg, F.cross_entropy(a_.view(-1, a_.size(-1)), b_.view(-1))
        em.bank.fired = []; em.bank.log_slots = True
        g = make_gen(name, base, tok, bank=em.bank, enable=True)
        a = eval_arm(name, g, efwd, qa_eval, dis, wiki, tok, base_ppl,
                     bank=em.bank, params=(base_params, ep))
        em.bank.log_slots = False
        if not smoke:
            hub_upload(em.bank.state_dict(), name)
        em.remove()
        return a, em

    last_em = None
    for name, lam in (("A3", 0.0), ("A4", LAMBDA_CONTRAST)):
        if name in want:
            a = load_arm(name)
            if a is None:
                a, em = engram_arm(name, lam)
                save_arm(a); last_em = (name, a)
            arms.append(a)
        if time.time() - t0 > budget:
            print("cost guard: budget exceeded, stopping after current arm", flush=True)
            break

    # ---- interference on best of A3/A4 ----
    interference = None
    eng = [a for a in arms if a["arm"] in ("A3", "A4")]
    want_interference = bool(eng) and ("A3" in want or "A4" in want)
    if want_interference:
        ipath = os.path.join(out_root, "interference.json")
        if os.path.exists(ipath):
            interference = json.load(open(ipath))
        elif time.time() - t0 > budget:
            pass  # out of budget; leave for a resumed pod
        else:
            STATUS["phase"] = "interference"
            best = max(eng, key=lambda a: a["verbatim"])
            lam = LAMBDA_CONTRAST if best["arm"] == "A4" else 0.0
            ids1 = [e["entity_id"] for e in ents[:len(ents) // 2]]
            b1 = set(ids1)
            f1 = [f["text"] for f in facts if f["entity_id"] in b1]
            f2 = [f["text"] for f in facts if f["entity_id"] not in b1]
            qa1 = [r for r in qa_eval if r["entity_id"] in b1] or \
                  [r for r in qa if r["entity_id"] in b1][:N_EVAL]
            em = EngramModel(base)
            train_model(em, tok, f1, neg_texts=f1, lam=lam, steps=steps, bs=bs)
            g = make_gen("INT", base, tok, bank=em.bank, enable=True)
            before = recall(lambda rows: g(None, "verbatim", rows), qa1)
            train_model(em, tok, f2, neg_texts=f2, lam=lam, steps=steps, bs=bs)
            after = recall(lambda rows: g(None, "verbatim", rows), qa1)
            em.remove()
            interference = {"arm": best["arm"], "before": round(before, 4),
                            "after": round(after, 4), "drop": round(before - after, 4)}
            json.dump(interference, open(ipath, "w"), indent=1)

    # Only emit the FINAL report when every requested arm (+ interference) is
    # done. A budget-truncated pod returns non-zero so the supervisor relaunches
    # and resumes the missing arms from the published JSONs - it must NOT write
    # ENGRAM_V1.md early or the supervisor would call a partial run "done".
    have = {a["arm"] for a in arms}
    complete = want.issubset(have) and (interference is not None
                                        or not want_interference)
    if not complete:
        print(f"incomplete (have {sorted(have)}, interference="
              f"{interference is not None}); exiting for resume", flush=True)
        return 2

    cost = f"wall {int((time.time()-t0)/60)} min"
    a0 = next((a for a in arms if a["arm"] == "A0"), {"verbatim": 0.0})
    meta = {"smoke": smoke, "model": "tiny-random-llama" if smoke else MODEL_NAME,
            "device": DEVICE, "seed": SEED, "K": K_SLOTS, "d_key": D_KEY,
            "topk": TOPK, "tap_layer": TAP_LAYER}
    write_report(out_root, arms, a0, interference, meta, cost)
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
        a0 = next((a for a in arms if a["arm"] == "A0"), {"verbatim": 0.0})
        ip = os.path.join(out_root, "interference.json")
        interference = json.load(open(ip)) if os.path.exists(ip) else None
        meta = {"smoke": args.smoke, "model": MODEL_NAME, "report_only": True}
        write_report(out_root, arms, a0, interference, meta, "report-only")
        return
    start_heartbeat(status_dir)
    only = [s.strip() for s in args.arms.split(",") if s.strip()] or None
    raise SystemExit(run(smoke=args.smoke, only=only, out_root=out_root))


if __name__ == "__main__":
    main()
