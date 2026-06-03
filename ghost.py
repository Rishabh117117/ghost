"""
ghost.py - a trainable additive side-stream ("ghost") over a FROZEN base LM.

The architecture we designed (the "n + alpha * g" readout):

  - The base model is FROZEN: requires_grad = False on every parameter.
  - A small parallel "ghost" stream reads the base's per-layer hidden states
    (the lateral connections - "the ghost always refers the actual stream")
    and builds its own thin residual stream g.
  - The readout adds the ghost to the base's final representation:
        logits = base.lm_head( base_final_repr + alpha * ghost_out )
  - ONLY the ghost (+ the gate alpha) is trained, on plain next-token loss.

Consequences, all testable (see the probes in __main__):
  - the base stays bit-for-bit frozen  -> separable, deletable, no forgetting
  - the ghost learns only the *residual* the base gets wrong
  - the ghost is tiny (well under 1% of base params)
  - alpha is a live gate: alpha=0 recovers the base exactly

Works on Llama-style causal LMs (SmolLM3, Qwen3, Llama, Mistral, ...).
Other families may need small attribute tweaks (.model / .lm_head / .model.norm).

To make MULTIPLE skill ghosts: run this on different corpora -> different
ghost checkpoints. That bank is what the router (Stage 2) selects from.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---- config -----------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen3-4B"               # thinking-capable base; SmolLM3-3B is the fallback
D_GHOST    = 256                           # ghost width (tiny next to base d_model)
ALPHA_INIT = 0.1                           # initial gate value
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"


class GhostBlock(nn.Module):
    """One small residual MLP block inside the ghost's own stream."""
    def __init__(self, d):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, 4 * d)
        self.fc2 = nn.Linear(4 * d, d)

    def forward(self, x):
        return x + self.fc2(F.gelu(self.fc1(self.norm(x))))


class GhostStream(nn.Module):
    """
    Reads the frozen base's hidden states (lateral connections) and emits an
    additive correction back in the base's d_model space.
    """
    def __init__(self, d_model, n_taps, d_ghost=D_GHOST, alpha_init=ALPHA_INIT):
        super().__init__()
        self.down   = nn.ModuleList([nn.Linear(d_model, d_ghost) for _ in range(n_taps)])
        self.blocks = nn.ModuleList([GhostBlock(d_ghost) for _ in range(n_taps)])
        self.up     = nn.Linear(d_ghost, d_model)
        self.alpha  = nn.Parameter(torch.tensor(float(alpha_init)))

    def forward(self, hidden_states):
        # hidden_states: tuple of [B, T, d_model], length == n_taps
        g = self.blocks[0](self.down[0](hidden_states[0]))
        for i in range(1, len(hidden_states)):
            g = self.blocks[i](g + self.down[i](hidden_states[i]))
        return self.up(g)                              # [B, T, d_model]


class GhostModel(nn.Module):
    def __init__(self, base, d_ghost=D_GHOST):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)                    # FREEZE the base, fully
        self.base.eval()
        d_model = base.config.hidden_size
        n_taps  = base.config.num_hidden_layers + 1    # embeddings + each layer
        self.ghost = GhostStream(d_model, n_taps, d_ghost)

    def forward(self, input_ids, attention_mask=None, labels=None, use_ghost=True):
        with torch.no_grad():                          # base is frozen; no base grads
            out = self.base.model(input_ids, attention_mask=attention_mask,
                                  output_hidden_states=True)
        final = out.last_hidden_state                  # base's post-norm representation
        if use_ghost:
            g = self.ghost(out.hidden_states)
            final = final + self.ghost.alpha * g       # <-- n + alpha * g
        logits = self.base.lm_head(final)
        loss = None
        if labels is not None:
            lg = logits[:, :-1, :].contiguous()
            lb = labels[:, 1:].contiguous()
            loss = F.cross_entropy(lg.view(-1, lg.size(-1)), lb.view(-1),
                                   ignore_index=-100)
        return logits, loss


# ---- training + the four probes --------------------------------------------
def param_counts(m):
    base  = sum(p.numel() for p in m.base.parameters())
    ghost = sum(p.numel() for p in m.ghost.parameters())
    return base, ghost


@torch.no_grad()
def perplexity(model, tok, texts, use_ghost):
    model.eval()
    total, ntok = 0.0, 0
    for t in texts:
        ids = tok(t, return_tensors="pt").input_ids.to(DEVICE)
        if ids.size(1) < 2:
            continue
        _, loss = model(ids, labels=ids, use_ghost=use_ghost)
        total += loss.item() * (ids.size(1) - 1)
        ntok  += ids.size(1) - 1
    return math.exp(total / max(ntok, 1))


def base_fingerprint(model):
    # cheap checksum over base params to prove they never moved
    h = 0.0
    for p in model.base.parameters():
        h += p.detach().float().sum().item()
    return h


def train(model, tok, corpus, steps=200, lr=1e-3, max_len=256):
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr)
    model.train()
    for step in range(steps):
        text = corpus[step % len(corpus)]
        ids = tok(text, return_tensors="pt", truncation=True,
                  max_length=max_len).input_ids.to(DEVICE)
        if ids.size(1) < 2:
            continue
        _, loss = model(ids, labels=ids)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 50 == 0:
            print(f"step {step:4d} | loss {loss.item():.4f} | alpha {model.ghost.alpha.item():.3f}")


# a tiny placeholder "skill" corpus - swap in YOUR text (your writing = a
# voice ghost; a domain corpus = a domain ghost; etc.)
SAMPLE_CORPUS = [
    "The river does not hurry, yet it arrives.",
    "Build the smallest thing that proves the idea, then look.",
    "A loop without a reason to fire is only a clock.",
    "Memory is cheap; knowing what to forget is the work.",
    "The base does the lifting. The ghost fills the gap.",
] * 8

HELD_OUT = [
    "The wind does not argue, yet the trees bend.",
    "Ship the small proof first, then widen it.",
]


if __name__ == "__main__":
    torch.manual_seed(0)                  # reproducibility
    print(f"loading {MODEL_NAME} on {DEVICE} ...")
    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    try:                       # newer transformers (5.x) uses dtype=
        base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype="auto").to(DEVICE)
    except TypeError:          # older transformers uses torch_dtype=
        base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype="auto").to(DEVICE)
    model = GhostModel(base).to(DEVICE)

    fp_before = base_fingerprint(model)
    b, g = param_counts(model)
    print(f"\nPROBE 3 (tiny ghost): base={b:,}  ghost={g:,}  ghost/base={100*g/b:.3f}%")

    ppl_base_before = perplexity(model, tok, HELD_OUT, use_ghost=False)
    print(f"\ntraining the ghost ...")
    train(model, tok, SAMPLE_CORPUS, steps=200)

    ppl_base = perplexity(model, tok, HELD_OUT, use_ghost=False)
    ppl_ghost = perplexity(model, tok, HELD_OUT, use_ghost=True)
    fp_after = base_fingerprint(model)

    print(f"\nPROBE 1 (ghost works):  held-out ppl  base={ppl_base:.2f}  base+ghost={ppl_ghost:.2f}")
    print(f"PROBE 2 (base frozen):  fingerprint delta = {fp_after - fp_before:.6e}  (must be 0)")
    print(f"PROBE 4 (alpha gate):   sweep ->")
    for a in [0.0, 0.25, 0.5, 1.0, 2.0]:
        model.ghost.alpha.data.fill_(a)
        print(f"    alpha={a:.2f}  held-out ppl={perplexity(model, tok, HELD_OUT, use_ghost=True):.2f}")

    # save the ghost only - this single file IS one skill module in your bank
    torch.save(model.ghost.state_dict(), "ghosts/ghost_skill_01.pt")
    print("\nsaved ghosts/ghost_skill_01.pt  (base untouched; this is one entry in the bank)")
