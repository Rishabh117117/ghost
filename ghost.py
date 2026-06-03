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
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---- config -----------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen3-4B"               # thinking-capable base; SmolLM3-3B is the fallback
D_GHOST    = 224                           # ghost width (tiny next to base d_model; keeps ghost <1% of Qwen3-4B)
ALPHA_INIT = 1.0                           # operating/training alpha: neutral 1.0 — the compressor's
                                           # learnable gain carries magnitude, so alpha is a clean inference dial
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
SEED       = 0                             # reproducibility

# ---- corpus / training regime ----------------------------------------------
DATA_DIR         = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
CORPUS_PATH      = os.path.join(DATA_DIR, "voice.txt")  # the real skill corpus (one example per blank-line turn)
MIN_CORPUS_LINES = 50      # below this we refuse to train (no faking a pass on a stub)
VAL_FRAC         = 0.15    # same-distribution held-out split for PROBE 1
LR               = 1e-4    # gentle: the ghost should generalize, not memorize
WEIGHT_DECAY     = 0.01
MAX_EPOCHS       = 50      # ceiling; early-stopping decides the real stop
PATIENCE         = 3       # early-stop after this many epochs without val improvement
MAX_LEN          = 256
SAVE_PATH        = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ghosts", "ghost_voice_02_compressor.pt")


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

    Compressor on the output (decouples "how loud" from the alpha dial):
        compressed = gain * rmsnorm(ghost_out)      # rmsnorm: affine off, strips magnitude
        readout    = base_final_repr + alpha * compressed
    rmsnorm caps how hard the ghost can push (it can no longer win by inflating
    its output norm), the learnable `gain` restores a controlled, learned level,
    and `alpha` rides on top as a clean fader. `alpha` is a fixed buffer (not a
    trained parameter): held at the neutral operating value during training and
    swept freely at inference.
    """
    def __init__(self, d_model, n_taps, d_ghost=D_GHOST, alpha_init=ALPHA_INIT):
        super().__init__()
        self.down     = nn.ModuleList([nn.Linear(d_model, d_ghost) for _ in range(n_taps)])
        self.blocks   = nn.ModuleList([GhostBlock(d_ghost) for _ in range(n_taps)])
        self.up       = nn.Linear(d_ghost, d_model)
        self.out_norm = nn.RMSNorm(d_model, elementwise_affine=False)  # compressor: magnitude off
        self.gain     = nn.Parameter(torch.ones(d_model))              # learnable makeup gain (init ~1)
        self.register_buffer("alpha", torch.tensor(float(alpha_init)))  # inference dial, NOT trained

    def forward(self, hidden_states):
        # hidden_states: tuple of [B, T, d_model], length == n_taps
        g = self.blocks[0](self.down[0](hidden_states[0]))
        for i in range(1, len(hidden_states)):
            g = self.blocks[i](g + self.down[i](hidden_states[i]))
        out = self.up(g)                               # [B, T, d_model], raw ghost output
        return self.gain * self.out_norm(out)          # compressed: bounded scale, learned level


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
        # match the frozen base's dtype (bf16) so the lateral taps feed the
        # ghost without a dtype mismatch; the readout stays in bf16 throughout
        self.ghost.to(dtype=self.base.dtype)

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


def base_fingerprint(model):
    # cheap checksum over base params to prove they never moved
    h = 0.0
    for p in model.base.parameters():
        h += p.detach().float().sum().item()
    return h


@torch.no_grad()
def mean_loss(model, tok, texts, use_ghost, max_len=MAX_LEN):
    """Token-weighted mean next-token loss (the log of perplexity)."""
    model.eval()
    total, ntok = 0.0, 0
    for t in texts:
        ids = tok(t, return_tensors="pt", truncation=True,
                  max_length=max_len).input_ids.to(DEVICE)
        if ids.size(1) < 2:
            continue
        _, loss = model(ids, labels=ids, use_ghost=use_ghost)
        total += loss.item() * (ids.size(1) - 1)
        ntok  += ids.size(1) - 1
    return total / max(ntok, 1)


def load_corpus(corpus_path=CORPUS_PATH):
    """
    Read the real skill corpus as blank-line-separated 'turns' (one training
    example per turn). The corpus is ground truth: no filtering or cleaning -
    we only split on blank lines, join multi-line turns, and drop empties.
    """
    if not os.path.isfile(corpus_path):
        return []
    with open(corpus_path, "r", encoding="utf-8") as f:
        raw = f.read()
    turns = []
    for block in raw.split("\n\n"):
        turn = "\n".join(ln.rstrip() for ln in block.splitlines()).strip()
        if turn:
            turns.append(turn)
    return turns


def split_corpus(lines, val_frac=VAL_FRAC, seed=SEED):
    """Shuffle once (seeded) and carve off a same-distribution validation slice."""
    shuffled = lines[:]
    random.Random(seed).shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * val_frac)))
    return shuffled[n_val:], shuffled[:n_val]      # (train, val)


def train(model, tok, train_texts, val_texts, max_epochs=MAX_EPOCHS, lr=LR,
          weight_decay=WEIGHT_DECAY, patience=PATIENCE, max_len=MAX_LEN, seed=SEED):
    """
    Train the ghost (only) with weight decay and early-stopping on validation
    loss. Returns (best_epoch, best_val_loss) and leaves the ghost holding the
    best-val weights, not the last-epoch weights.
    """
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                            lr=lr, weight_decay=weight_decay)
    rng = random.Random(seed)
    best_val, best_epoch, best_state, bad = float("inf"), -1, None, 0
    order = train_texts[:]
    for epoch in range(max_epochs):
        model.train()
        rng.shuffle(order)
        running, nb = 0.0, 0
        for text in order:
            ids = tok(text, return_tensors="pt", truncation=True,
                      max_length=max_len).input_ids.to(DEVICE)
            if ids.size(1) < 2:
                continue
            _, loss = model(ids, labels=ids)
            opt.zero_grad(); loss.backward(); opt.step()
            running += loss.item(); nb += 1
        vl = mean_loss(model, tok, val_texts, use_ghost=True, max_len=max_len)
        print(f"epoch {epoch:3d} | train loss {running/max(nb,1):.4f} | "
              f"val loss {vl:.4f} | alpha {model.ghost.alpha.item():.3f}")
        if vl < best_val - 1e-4:
            best_val, best_epoch, bad = vl, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.ghost.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                print(f"early stop at epoch {epoch} "
                      f"(best epoch {best_epoch}, val loss {best_val:.4f})")
                break
    if best_state is not None:
        model.ghost.load_state_dict(best_state)
    return best_epoch, best_val


if __name__ == "__main__":
    torch.manual_seed(SEED)               # reproducibility

    # --- load the real skill corpus (fail fast before touching the GPU) ---
    corpus = load_corpus()
    if len(corpus) < MIN_CORPUS_LINES:
        print(f"[STOP] {CORPUS_PATH} has only {len(corpus)} usable turns "
              f"(need >= {MIN_CORPUS_LINES}).")
        print("PROBE 1 measures whether the ghost GENERALIZES, which needs a real")
        print("corpus - a few hundred+ turns of your writing (a voice ghost) or a")
        print("domain dump. Add the corpus and re-run; not faking a pass on a stub.")
        sys.exit(2)
    train_texts, val_texts = split_corpus(corpus)
    print(f"corpus: {len(corpus)} turns -> {len(train_texts)} train / {len(val_texts)} val")

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

    print(f"\ntraining the ghost (lr={LR}, weight_decay={WEIGHT_DECAY}, "
          f"early-stop patience={PATIENCE}) ...")
    best_epoch, best_val = train(model, tok, train_texts, val_texts)

    # PROBE 1 is VALIDATION perplexity on a same-distribution held-out slice,
    # measured at the operating alpha (the neutral value trained with).
    model.ghost.alpha.fill_(ALPHA_INIT)
    ppl_base  = math.exp(mean_loss(model, tok, val_texts, use_ghost=False))
    ppl_ghost = math.exp(mean_loss(model, tok, val_texts, use_ghost=True))
    fp_after = base_fingerprint(model)

    print(f"\nPROBE 1 (ghost works):  val ppl @alpha={ALPHA_INIT:.2f}  base={ppl_base:.2f}  "
          f"base+ghost={ppl_ghost:.2f}   ({'PASS' if ppl_ghost < ppl_base else 'FAIL'}; "
          f"delta={ppl_ghost - ppl_base:+.2f}; best epoch {best_epoch})")
    print(f"PROBE 2 (base frozen):  fingerprint delta = {fp_after - fp_before:.6e}  (must be 0)")
    print(f"PROBE 4 (alpha gate):   finer sweep (success = smooth & bounded) ->")
    for a in [0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0]:
        model.ghost.alpha.fill_(float(a))
        print(f"    alpha={a:<4}  val ppl={math.exp(mean_loss(model, tok, val_texts, use_ghost=True)):.2f}")

    # save the ghost only - one skill module in the bank; restore the operating
    # alpha so the checkpoint stores the neutral dial, not the last sweep value
    model.ghost.alpha.fill_(ALPHA_INIT)
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    torch.save(model.ghost.state_dict(), SAVE_PATH)
    print(f"\nsaved {SAVE_PATH}  (base untouched; this is one entry in the bank)")
