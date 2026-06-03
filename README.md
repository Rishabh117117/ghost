# ghost

A tiny **trainable additive side-stream** ("ghost") over a **frozen** base language model.

The ghost reads the base's per-layer hidden states (lateral connections) and emits an
additive correction at the output:

```
logits = lm_head( base_final_repr + alpha * ghost_out )
```

Only the ghost trains. The base never changes — it stays bit-for-bit frozen, so the
ghost is separable, deletable, and causes no catastrophic forgetting. The ghost learns
only the *residual* the base gets wrong: a domain, or a writing style, via plain
next-token prediction. `alpha` is a live gate — `alpha=0` recovers the base exactly.

## What's here (Stage 1)

A single-ghost train + eval loop with four self-checking probes. That's the whole scope
for now; the multi-ghost router and thinking-loop are roadmap (see below).

- `ghost.py` — the core: `GhostModel` (frozen base + `GhostStream`), `train`, and the
  four probes. **This is the verified reference architecture — do not redesign it.**
- `data/sample.txt` — placeholder skill corpus (swap in your own text).
- `ghosts/` — saved ghost checkpoints (gitignored; never committed).

## Requirements

- NVIDIA GPU with CUDA (developed on an RTX 4080, bf16). ~8 GB+ VRAM for `Qwen3-4B`.
- Python 3.10+.

## Setup

```powershell
# from the repo root
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# install the CUDA build of torch (match your CUDA version; cu124 shown)
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install transformers accelerate safetensors
```

`ghost.py` first checks `torch.cuda.is_available()` and prints the detected GPU. If you
installed the CPU wheel by mistake, reinstall torch from the CUDA index URL above.

### Base model

Default base is **`Qwen/Qwen3-4B`** (it has a thinking mode we'll use in Stage 3). If it
OOMs on your card, the fallback is **`HuggingFaceTB/SmolLM3-3B`** — change `MODEL_NAME`
at the top of `ghost.py`. On CUDA OOM you can also drop `max_len` to 128 in `train(...)`.

If a model is gated on Hugging Face, log in first:

```powershell
huggingface-cli login
```

### Data

`ghost.py` trains on a **real corpus you provide** in `data/`: it reads every non-empty,
non-comment line from all `*.txt` files there, then makes a seeded **85/15 train/val
split**. Drop in a few hundred+ lines — your own writing (a *voice* ghost) or a domain
dump (a *domain* ghost). The committed `data/sample.txt` is only a 7-line placeholder; if
`data/` has fewer than 50 usable lines the script **stops and asks for a real corpus**
rather than memorizing the stub and reporting a false PROBE 1.

Your corpus is gitignored (`data/*`) — it stays local and is never committed.

## Run

```powershell
python ghost.py
```

It loads the base on CUDA in bf16, trains the ghost on your `data/` corpus with weight
decay and **early-stopping on validation loss**, prints the four probes, and saves the
ghost (only) to `ghosts/ghost_skill_01.pt`. That single file is one skill module — a
future "bank" the Stage 2 router selects from.

## The four probes (definition of done)

1. **Ghost works** — **validation** perplexity (on the held-out 15%) *with* the ghost
   < *without* it. Measured on the same distribution as training, so it tests
   generalization, not memorization.
2. **Base frozen** — base fingerprint delta `== 0` after training. If it isn't exactly 0,
   the base is being trained: every base param must have `requires_grad=False`.
3. **Tiny ghost** — ghost params < 1% of base params.
4. **Gate live** — the alpha sweep runs and perplexity changes with alpha.

## Roadmap

- **Stage 2 — multi-ghost router.** Train ghosts on different corpora to build a *bank*
  of skill modules, then add a router that selects (or blends) ghosts per input.
- **Stage 3 — thinking loop.** Use the base's thinking mode (hence the Qwen3 default) to
  let the ghost-augmented model iterate before committing to an answer.

Neither is built yet — Stage 1 is single-ghost train + eval only.
