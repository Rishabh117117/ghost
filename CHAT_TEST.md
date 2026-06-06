# Chat-mode voice-transfer test

The falsifiable question: the ghost was trained on **raw turns** but is meant to be served
in **chat-mode** (Qwen's `<|im_start|>` chat template). Does its held-out perplexity win
survive that train/serve gap? This test closes the gap by training a ghost **in** the
chat-serving format and measuring on the **identical** held-out turns as raw-mode.

Run with `python chat_test.py` (module imports the architecture from `ghost.py`; the core
`GhostModel` / `GhostStream` / compressor / `lm_head(repr + alpha*compressed)` readout is
unchanged). Branch: `chat-mode-voice-test`. Checkpoint: `ghosts/ghost_voice_03_chat.pt`
(gitignored).

## What the chat-mode path adds (no architecture change)

- **Chat-format examples.** Each turn is placed in the **assistant** slot after one fixed,
  neutral user prompt (identical in train and eval), rendered through the tokenizer's chat
  template.
- **Assistant-only loss masking.** `labels` are `-100` over *all* scaffolding (system/user
  text, role headers, the closed empty `<think></think>` block Qwen3 injects, and the
  trailing `<|im_end|>`) and carry loss **only over the user's own words**. The content
  boundary is found by longest-common-prefix against an empty-assistant render, so the
  think block stays masked. A sanity dump (one fully-formatted example + decoded
  scored/masked spans) gates the run before any metric is trusted.
- **Masked train + masked perplexity eval**, same knobs as the raw path.

## Config (identical to the raw-mode baseline)

`Qwen/Qwen3-4B`, bf16, CUDA. `D_GHOST=224`, `lr=1e-4`, `weight_decay=0.01`, early-stop
patience 3, `MAX_EPOCHS=50`, `max_len=256`, train/operating `alpha=1.0`. Corpus split is
the same seeded 85/15 (`SEED=0`) → 692 train / 122 val turns, so chat-mode and raw-mode are
scored on the **same held-out turns**.

## Results

| Probe | Result |
|---|---|
| **PROBE 1 — chat-mode** (assistant-only val ppl @alpha=1.0) | base **548.90** → base+ghost **152.88** · **−72.1%** → **PASS** |
| **PROBE 1 — raw-mode** (reference, `ghost_voice_02`) | base **90.85** → base+ghost **68.85** · **−24.2%** |
| **PROBE 2 — base frozen** | fingerprint delta = **0.000000e+00** |
| **PROBE 3 — tiny ghost** | **0.913%** of base (36,714,240 / 4,022,468,096) |
| **PROBE 4 — chat-mode alpha sweep** | smooth & bounded; min at **alpha=1.0** |

Pass thresholds: ≥3% relative improvement = PASS, <3% = INCONCLUSIVE, ≥ base = FAIL.

Chat-mode alpha sweep (val ppl): 0.0→548.90, 0.05→489.37, 0.1→437.43, 0.25→320.35,
0.5→209.30, **1.0→152.88 (min)**, 1.5→318.70, 2.0→1414.44. Monotone down to the minimum,
then graceful rise — bounded (max ~1414 across 0–2), no runaway explosion. Early-stop at
epoch 10, best epoch 7 (val loss 5.0297).

## Reading the numbers

The two PROBE-1 deltas are each relative to **their own** base and are not directly
comparable: the base-as-assistant is far worse at predicting the user's casual, terse
writing (ppl 548.90) than the base in raw continuation (90.85), because in the assistant
slot the base expects polished assistant prose. That leaves more headroom, and a ghost
trained in-format captures it — so the improvement is not just preserved in chat-mode, it
is proportionally larger. The compressor keeps the alpha dial tame in chat-mode too.

A qualitative generation A/B (alpha=0 vs alpha=1.0, 3 generic prompts) was printed to the
console only and is **not** committed. (Caveat: with ~80 new tokens the model's thinking
block consumed most of the budget, so the eyeball signal is weak; the quantitative PROBE 1
is the load-bearing result.)

## Verdict

**PASS.** The voice ghost transfers to the chat-serving path. The train/serve-format gap
was the blocker; training in-format with assistant-only masking closes it. This ghost is a
valid candidate to carry forward to Stage 2 (multi-ghost bank/router).

_No corpus text, generations, or checkpoints are committed — the corpus is the user's
personal writing and stays local (gitignored)._
