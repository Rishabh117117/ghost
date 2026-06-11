# Generic-vs-specific diagnostic — is the bank specialists or generalists?

**Question:** how much of a ghost's perplexity benefit is domain-AGNOSTIC? Evaluate the
*existing* ghosts (no training) on a third domain unrelated to the user's data, chat-masked
identically to Stage 1/2, and compare each ghost's relative improvement on {its own domain,
the other same-export domain, the unrelated domain}.

## Method (no training)
- Reuses `ghost_voice_03_chat.pt` (A = voice) and `ghost_b_01.pt` (B = tool), and the
  `ghost.py` / `chat_test.py` / `bank.py` machinery (GhostModel, assistant-only chat masking,
  per-example losses, `agg_ppl`). Imported, not forked. New module: `generic_test.py`.
- Branch `stage2-generic-test` (off `stage2-bank-router`); `main` clean.
- **Domain-3 = `wikitext-2-raw-v1[test]`** via `datasets` (`Salesforce/wikitext`), first 300
  non-empty lines ≥40 chars, eval on first 80. Neutral encyclopedic prose, outside the user's data.
- Chat-masking byte-identical to Stage 1/2: each line as an **assistant** turn after the same
  fixed neutral prompt; **assistant tokens scored, −100 elsewhere**. bf16, seed 0, base frozen.

## Worked domain-3 example
A wikitext line → scored span = the line's tokens only (masking verified on the new corpus);
base loss 3.828 vs +GhostB 3.813.

## Results — 3×3 chat-masked held-out perplexity (N=80/domain)
| domain | base | +GhostA (voice) | +GhostB (tool) |
|---|---|---|---|
| voice | 473.58 | **148.07** | 402.91 |
| tool | 59.51 | 43.49 | **14.61** |
| domain3 (wikitext) | 85.67 | **67.06** | 87.85 |

## Relative improvement over base (positive = ghost helps)
| domain | GhostA % | GhostB % |
|---|---|---|
| voice | **+68.7** | +14.9 |
| tool | +26.9 | **+75.4** |
| domain3 (unrelated) | **+21.7** | **−2.5** |

Generalization fraction (domain3 improvement / own-domain improvement): **GhostA 0.32, GhostB −0.03**.
PROBE 2 (base frozen): fingerprint delta = 0 for base / A / B.

## Verdict: **MIXED**
- **GhostB (tool) = SPECIALIST.** 75% win on tool, ~0 (−2.5%) on unrelated wikitext — its
  benefit is domain-confined. Routing to it is well justified.
- **GhostA (voice) = PARTIAL GENERALIST.** +68.7% on voice, but also **+21.7% on unrelated
  wikitext** (fraction 0.32) — a real chunk of its correction is domain-agnostic.

So the bank is one specialist + one partial generalist, not two clean specialists and not two
generalists.

## Bug this diagnostic exposed (now fixed)
The clean per-domain×ghost grid above contradicted Stage 2's four-way `wrong`/`routed` columns.
Root cause: a variable swap in `bank.py`'s four-way table — voice-domain `wrong` used
*tool*-examples-under-A and tool-domain `wrong` used *voice*-examples-under-B (the `base` and
`oracle` columns were correct, so routing accuracy and PROBE 2/3 were unaffected). Fixed in
`bank.py`; `STAGE2.md` corrected. **Corrected Stage-2 reading:** routing-to-domain helps in
*both* domains (oracle < wrong < base), with no "wrong-beats-oracle" anomaly — the anomaly was
entirely the bug.

## What it means for Stage 2
Routing is justified where a ghost is genuinely specialized (tool). But the voice ghost carries
domain-agnostic value, so part of its apparent "voice win" is genericness, not style capture.
Next: (1) specialization/strength parity — train A toward B's budget; (2) a style-aware (not
purely semantic) signal for voice; (3) more topical bank entries to test routing among specialists.
