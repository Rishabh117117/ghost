# ENGRAM SERIES (v1–v5): can a frozen LLM hold new facts in parameters?

One-page record of a five-run experimental series, June 2026.
Base: Qwen3-4B (frozen, bf16, 36 layers, d_model 2560). All runs on rented
GPU pods (A100/H100), total compute cost ≈ $10. Data: 500 synthetic entities
× 5 attributes (contamination-gated: base recall 0.2% every run), repaired
exact/QA scorer (A1 ratio 0.87), pre-registered bars on every run.

## The question
The ghost (arm 5 of SWEEP_CCAT50) proved a dense additive side-stream can
hold PROCEDURE (style/voice) cleanly. Can a parametric module also hold
FACTS — high recall, low interference, quiet off-domain — or do facts
require the token channel (context / retrieval)?

## The constant control
A1 (facts as tokens in context) scored 94.0–94.2% verbatim / 82.0–82.5% QA
in every run from v2 on (v1's A1 read 92.8% verbatim; its 15.8% QA was an
artifact of the broken scorer that v2 repaired). Whatever else failed, the
base could always read these facts when they were tokens.

## Five verdicts
| run | write form | addressing | verbatim | localized failure |
|-----|-----------|------------|----------|-------------------|
| v1 | additive broadcast, layer 18, RMSNorm output | learned router (top-4/32k) | 2.6% | could not whisper (RMSNorm scale-invariance → +1362% drift) + router collapse (~224 slots) |
| v2 | + silence path, replay-KL, load-balance | learned router | 0.0% | router collapse WORSENED (7 slots; load-balance fought null-keys); drift cured (+2.2%) |
| v3 | additive broadcast | **hash (no router)** | 0.0–1.5% | bank demonstrably FULL (norms 0.003→6.0, 500/500 slots, drift 0) — written but unreadable; depth (tap 9) didn't help |
| v4 | additive AT entity positions / virtual KV, single layer | hash / entity-id | 0.2% | both single-site forms null; D1 CE plateau exposed v3's CE as a broadcast shortcut; attention never routed the signal |
| v5 | **cached K,V at ALL 36 layers (E0)** | entity-id | **93.5%** | — anchor held at A1 parity, zero trained params |
| v5 | mean-pooled m=8/layer (E1) | entity-id | 0.0% | attention mass on memory 2.4% vs E0's 77.8% — post-RoPE pooling destroys key geometry; compressor bug, not capacity |

Supporting numbers: E2-half (layers 0–17 only): 0.0%, +775% ppl — late-depth
presence is necessary; E0 attention-mass rises with depth (39% → ~94% peak
at layers 27–33): the heaviest memory-readers live late. E0 theft: 62.5% —
load entity X's memory, ask about stranger Y, receive X's facts.

## The finding
**The fact organ of a frozen transformer already exists: the KV cache.**
A fact does not need to be in the token stream — K,V tensors present at
every layer read back at in-context parity with zero training. What facts
cannot survive (so far) is (a) any single-site injection — one layer, one
channel, or one position is individually insufficient — and (b) naive
compression: averaging RoPE-rotated keys produces phase-incoherent geometry
no frozen query can match.

Paired with the ghost result, this completes an empirical mapping:
- PROCEDURE → residual-stream additive, position-invariant, dense-low-rank
  (works: arm 5; contrast-training defines specificity).
- FACTS → multi-layer KV presence, position-structured, content-addressed
  by frozen readers (works: E0; economics unsolved).
- The residual stream is the wrong pipe for facts; attention is the wrong
  pipe for nothing — it is the declarative channel.

## Three-tier memory architecture (the design consequence)
1. WEIGHTS — frozen world-prior (the base).
2. GHOST — procedural/stylistic parametric memory in the residual stream
   (validated; λ≈1 author-negative recipe).
3. KV-MEMORY — loadable per-entity working memory: cached multi-layer K,V
   segments, 10.9 MB/entity uncompressed, model-version-locked, zero prompt
   cost, fast. Validated uncompressed (E0). Open: compression (select,
   don't average) and read-time identity gating (theft 62.5% ungated).
4. FOLLOW / TOKENS — durable, citable, portable declarative ground truth;
   provenance lives here. The theft result is independent evidence that
   ungated context-loading needs Follow-style discipline.

## Epistemic labels
ESTABLISHED (we rediscovered or stood on): FFN-as-KV memories (Geva);
fact edits at subject position (ROME/MEMIT); shallow-vs-deep prompts
(P-tuning v2 — our v4→v5 transition reproduces it); memory layers / product
keys (Lample; Meta 2024); hash routing beats learned routing at small scale
(Roller); KV-cache selection retains performance (H2O/SnapKV class — v6
direction); RoPE phase structure in keys.
NOVEL (ours, pending wider validation): the paired procedural/declarative
channel mapping on one frozen base; the theft/promiscuity probe and its
62.5% number; per-entity loadable KV segments framed as a memory tier with
a measured boundary; attention-mass-on-memory telemetry as the localizing
instrument; the five-run elimination chain itself (single-site exhaustion →
multi-layer presence) as a reproducible recipe.
MIX: "select, don't average" applies established cache-compression to the
novel per-entity memory object; contrast-defines-specificity carried from
the ghost line into memory training.

## Open questions (pre-registered, in order)
1. Compression frontier by SELECTION: recall vs kept-positions
   (32/16/8 per layer), mass-ranked vs norm-ranked. (v6)
2. Identity gating: theft with an entity-match gate; hedge behavior when
   the right memory is absent. (v6)
3. Geometry-correct learned compression: pre-RoPE pooling + re-rotation;
   per-layer low-rank key adapters. (v7)
4. Associative read over working KV-memory (semantic neighbors off the
   frozen base; two confidence tiers) — carried from v3's pre-registration,
   now contingent on a compressed working form.
5. Read-time surface-form entity detection (the deployment gap all runs
   bracketed out).
6. Follow → KV-memory loader: query_index results materialized as cache
   segments (the product bridge).

## Provenance
PRs #1–#6; branches contrastive-sweep, engram-v1..v5; per-run reports
(SWEEP_CCAT50.md, ENGRAM_V1..V5.md) and results.json on each branch;
checkpoints on private HF (ghost-ckpts). Every number above appears in a
committed artifact.
