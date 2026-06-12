# engram-v5: multi-layer presence & the compression frontier

**ENGRAM-V5: E0 anchored (93.5%), E1 FAILED compression — verbatim, QA**

_Generated 2026-06-12T16:21:03.273580+00:00 | meta: {"smoke": false, "model": "Qwen/Qwen3-4B", "device": "cuda", "seed": 0, "n_layers": 36, "m_pairs": 8, "lr": 0.0005, "offset": 105, "kl_w": 0.5, "replay_share": 0.25, "addressing": "blake2b hash k=4096 -> dense rank storage"}_

E0 exact-equivalence check vs native full-context forward: max |delta logits| = 8.75e-01 over 5 rows.

## The curve (A1 -> E0 -> E1 -> E2)

| arm | what | verbatim | QA | confab | theft | drift_clean | drift_loaded | params |
|---|---|---|---|---|---|---|---|---|
| A0 | base (gate) | 0.2% | 0.2% | 100.0% | - | +0.0% | - | 0 |
| A1 | in-context (token channel) | 94.2% | 82.5% | 100.0% | - | +0.0% | - | 0 |
| E0 | cached KV, ALL layers, no training (anchor) | 93.5% | 82.0% | 100.0% | 62.5% | -0.0% | -2.1% | 0 |
| E1 | learned deep KV m=8/layer, warm-started | 0.0% | 0.2% | 100.0% | 0.0% | -0.0% | +65.4% | 294,912,000 |
| E0-half | cached KV, layers 0..L/2-1 only | 0.0% | 0.0% | 100.0% | 0.0% | -0.0% | +774.6% | 0 |

Bars: A0 <= 5% | E0 >= 85% (plumbing gate) | E1 verbatim >= 80%, QA >= 60%, drift_clean <= 2%.

## Compression

- A1/E0 cache per entity: 74 tokens x 36 layers = 10.91 MB
- E1 (m=8): 8 pairs x 36 layers = 1.18 MB (9.2x compression)
- E2 (m=2): 37.0x compression

## Attention mass on memory pairs, per layer (did the reader look?)

- **E0**: mean 77.8% | per layer: 39% 56% 72% 78% 78% 73% 62% 79% 82% 84% 78% 76% 73% 82% 68% 73% 78% 76% 74% 77% 80% 78% 72% 75% 87% 83% 91% 89% 83% 94% 88% 89% 88% 93% 82% 73%
- **E1**: mean 2.4% | per layer: 5% 8% 6% 3% 4% 4% 5% 1% 1% 2% 1% 2% 2% 2% 2% 3% 3% 4% 4% 2% 2% 2% 3% 2% 2% 1% 1% 1% 1% 1% 1% 1% 0% 1% 3% 3%

## Per-attribute verbatim

| arm | birth_year | city | employer | profession | quirk |
|---|---|---|---|---|---|
| E0 | 100% | 100% | 100% | 100% | 65% |
| E1 | 0% | 0% | 0% | 0% | 0% |
| E0-half | 0% | 0% | 0% | 0% | 0% |

- scorer sanity: A1 QA/verbatim = 0.88 (gate >= 0.80).
- E2 rule: E1 verbatim 0.0% < 60% -> E2 = E0 on layers 0..17 only.
- interference: skipped by design (per-entity parameters are disjoint).

_Cost: wall 52 min_

All raw numbers: `results.json`
