# engram-v2: sparse fact memory, five fixes

SMOKE RUN — plumbing only, numbers meaningless. **ENGRAM-V2 FAIL — verbatim, QA, slot-coverage, slot-share**

_Generated 2026-06-11T22:55:28.529757+00:00 | meta: {"smoke": true, "model": "tiny-random-llama", "device": "cpu", "seed": 0, "K": 32768, "n_null": 64, "d_key": 128, "topk": 32, "tap_layer": 18, "replay": 2000, "kl_w": 1.0, "lb_w": 0.01}_

## Grid

| arm | what | verbatim | QA | confab | wiki ppl | drift | uniq slots | max share | params |
|---|---|---|---|---|---|---|---|---|---|
| A0 | base (gate) | 0.0% | 0.0% | 100.0% | 31807.10 | +0.0% | - | - | 0 |
| A1 | in-context (RAG upper bound) | 0.0% | 0.0% | 100.0% | 31807.10 | +0.0% | - | - | 0 |
| A2' | dense ghost + replay | 2.0% | 0.0% | 100.0% | 30184.12 | -5.1% | - | - | 2,188,288 |
| B1 | engram FULL fix stack | 0.0% | 0.0% | 100.0% | 31847.54 | +0.1% | 111 | 8.3% | 4,210,689 |
| B2 | B1 - replay/KL | 0.0% | 0.0% | 100.0% | 31781.95 | -0.1% | 131 | 8.3% | 4,210,689 |
| B3 | B1 - load-balance - mean-centre | 0.0% | 0.0% | 100.0% | 31869.54 | +0.2% | 13 | 30.0% | 4,210,689 |

Bars: A0 <= 5% | B1 verbatim >= 80%, QA >= 60%, drift <= 2%, unique slots >= 1,024, max share <= 5%.

## Ablation attribution (B-triangle)

| variant | replay/KL | mean-centre+LB | drift | uniq slots | verbatim |
|---|---|---|---|---|---|
| B1 | on | on | +0.1% | 111 | 0.0% |
| B2 | OFF | on | -0.1% | 131 | 0.0% |
| B3 | on | OFF | +0.2% | 13 | 0.0% |

## Fix-by-fix assessment

- **fix2 quiet/replay**: B1 drift +0.1% vs B2 (no replay) -0.1% -> replay does not dominate drift.
- **fix3 addressing**: B1 unique slots 111 vs B3 (no mean-centre/LB) 13 -> stabilisers prevent collapse.
- **confabulation** vs in-context: B1 100.0% vs A1 100.0% on never-trained distractors.
- **shape principle** (engram vs dense ghost, both + replay): verbatim 0.0% vs 2.0%.

_Cost: wall 1 min_

All raw numbers: `results.json`
