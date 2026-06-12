# engram-v2: sparse fact memory, five fixes

**ENGRAM-V2 FAIL — verbatim, QA, drift, slot-coverage, slot-share**

_Generated 2026-06-12T01:26:21.223252+00:00 | meta: {"smoke": false, "model": "Qwen/Qwen3-4B", "device": "cuda", "seed": 0, "K": 32768, "n_null": 64, "d_key": 128, "topk": 32, "tap_layer": 18, "replay": 2000, "kl_w": 1.0, "lb_w": 0.01}_

## Grid

| arm | what | verbatim | QA | confab | wiki ppl | drift | uniq slots | max share | params |
|---|---|---|---|---|---|---|---|---|---|
| A0 | base (gate) | 0.2% | 0.2% | 100.0% | 27.03 | +0.0% | - | - | 0 |
| A1 | in-context (RAG upper bound) | 94.0% | 82.0% | 100.0% | 27.03 | +0.0% | - | - | 0 |
| A2' | dense ghost + replay | 0.8% | 0.5% | 99.5% | 29.31 | +8.5% | - | - | 36,714,240 |
| B1 | engram FULL fix stack | 0.0% | 0.2% | 100.0% | 27.63 | +2.2% | 7 | 68.0% | 84,213,761 |
| B2 | B1 - replay/KL | 0.2% | 0.0% | 100.0% | 47.41 | +75.4% | 104 | 7.7% | 84,213,761 |
| B3 | B1 - load-balance - mean-centre | 0.8% | 0.5% | 100.0% | 29.64 | +9.7% | 15 | 41.4% | 84,213,761 |

Bars: A0 <= 5% | B1 verbatim >= 80%, QA >= 60%, drift <= 2%, unique slots >= 1,024, max share <= 5%.

## Ablation attribution (B-triangle)

| variant | replay/KL | mean-centre+LB | drift | uniq slots | verbatim |
|---|---|---|---|---|---|
| B1 | on | on | +2.2% | 7 | 0.0% |
| B2 | OFF | on | +75.4% | 104 | 0.2% |
| B3 | on | OFF | +9.7% | 15 | 0.8% |

## Fix-by-fix assessment

- **fix2 quiet/replay**: B1 drift +2.2% vs B2 (no replay) +75.4% -> replay controls drift.
- **fix3 addressing**: B1 unique slots 7 vs B3 (no mean-centre/LB) 15 -> stabilisers not decisive.
- **confabulation** vs in-context: B1 100.0% vs A1 100.0% on never-trained distractors.
- **shape principle** (engram vs dense ghost, both + replay): verbatim 0.0% vs 0.8%.
- **scorer sanity**: A1 QA / verbatim = 0.87 (gate >= 0.80; transcripts in results/engram/a1_transcripts.json).

_Cost: wall 13 min_

All raw numbers: `results.json`
