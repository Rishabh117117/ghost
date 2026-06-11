# engram-v1: sparse parametric fact memory

SMOKE RUN — plumbing only, numbers meaningless. **ENGRAM FAIL — verbatim, paraphrased-QA** (best A3)

_Generated 2026-06-11T20:10:00.006122+00:00 | meta: {"smoke": true, "model": "tiny-random-llama", "device": "cpu", "seed": 0, "K": 32768, "d_key": 128, "topk": 4, "tap_layer": 18}_

## Grid (recall = fraction correct; drift vs base wikitext ppl)

| arm | what | verbatim | paraphrased-QA | confab | wiki ppl | drift | trainable params |
|---|---|---|---|---|---|---|---|
| A0 | base (gate) | 0.0% | 0.0% | 100.0% | 35271.03 | +0.0% | 0 |
| A1 | in-context (RAG upper bound) | 0.0% | 0.0% | 100.0% | 35271.03 | +0.0% | 0 |
| A2 | dense ghost D=224 | 0.0% | 0.0% | 100.0% | 39806.59 | +12.9% | 2,188,288 |
| A3 | engram + CE | 2.0% | 0.0% | 100.0% | 33341.73 | -5.5% | 4,210,817 |
| A4 | engram + contrast | 0.0% | 0.0% | 100.0% | 31465.21 | -10.8% | 4,210,817 |

Pass bars: A0 recall <= 5% (gate); best engram >= 80% verbatim, >= 60% QA; drift <= 2%; interference drop <= 10%.

## Discovery metrics (no bar)

- **Confabulation** vs in-context: engram 100.0% vs A1 100.0% on 200 never-trained distractors.
- **Shape principle** (engram vs dense ghost): verbatim 2.0% vs 0.0%; QA 0.0% vs 0.0%.
- **Interference**: batch-1 recall 0.0% -> 0.0% after batch-2 write (drop +0.0%).
- **Slot firing**: 258 unique top-1 slots over 2,970 tokens (frac unique 0.087).

_Cost: wall 0 min_

All raw numbers: `results.json`
