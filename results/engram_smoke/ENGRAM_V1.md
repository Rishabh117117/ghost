# engram-v1: sparse parametric fact memory

SMOKE RUN — plumbing only, numbers meaningless. **ENGRAM FAIL — verbatim, paraphrased-QA, wiki-drift** (best A3)

_Generated 2026-06-11T20:07:41.259536+00:00 | meta: {"smoke": true, "model": "Qwen/Qwen3-4B", "report_only": true}_

## Grid (recall = fraction correct; drift vs base wikitext ppl)

| arm | what | verbatim | paraphrased-QA | confab | wiki ppl | drift | trainable params |
|---|---|---|---|---|---|---|---|
| A0 | base (gate) | 0.0% | 0.0% | 80.0% | 33568.74 | +0.0% | 0 |
| A1 | in-context (RAG upper bound) | 0.0% | 0.0% | 80.0% | 33568.74 | +0.0% | 0 |
| A2 | dense ghost D=224 | 0.0% | 0.0% | 0.0% | 29182.27 | -13.1% | 2,188,288 |
| A3 | engram + CE | 10.0% | 0.0% | 100.0% | 35368.54 | +5.4% | 4,210,817 |
| A4 | engram + contrast | 0.0% | 0.0% | 60.0% | 31075.81 | -7.4% | 4,210,817 |

Pass bars: A0 recall <= 5% (gate); best engram >= 80% verbatim, >= 60% QA; drift <= 2%; interference drop <= 10%.

## Discovery metrics (no bar)

- **Confabulation** vs in-context: engram 100.0% vs A1 80.0% on 200 never-trained distractors.
- **Shape principle** (engram vs dense ghost): verbatim 10.0% vs 0.0%; QA 0.0% vs 0.0%.
- **Interference**: batch-1 recall 4.0% -> 0.0% after batch-2 write (drop +4.0%).
- **Slot firing**: 166 unique top-1 slots over 2,970 tokens (frac unique 0.056).

_Cost: report-only_

All raw numbers: `results.json`
