# engram-v1: sparse parametric fact memory

**ENGRAM FAIL — verbatim, paraphrased-QA, wiki-drift** (best A3)

_Generated 2026-06-11T22:16:43.993906+00:00 | meta: {"smoke": false, "model": "Qwen/Qwen3-4B", "device": "cuda", "seed": 0, "K": 32768, "d_key": 128, "topk": 4, "tap_layer": 18}_

## Grid (recall = fraction correct; drift vs base wikitext ppl)

| arm | what | verbatim | paraphrased-QA | confab | wiki ppl | drift | trainable params |
|---|---|---|---|---|---|---|---|
| A0 | base (gate) | 0.2% | 0.0% | 99.5% | 27.03 | +0.0% | 0 |
| A1 | in-context (RAG upper bound) | 92.8% | 15.8% | 99.5% | 27.03 | +0.0% | 0 |
| A2 | dense ghost D=224 | 2.0% | 0.4% | 98.5% | 56.01 | +107.3% | 36,714,240 |
| A3 | engram + CE | 2.6% | 0.4% | 96.0% | 395.21 | +1362.3% | 84,216,321 |
| A4 | engram + contrast | 1.0% | 0.0% | 98.0% | 430.63 | +1493.4% | 84,216,321 |

Pass bars: A0 recall <= 5% (gate); best engram >= 80% verbatim, >= 60% QA; drift <= 2%; interference drop <= 10%.

## Discovery metrics (no bar)

- **Confabulation** vs in-context: engram 96.0% vs A1 99.5% on 200 never-trained distractors.
- **Shape principle** (engram vs dense ghost): verbatim 2.6% vs 2.0%; QA 0.4% vs 0.4%.
- **Interference**: batch-1 recall 4.9% -> 3.2% after batch-2 write (drop +1.6%).
- **Slot firing**: 224 unique top-1 slots over 60,108 tokens (frac unique 0.004).

_Cost: wall 87 min_

All raw numbers: `results.json`
