# engram-v6.5: allocation, the knee, and composition

**ENGRAM-V6.5: no working point — best G48 89.2% (attr floor 44.0%)**

_Generated 2026-06-13T01:38:46.143200+00:00 | meta: {"smoke": false, "model": "Qwen/Qwen3-4B", "device": "cuda", "seed": 0, "seg": 105, "g_ks": [24, 40, 48], "f_ks": [16, 24, 32], "k_compose": 32, "n_comp_rows": 200, "inference_only": true}_

Part 0: ANALYSIS_V6_SINKS.md committed to main (pos-0 kept 273-281/288 by mass vs 124/288 by norm; starvation table) - every number re-derived from v6 kept_maps.json.

## Both curves (equal budget at equal k)

| arm | MB/entity | verbatim | QA | attr floor | theft | drift_loaded | recapture |
|---|---|---|---|---|---|---|---|
| A1 (carried) | - | 94.2% | 82.5% | - | - | - | - |
| E0 | 10.91 | 93.5% | 82.0% | 65.3% | 62.5% | -2.1% | - |
| G32 (carried) | 4.72 | 79.8% | 79.2% | 26.7% | 60.0% | -4.8% | 92.6% |
| G24 | 3.54 | 63.2% | 67.8% | 21.3% | 50.5% | -5.9% | 89.3% |
| G40 | 5.90 | 87.2% | 82.0% | 38.7% | 62.5% | -3.7% | 95.0% |
| G48 | 7.08 | 89.2% | 82.5% | 44.0% | 63.5% | -3.2% | 96.9% |
| F16 | 2.36 | 42.5% | 43.0% | 9.3% | 31.5% | -6.6% | 84.8% |
| F24 | 3.54 | 63.5% | 68.0% | 22.7% | 50.5% | -5.9% | 89.3% |
| F32 | 4.72 | 80.2% | 80.0% | 26.7% | 60.0% | -4.7% | 92.5% |

Working point = smallest config with verbatim >= 80% AND every attribute >= 70%. A0 0.2%.

## Per-attribute verbatim (the starvation check)

| arm | birth_year | city | employer | profession | quirk |
|---|---|---|---|---|---|
| G32 | 79% | 92% | 98% | 100% | 27% |
| G24 | 43% | 70% | 81% | 100% | 21% |
| G40 | 95% | 99% | 100% | 100% | 39% |
| G48 | 100% | 99% | 100% | 100% | 44% |
| F16 | 13% | 24% | 69% | 100% | 9% |
| F24 | 43% | 70% | 81% | 100% | 23% |
| F32 | 80% | 92% | 99% | 100% | 27% |

## Budget-waste audit (cost of the floor guarantee)

| arm | floor/lh | duplicates greedy | forced picks |
|---|---|---|---|
| F16 | 10.05 | 8.9 | 1.15 |
| F24 | 10.05 | 9.76 | 0.29 |
| F32 | 10.05 | 9.99 | 0.07 |

## Composition (S32 segments, sequential RoPE slots)

| load | recall | intrusion | stranger theft | stranger hedge | gated theft | gated model hedge |
|---|---|---|---|---|---|---|
| C1 | 83.0% | 0.0% | 60.0% | 0.0% | 0.0% | 0.0% |
| C2 | 40.0% | 41.5% | 51.0% | 0.0% | 0.0% | 0.0% |
| C4 | 22.0% | 53.0% | 42.0% | 0.0% | 1.0% | 0.0% |

- C4/C1 = 0.27 (bar >= 0.9) -> FAIL. Gated Cn collapses to C1 + abstention on strangers by construction (id-gate).

_Cost: wall 94 min_

All raw numbers: `results.json`
