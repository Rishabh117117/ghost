# engram-v6: selection compression + identity gate

**ENGRAM-V6: no working point — best S32-mass 79.8% < 80%**

_Generated 2026-06-12T20:39:04.096187+00:00 | meta: {"smoke": false, "model": "Qwen/Qwen3-4B", "device": "cuda", "seed": 0, "n_layers": 36, "s_ks": [32, 16, 8], "offset": 105, "mean_ctx_tokens": 74.0, "calibration": "train-template sentences only (no held-out QA)", "addressing": "blake2b hash k=4096 -> dense rank storage"}_

Part 0: ENGRAM_SERIES.md cross-checked against per-branch artifacts and committed to main (fixes: v1 A1 was 92.8% pre-repair-scorer, not in the 94.0-94.2% range; one typo).

## The curve (recall vs MB/entity)

| arm | MB/entity | verbatim | QA | theft (ungated) | drift_loaded | mass recapture |
|---|---|---|---|---|---|---|
| E0 | 10.91 | 93.5% | 82.0% | 62.5% | -2.1% | - |
| S32-mass | 4.72 | 79.8% | 79.2% | 60.0% | -4.8% | 92.6% |
| S16-mass | 2.36 | 40.0% | 40.8% | 28.0% | -6.7% | 85.0% |
| S8-mass | 1.18 | 19.8% | 19.8% | 17.5% | -8.8% | 78.5% |
| S16-norm | 2.36 | 2.8% | 16.0% | 8.5% | +725.1% | 25.0% |
| S8-tune | 1.18 | 19.0% | 20.0% | 17.0% | -8.7% | - |

A1 (token channel): 94.2% / 82.5%. A0 gate 0.2%.
Bars: A0 <= 5%; E0 >= 90%; working point = smallest k with verbatim >= 80%.

## Mass vs norm ranking (k=16)

- mass-ranked: 40.0% verbatim / 40.8% QA
- norm-ranked: 2.8% verbatim / 16.0% QA
- gap +37.2%: calibration earns its cost.

## Identity gate (entity-id match - upper bound)

| quantity | ungated (E0) | gated |
|---|---|---|
| theft on distractors | 62.5% | 0.5% |
| hedge when right memory absent | - | 100.0% (system-level, by construction) |
| model's own hedge, no injection | 0.0% | - |

Under id-gating, blocking and absence-detection are deterministic, so the gated numbers hold by construction; the ungated-vs-gated contrast prices what surface-form detection must deliver. The model itself never hedges - the 'I don't know' must come from the gate, not the LM.

- S-tune rule: S8-mass 19.8% < 70% -> tuned kept VALUES only (keys frozen), 300 steps.

_Cost: wall 64 min_

All raw numbers: `results.json`; kept-position maps: `kept_maps.json`; curve: `curve.png`
