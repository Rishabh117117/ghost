# engram-v3: hash-addressed fact memory (no learned router)

**ENGRAM-V3 FAIL — verbatim, QA**

_Generated 2026-06-12T03:23:59.933944+00:00 | meta: {"smoke": false, "model": "Qwen/Qwen3-4B", "device": "cuda", "seed": 0, "K": 32768, "n_null": 64, "tap_layer": 18, "addressing": "blake2b+linear-probe", "c1": "entity-slot", "c2": "fact-slot (deviation, see docstring)", "replay": 2000, "kl_w": 1.0, "warmup_frac": 0.3, "replay_share": 0.25}_

## v2 autopsy (closes v2 on evidence)

B1 dominant slot 20916: |values[20916]| = **13.59653854** vs mean |values| over all slots = 0.62423891 (max 16.58694267, nonzero>1e-3: 16613/32768). **NOT ~0** (22x the mean): the dominant slot was HEAVILY written, yet recall was 0. The v2 story is revised from 'nothing was written' (cold-start) to 'the write happened but was unreadable' — 16-of-20 entities superimposed on one shared slot. Per-entity deterministic addressing (this run) is still the correct isolation.

## Grid

| arm | what | verbatim | QA | confab | wiki ppl | drift | mean/max |val| (assigned) | params |
|---|---|---|---|---|---|---|---|---|
| A0 | base (gate) | 0.2% | 0.2% | 100.0% | 27.03 | +0.0% | - | 0 |
| A1 | in-context (RAG upper bound) | 94.0% | 82.0% | 100.0% | 27.03 | +0.0% | - | 0 |
| C1 | hash-addressed, one slot per ENTITY, tap 18 | 0.0% | 1.0% | 100.0% | 27.02 | -0.0% | 6.039 / 7.965 | 83,886,081 |
| C2 | hash-addressed, one slot per FACT (entity x attr), tap 18 | 1.5% | 2.2% | 100.0% | 27.02 | -0.0% | 4.266 / 8.755 | 83,886,081 |
| C3 | C1 at tap layer 9 (depth probe) | 0.0% | 0.8% | 100.0% | 27.02 | -0.0% | 4.639 / 6.214 | 83,886,081 |

Bars: A0 <= 5% | C1 verbatim >= 80%, QA >= 60%, drift <= 2% | interference drop <= 10%.

## Readings

- **did the bank fill?** C1 assigned-slot |values|: mean 6.039368, max 7.964925, nonzero slots 500 (trajectory: results/.../norm_traj_C1.json + value_norms.png).
- **capacity (C2 per-fact slots vs C1 shared entity slot)**: verbatim 1.5% vs 0.0%, QA 2.2% vs 1.0% — a private value vector per fact does not help (NOTE: dispatch's S=4-mean variant was dropped — it is provably a numerical replicate of S=1 under Adam with zero init; see module docstring).
- **depth (C3 tap 9 vs C1 tap 18)**: verbatim 0.0% vs 0.0%.
- **confabulation vs A1**: C1 100.0% vs A1 100.0% on never-trained distractors (distractors read null slots: the bank is silent off-entity by construction).
- **scorer sanity**: A1 QA / verbatim = 0.87 (gate >= 0.80).

_Cost: wall 34 min_

All raw numbers: `results.json`
