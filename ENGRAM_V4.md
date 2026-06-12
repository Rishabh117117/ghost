# engram-v4: the position experiment (targeted-additive vs virtual-KV)

**ENGRAM-V4 FAIL — best arm D1: verbatim, QA**

_Generated 2026-06-12T05:09:01.498591+00:00 | meta: {"smoke": false, "model": "Qwen/Qwen3-4B", "device": "cuda", "seed": 0, "tap_layer": 18, "m_pairs": 8, "k_kv_slots": 4096, "kl_w_d1": 0.5, "replay_share_d1": 0.25, "addressing": "blake2b+linear-probe (entity), id->slot at eval"}_

## Grid

| arm | what | verbatim | QA | confab | theft | drift_clean | drift_loaded | norms | params |
|---|---|---|---|---|---|---|---|---|---|
| A0 | base (gate) | 0.2% | 0.2% | 100.0% | - | +0.0% | +- | - | 0 |
| A1 | in-context (RAG upper bound) | 94.0% | 82.0% | 100.0% | - | +0.0% | +- | - | 0 |
| D1 | additive at entity token positions, tap 18 | 0.2% | 0.0% | 100.0% | 0.5% | -0.0% | -0.9% | v 6.34/9.10 | 83,886,081 |
| D2 | virtual KV m=8, tap 18, warm-start | 0.2% | 0.0% | 100.0% | 0.5% | -0.0% | -0.5% | k 17.51 v 3.39 | 67,108,864 |

Bars (per arm): verbatim >= 80%, QA >= 60%, drift_clean <= 2%, interference drop <= 10% (if run). A0 gate <= 5%.

## Channel vs position (the headline)

- D1 (position-targeted additive): verbatim 0.2%, QA 0.0%.
- D2 (attention-channel virtual KV): verbatim 0.2%, QA 0.0%.
- Reading: NEITHER channel recovers recall - position alone and the attention channel alone are both insufficient at this layer/budget.

## False-memory probe (distractor + random real entity's memory)

| arm | theft rate | hedge rate (loaded) | confab (unloaded) |
|---|---|---|---|
| D1 | 0.5% | 0.0% | 100.0% |
| D2 | 0.5% | 0.0% | 100.0% |

## Per-attribute verbatim recall

| arm | birth_year | city | employer | profession | quirk |
|---|---|---|---|---|---|
| D1 | 1% | 0% | 0% | 0% | 0% |
| D2 | 1% | 0% | 0% | 0% | 0% |

- scorer sanity: A1 QA/verbatim = 0.87 (gate >= 0.80).
- D3 rule: skipped: neither D1 nor D2 reached 60% verbatim.

_Cost: wall 55 min_

All raw numbers: `results.json`

## Post-run validation (sandbox)

The byte-identical D1/D2 transcripts triggered an eval-path audit: with a
norm-50 value planted, D1's set_for/hook path provably alters generation when
attention layers exist downstream of the tap (the initial CPU repro failed
only because a 4-layer tiny model clamps tap 18 to the LAST layer, where a
name-position write cannot reach the final position - the pod taps 18/36).
D2's path alters generation directly at the tap layer. Injection was live at
eval (drift_loaded is nonzero for both arms); identical transcripts mean
neither memory flipped a single greedy token on those 20 prompts. The
double-null stands. Note for future smokes: a last-layer tap structurally
hides D1-class effects; use an early tap in tiny models.

Also: D1 train CE plateaued at ~7.2 vs v3's broadcast-write 4.6 - v3's CE
drop came from writing into ALL positions (including answer positions), a
shortcut not a memory. Constraining the write to name positions removes the
shortcut and the frozen attention never routes it to the answer.
