# Sweep: contrastive lambda x negative-type factorial on CCAT50

SMOKE RUN — plumbing only, numbers meaningless. **CURE FAIL — structural** (no arm met all bars)

_Generated 2026-06-10T23:09:56.349932+00:00 | arms completed: 2/13 | meta: {"smoke": true, "model": "tiny-random-llama", "device": "cpu", "alpha_serve": 0.5, "seed": 0}_

## Verdict table (chat-masked held-out ppl; rel% vs base)

| arm | lambda | neg | A_target | B_seen | C_unseen | D_wikitext | own-win | retention | leak B | leak C | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| base | — | — | 181.27 | 183.97 | 187.20 | 181.48 | — | — | — | — | — |
| style-prompt | — | — | 182.45 (-0.7%) | 181.71 (+1.2%) | 184.78 (+1.3%) | 185.59 (-2.3%) | — | — | — | — | — |
| 0 | 0 | none | 175.25 (+3.3%) | 183.71 (+0.1%) | 187.45 (-0.1%) | 180.56 (+0.5%) | +3.3% | 1.00 | 0.042 | -0.040 | anchor |
| 12 | 4 | both | 178.17 (+1.7%) | 184.10 (-0.1%) | 189.97 (-1.5%) | 183.67 (-1.2%) | +1.7% | 0.51 | -0.043 | -0.864 | FAIL |

Pass bars: leak B <= 0.1 abs, leak B <= 0.5 x arm-0 (0.042), retention >= 0.7. Unseen (C) mirror of bars 1-2 flags author memorization (counts as FAIL).

## Factorial analysis (means over hinged arms)

### Main effect of lambda

| level | leak B | leak C | retention |
|---|---|---|---|
| 4 | -0.043 | -0.864 | 0.51 |

### Main effect of negative type

| level | leak B | leak C | retention |
|---|---|---|---|
| both | -0.043 | -0.864 | 0.51 |

### Interaction (lambda x neg cells)

| level | leak B | leak C | retention |
|---|---|---|---|
| 4/both | -0.043 | -0.864 | 0.51 |

## Tradeoff curve

- CSV: `results/smoke/retention_vs_leak.csv`
- PNG: `results/smoke/retention_vs_leak.png`

All raw numbers: `results/smoke/results.json`
