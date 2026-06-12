# v6 sink analysis: why norm-ranked selection exploded (+725% ppl)

Derived from `results/engram_v6/kept_maps.json` (engram-v6 branch; 5 entities
x 36 layers x 8 kv-heads = 288 layer-heads per entity, k=16 unless noted).
Every number below is re-derived from the artifact, not transcribed.

## Position-0 retention (the attention sink)

| ranker | layer-heads keeping pos-0 (avg over 5 entities) |
|---|---|
| mass, k=8  | 272.8 / 288 |
| mass, k=16 | 277.2 / 288 |
| mass, k=32 | 281.2 / 288 |
| norm, k=16 | **123.8 / 288** |

## Top kept positions (count over 5 entities x 288 layer-heads, k=16)

- **mass**: 0 (277), 65 (124), 67 (114), 63 (112), 58 (110), 62 (109), 51 (106)
  -> the sink + late-paragraph content positions.
- **norm**: 0 (123), 33 (86), 39 (85), 51 (84), 8 (83), 7 (81), 46 (81)
  -> sink kept barely 43% of the time; the rest scattered by key magnitude.

## Mechanism (one line)

Attention sinks (StreamingLLM, Xiao et al. 2023): position 0 absorbs default
attention mass; evict it and that mass redistributes onto arbitrary kept
positions, corrupting every read - S16-norm kept pos-0 in only 124/288
layer-heads and went +725% ppl when loaded, while mass-ranking re-discovers
the sink automatically (272-281/288) because the sink IS where mass goes.

## The second v6 mechanism: global top-k starves attributes

Per-attribute verbatim from the v6 grid (mass ranker):

| k | birth_year | city | employer | profession | quirk |
|---|---|---|---|---|---|
| 8  | 1% | 2% | 2% | **100%** | 1% |
| 16 | 17% | 33% | 44% | **100%** | 9% |
| 32 | 79% | 92% | 98% | **100%** | 27% |

Global top-k allocates by total calibration mass, which concentrates on a
few high-traffic sentences; whole attributes (quirk above all) are starved
even at k=32 while profession saturates. This motivates v6.5's
floor-then-greedy allocator: {sink} + per-fact-span floor + greedy fill.
