# Stage 2 — bank + router (first increment)

A 2-entry **bank** of ghosts + a **no-training embedding-centroid top-1 router**, and a
falsifiable test of both halves: (a) does the router pick the right ghost, and (b) does
picking right actually matter? Architecture is unchanged — `GhostModel`/`GhostStream`/
compressor are imported from `ghost.py`.

## What was built (`bank.py`)
- **Bank** over the frozen `Qwen/Qwen3-4B`:
  - **Ghost A = voice** (`ghost_voice_03_chat.pt`) — Stage-1 chat-trained on the voice corpus. STYLE signal.
  - **Ghost B = tool** (`ghost_b_01.pt`) — trained on tool-call traces via `ghost.py`'s raw
    pipeline, same knobs (D_GHOST 224, lr 1e-4, wd 0.01, patience 3, max_len 256), finalized
    at val loss ≈1.50. TOPIC signal.
- **Router** (no training): embeds an input as the base model's own **mean-pooled final
  hidden state**, picks top-1 by cosine to each ghost's **domain centroid** (centroid = mean
  base embedding over that ghost's *train* corpus). Routes in the same space the ghosts read from.
- **Serving**: route → load selected ghost into the frozen base → run.

## Method
- Same seeded 85/15 split (`SEED=0`) as Stage 1, so raw/chat/Stage-2 use identical held-out turns.
- Corpora (gitignored — personal data, never committed): voice 814 turns (692/122),
  tool 1336 turns (1136/200).
- Routing probe: 50 held-out voice + 50 held-out tool inputs (not in any centroid/train). Chance = 50%.
- Four-way ppl: **chat-masked** (assistant tokens only, same masking as the Stage-1 chat test)
  on each domain's held-out set, under base / wrong ghost / routed / oracle.
- Implementation note: Ghost B used a **resumable, per-epoch-checkpoint** trainer (same objective
  and knobs as `ghost.train`) because this laptop's Modern Standby kept killing long GPU runs;
  a `keep-awake` power assertion was also added to the run.

## Results

**Routing accuracy** (chance = 50%):

| class | routed correctly |
|---|---|
| tool inputs → B | **88.0%** (44/50) |
| voice inputs → A | **96.0%** (48/50) |

Worked example (one): a held-out tool trace → cosine(A_voice)=0.654, cosine(B_tool)=0.957 → picked **B_tool** (correct).

**Four-way perplexity** (chat-masked, assistant tokens only) — *corrected*:

| domain | base | wrong ghost | routed | oracle |
|---|---|---|---|---|
| voice | 473.58 | 402.91 | 148.69 | 148.07 |
| tool  | 59.51 | 43.49 | 17.54 | 14.61 |

- **PROBE 2 (base frozen):** fingerprint delta after loading A and after loading B = `0.000000e+00`.
- **PROBE 3 (tiny ghost):** ghost/base = **0.913%**.

> **Correction:** the first version of this table (commit `d123da3`) had a variable swap in
> `bank.py`'s four-way `wrong`/`routed` columns (voice-domain `wrong` used tool-examples-under-A,
> tool-domain `wrong` used voice-examples-under-B; `base`/`oracle` and the routing-accuracy numbers
> were unaffected). The generic-vs-specific diagnostic caught it. Numbers above are the corrected run.

## Verdict: **PASS**

- **Router discriminates** well above chance both ways (tool→B 88%, voice→A 96%).
- **Routing matters in BOTH domains** — oracle ≈ routed ≪ wrong ≤ base:
  - voice: oracle 148.07 ≈ routed 148.69 ≪ wrong (tool ghost) 402.91 < base 473.58.
  - tool:  oracle 14.61 ≈ routed 17.54 ≪ wrong (voice ghost) 43.49 < base 59.51.
  Routing to the domain-matched ghost is clearly best in each domain; the earlier
  "wrong-beats-oracle on voice" anomaly was the bug, not a real effect.
- Note: on tool, the *wrong* (voice) ghost still beats base (43.49 < 59.51) — because the voice
  ghost partially generalizes (see `GENERIC_TEST.md`: +27% on tool, +22% on unrelated wikitext).
  Oracle still ≪ wrong, so routing matters; but the voice ghost is not a pure specialist.

## What's next
The bank+router mechanism works and routing-to-domain helps in both domains. The honest gap
(from the generic-vs-specific test) is **specialization parity**: Ghost B (tool) is a clean
specialist, Ghost A (voice) a partial generalist. Next: parity-train A toward B's budget, add
more topical entries to test routing among specialists, and a **style-aware** signal for voice.
