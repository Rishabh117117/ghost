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

**Four-way perplexity** (chat-masked, assistant tokens only):

| domain | base | wrong ghost | routed | oracle |
|---|---|---|---|---|
| voice | 473.58 | 43.49 | 122.80 | 148.07 |
| tool  | 59.51 | 402.91 | 16.48 | 14.61 |

- **PROBE 2 (base frozen):** fingerprint delta after loading A and after loading B = `0.000000e+00`.
- **PROBE 3 (tiny ghost):** ghost/base = **0.913%**.

## Verdict: **PASS (mechanism)** — with one real caveat

- **Router discriminates** well above chance both ways (88% / 96%).
- **Routing matters — decisively on the tool domain:** routed (16.48) ≈ oracle (14.61) ≪ base
  (59.51), and the *wrong* (voice) ghost is catastrophic on tool data (402.91). Picking right is essential.
- **Caveat — voice domain inverts:** the *wrong* tool ghost (43.49) actually **beats** the
  domain-matched voice ghost (148.07). So on voice, routing-to-domain is *not* optimal. Likely
  cause: Ghost B trained on a larger/more diverse corpus (1136 turns, raw full-token, ~13 epochs)
  → a stronger *general* corrector, while Ghost A (chat-masked, ~700 turns, early-stopped epoch 7)
  is a weaker specialist. Here **ghost strength, not domain-match, dominates** on the voice side.
  This is the expected "are the ghosts specialized enough?" granularity finding — surfaced honestly,
  not hidden.

## What's next
The bank+router mechanism works; the honest gap is **specialization/strength parity** — bring
Ghost A's training budget/objective up to Ghost B's (or add more, better-matched topical entries)
before trusting "route-to-domain = best", and a **style-aware** signal (not a semantic centroid)
remains the eventual need for voice.
