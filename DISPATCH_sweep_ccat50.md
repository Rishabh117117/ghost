# Build task: ghost — contrastive sweep on CCAT50 (λ × negative-type factorial)

## Why this exists
CONTRASTIVE_CCAT50.md (commit 58feaa8): leak REPLICATED at 0.471; the λ=1.0
self-paraphrase hinge cure FAILED — leak 0.332, missed both bars (≤0.5× plain, ≤0.10 abs)
while retaining 86% own-win. Open question is two-dimensional: is the miss a MAGNITUDE
problem (λ too small) or a NEGATIVES-DIVERSITY problem (self-paraphrases too narrow)?
This sweep is the factorial that disentangles them. It is also the Stage-2 gate: no arm
passes → CE+hinge is structurally insufficient at this scale; do NOT extend the grid.

## Environment & roles
- You are running in a **CPU-only cloud sandbox**. You author code, commit, and
  orchestrate. ALL training/eval runs on a RunPod GPU pod you create via the RunPod API.
- Secrets expected in env: `RUNPOD_API_KEY`, `HF_TOKEN`, git push auth for
  github.com/Rishabh117117/ghost. **Verify all three exist before writing any code;
  stop and report if missing.**
- Branch: create `contrastive-sweep` off `origin/contrastive-ccat50`. Everything —
  code, configs, status files, results — flows through this branch. Nothing lives
  only on the pod.

## Phase 0 — sweep code (sandbox, no GPU)
Extend the committed `contrastive_ccat50.py` into `sweep_ccat50.py`. Reuse its
machinery byte-for-byte where possible: CCAT50 download, chat masking, 220-token
pre-truncation, `agg_ppl` eval, v03 training recipe (D_GHOST=224, lr 1e-4, wd 0.01,
patience 3, seeded 85/15 split). Architecture untouched.

**Arms (13, sequential, one Qwen3-4B load for the whole run):**
- Arm 0: plain ghost (λ=0) — fresh anchor on this hardware, replicates the disease.
- Arms 1–12: λ ∈ {0.5, 1, 2, 4} × negatives ∈ {self, author, both}.

**Negative sets (target = AaronPressman, same as before):**
- `self`: neutral self-paraphrases, exact existing recipe (regenerate on pod;
  data gitignored).
- `author`: 50 docs sampled from the 10 contrast authors' **C50train** splits
  (5/author, seed=42). No paraphrase generation needed. CE_base precomputed once.
- `both`: hinge = mean of the two hinge terms (λ applied once, not doubled).

**Eval grid (rows = base, style-prompt, all 13 arms):**
- col A: target C50test (own-win)
- col B: seen-contrast — C50test of the 10 authors whose C50train fed negatives
- col C: **unseen-contrast — C50test of the NEXT 10 authors alphabetically**
  (never touched in training; this is the generalization check on the cure)
- col D: wikitext-2 (same filter as generic_test.py)
Leak fraction reported separately for B and C. Identical token spans across all rows.

## Phase 1 — local plumbing proof (sandbox, CPU)
Before any pod spend: dry-run `sweep_ccat50.py --smoke` with a tiny random
`LlamaConfig` model (the repo's established verification trick), 2 arms × 2 steps,
fake data. Must produce a syntactically complete grid + report. Commit.

## Phase 2 — pod orchestration
- Create one on-demand pod via RunPod API: **A100 80GB preferred, H100 if cheaper or
  A100 unavailable**, PyTorch CUDA template, ≥60 GB disk, per-second billing.
- Pod startup (single script, committed as `pod_run.sh`):
  clone branch → pip install → run sweep → push artifacts → **self-terminate via
  RunPod API** (key passed as pod env var). Termination must also fire on fatal error
  (trap), after committing partial results.
- Run discipline (hard rules, all in-script):
  - heartbeat: append timestamp + arm + step to `status/heartbeat.log` every 60 s;
    push `status/` to the branch every 15 min so the sandbox can monitor via git only.
  - checkpoints: keep last 1 per arm locally; on each arm's completion,
    `push_to_hub` to private HF repo `Rishabh117117/ghost-ckpts` under
    `sweep-ccat50/arm_{i}_{lam}_{negtype}`. No checkpoint may have a single
    physical home.
  - resumable: arms checkpoint independently; rerun skips completed arms
    (presence of arm result JSON = done).
- Expected cost/time: ~10 min/arm on A100 → ~2.5 h ≈ $2–3. If projection at arm 3
  exceeds 3× this, commit partials and terminate (report why).

## Phase 3 — verdict
Per-arm pass bars (all three required):
1. seen-leak (col B fraction) ≤ 0.10 absolute
2. seen-leak ≤ 0.5 × Arm-0 plain leak
3. own-win retention ≥ 70% of Arm 0's own-win
Report unseen-leak (col C) alongside — a B-pass with C-fail means the cure
memorizes authors instead of removing genericness; flag explicitly, counts as FAIL.

Write `SWEEP_CCAT50.md`: verdict table first (all arms × all cols + leak fractions),
then factorial analysis — main effect of λ, main effect of negative type,
interaction; retention-vs-leak tradeoff curve (CSV + PNG committed). One-line
verdict at top: **CURE PASS (arm …)** or **CURE FAIL — structural**.
Commit `results.json` with every number. Push branch.

## Out of scope (do not do)
- Voice-corpus tests (corpus pending recovery), architecture changes, λ outside
  the grid, objective classes beyond CE+hinge, touching main or other branches.
- If no arm passes, the recorded conclusion is the exit condition above —
  the next move is a design decision (domain+register reframe vs DPO-style
  pairs), made outside this task.

## Report back
Paste: verdict line, the grid, λ/negative-type main effects, total cost, and any
deviations from this spec with reasons.
