# SESSION HANDOFF — ghost / engram (2026-06-11→12)

**Session reference (Claude Code):** `https://claude.ai/code/session_01TYWvjr2KGZiek611RJ6YWj`
Resume by referencing that session in the next one, point it at branch
`engram-v2`, and read this file first.

Repo: `Rishabh117117/ghost` (public). One session, four dispatches closed.

---

## 0. TL;DR state

| Thing | State |
|---|---|
| **main** | carries the merged CCAT50 sweep + pod infra (PR #1 squash-merged) |
| **engram-v1** | FAIL run committed; draft **PR #2** (→ main) |
| **engram-v2** | FAIL-attributed run committed; draft **PR #3** (→ engram-v1). **HEAD here.** |
| RunPod balance | **~$20.55** (user topped up +$20 mid-session) |
| Pods running | none (all self-terminated) |
| Next science | engram **addressing collapse** is the unsolved problem (see §4) |

**Hard rules carried all session:** never print/commit secret values; HF-only
transport on the pod (the PAT cannot git-push); don't iterate inside a run
beyond sanctioned ablations; flag spend before launching.

---

## 1. Environment & secrets (verify FIRST each session)

Three env-var secrets, all live this session:
- `RUNPOD_API_KEY` — RunPod GraphQL (`api.runpod.io/graphql`). Balance ~$20.55.
- `HF_TOKEN` — user **Spartan117Ri**, gated-read, **write** to private repo
  `Spartan117Ri/ghost-ckpts` (the pod's only upstream + checkpoint home).
- `GIT_PUSH_TOKEN` — GitHub PAT, **push + admin on the repo via the GitHub API**
  but **CANNOT `git push`** (fine-grained token missing Contents:write —
  proven by dry-run 403). This is why the whole pipeline uses HF transport.

Verify commands (never echo values):
- RunPod: `POST api.runpod.io/graphql {myself{clientBalance}}`
- HF: `GET huggingface.co/api/whoami-v2`
- The sandbox git proxy CAN push (supervisor mirrors results to git from here).

**Recommend the user rotate RunPod + HF keys** — both were pasted in chat in an
earlier session.

The sandbox has **no GPU**; CPU `torch` is pip-installable from pypi
(`pip install torch transformers==5.9.0 ...`) — used for CPU smokes. The
RunPod A100 is the muscle.

---

## 2. The pipeline (hard-won; reuse as-is)

**Roles:** CC sandbox = conductor; RunPod A100 = muscle; **HF repo =
transport** (pod publishes, supervisor mirrors to git).

**Files:**
- `runpod_launch.py` — `create` / `status` / `terminate` / **`supervise`**.
  `supervise` owns the whole lifecycle: launch → watch HF for boot crumb →
  watch progress → relaunch on stall/death (arms resume) → mirror final to the
  branch → terminate. Kill-switches: 4 launches, $1.50 balance floor, 5 h wall.
- `pod_boot.sh` — single-token dockerArgs bootstrap (survives start-cmd
  splitting). `pod_run.sh` — staged self-evidencing run:
  `boot→deps→data→hf→gpu→smoke-gate→<run>→DONE`, every fatal path uploads its
  log, pod always self-terminates.
- `pod_hf.py` — HF up/updir/down (resume + artifacts).
- Image: `runpod/pytorch:1.0.3-cu1281-torch280-ubuntu2204` (cached on hosts).

**HF repo layout** (`Spartan117Ri/ghost-ckpts`):
- `crumbs/<pod>_<stage>.txt` — pre-dep breadcrumbs (curl-only)
- `runs/<pod>/{stages.log,*.log,boot.log,DONE}` — per-pod evidence
- `runs/arms_engram_v2/arm_*.json` — **resume keys** (per-experiment prefix!)
- `runs/final/{results.json,ENGRAM_V2.md,engram/*}` — final, mirrored to git

**Supervisor detection (HF, not git):** `done()` = `runs/final/results.json` +
`runs/final/ENGRAM_V2.md` both present; boot = `crumbs/<pod>_podrun-start`;
stall = HF commit age. `clear_final()` runs at startup to delete a prior run's
`runs/final` so it can't trip a false 0-launch DONE.

---

## 3. THE EXPENSIVE LESSONS (do not relearn these)

1. **RunPod GraphQL 403s `Python-urllib` UA** → set `User-Agent` header.
2. **Pods died silently** because (a) `${SECRET}` inside nested dockerArgs
   quotes, (b) zero-volume pods auto-reap on container exit, (c) a `git add`
   with a missing pathspec staged nothing. Fix: anonymous clone (public repo),
   **idempotent clone** (`[ -d .git ] || clone` — a bare `clone && run`
   crash-loops forever on container restart), boot-crumb-first evidence.
3. **2024 image tag** cold-pulled 7.4 GB and stalled 3 hosts → use a current
   cached tag.
4. **PAT can't git-push** → HF transport (above). Verified, not assumed.
5. **HF namespace = HF username** (`Spartan117Ri/...`), NOT the GitHub owner —
   wrong namespace silently kills the HF preflight.
6. **CPU smoke is blind to device bugs** — `GhostModel` sets dtype but not
   device; A2 crashed on GPU only (`gm.ghost.to(DEVICE)` fixes it). Always
   `.to(DEVICE)` every sub-model.
7. **Committed arm JSONs shadow a fresh run** — v1 left `arm_A0–A4.json` on the
   branch; v2's pod reused them (stale A0/A1, scorer never re-tested). Fix:
   `pod_run.sh` wipes `results/engram/*.json` before HF resume; use a
   **per-experiment resume prefix** (`runs/arms_engram_v2`); `clear_final()`.
8. **Verify before spending** — CPU smoke green + cheap probes first. The
   T-series diagnosis (§ history) found root causes for ~$0.70 vs blind
   $0.20–0.85 pod deaths.

---

## 4. Scientific results

### Sweep (CCAT50 contrastive λ × neg-type) — on `main`
Verdict **CURE PASS arm 5 (λ=1, neg=author)**: leak −0.079, retention 0.94 —
the cleanest cure. Author-document negatives are the active ingredient;
self-paraphrase negatives fail at every λ. (Selector = min|leak_b| **banded**
by retention; the raw min|leak| would have picked arm 9 — flagged & resolved.)
`SWEEP_CCAT50.md` / `results.json` on main.

### engram-v1 — FAIL (structural). PR #2.
Sparse KV fact memory, dense write through RMSNorm+gain. Recall ~0 **and**
wiki 27→**395** drift **and** 224/32768 slot collapse. Diagnosed 5 causes.

### engram-v2 — FAIL (attributed). PR #3. **The current result.**
Five fixes, B-triangle ablation. `ENGRAM_V2.md`/`results.json` on engram-v2.

| arm | verbatim | QA | drift | uniq slots | max share |
|---|---|---|---|---|---|
| A0 gate | 0.2% | 0.2% | +0% | – | – |
| A1 RAG | 94.0% | **82.0%** | +0% | – | – |
| A2' dense+replay | 0.8% | 0.5% | +8.5% | – | – |
| **B1 full** | 0.0% | 0.2% | **+2.2%** | **7** | **68%** |
| B2 −replay | 0.2% | 0.0% | +75.4% | 104 | 7.7% |
| B3 −addressing | 0.8% | 0.5% | +9.7% | 15 | 41.4% |

**B-triangle reading:**
- ✅ **fix-2 (replay-KL) cured drift**: B1 +2.2% vs B2 +75.4% (vs v1 +1362%).
- ✅ **fix-5 (scorer) validated**: A1 QA 15.8%→**82.0%** (ratio 0.87 ≥ 0.80).
  First-line + Question/Answer framing was the fix (see `a1_transcripts.json`).
- ❌ **fix-3 (mean-centre + Switch load-balance) did NOT solve collapse**:
  B1 fired only **7** slots — fewer than B3's 15. Slot 20916 eats 68% of
  tokens / 16-of-20 entities (`b1_slot_alignment.json`). Early-abort tripped on
  B1 (trained only 500 steps). **Collapse → 0 recall is the open problem.**
- Shape principle still unsupported (engram 0% ≈ dense ghost 0.8%).
- Confab metric non-discriminative (100% everywhere incl. base) — carried caveat.

---

## 5. OPEN THREADS / next moves

1. **engram addressing collapse is THE blocker.** mean-centre + load-balance
   (0.01) + top-k 32 were insufficient; B1 collapsed harder than B3. Candidate
   next moves (one per run, no in-run iteration):
   - much stronger / differently-scaled load-balance, or entropy reg on the
     router; reconsider the early-abort (it truncated B1 to 500 steps).
   - key/query redesign: learnable keys, product-key factorization (was OOS),
     or a non-softmax-router addressing scheme.
   - decouple "store" vs "read" — the zero-init values + collapsed router means
     gradients only ever reach a few slots (rich-get-richer from step 1).
2. **Confabulation metric** needs a real design (base model never hedges).
3. **CounterFact anchor** still skipped (optional in both engram dispatches).
4. **Key rotation** (RunPod + HF) recommended now both runs are done.
5. **Commit trailer**: the harness asks commits to end with the session URL;
   I did not add it this session — add going forward if desired.

---

## 6. File map (engram-v2)
- `engram.py` — module (EngramBank/EngramModel) + arms + train (answer-mask CE,
  replay KL, Switch LB) + eval + report + smoke + resume. `--smoke`/`--report-only`.
- `engram_data.py` — deterministic synthetic biographies (seed 0; 500 ent ×5×5
  train, 2500 QA, 200 distractors). Data gitignored, regenerable.
- `engram_score.py` + `test_scorer.py` — repaired matcher + unit tests (green).
- `runpod_launch.py` / `pod_run.sh` / `pod_boot.sh` / `pod_hf.py` — pipeline.
- `ghost.py` `bank.py` `contrastive_ccat50.py` `sweep_ccat50.py` etc. — prior stages.
- `results/engram/*` — the v2 arm JSONs + transcripts + alignment.

To regenerate the report from arm JSONs (no GPU): `python engram.py --report-only`.
