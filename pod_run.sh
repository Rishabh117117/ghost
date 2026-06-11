#!/usr/bin/env bash
# pod_run.sh - RunPod startup for the CCAT50 contrastive sweep.
#
# clone branch -> pip install -> run sweep -> push artifacts -> SELF-TERMINATE.
# Termination also fires on fatal error (EXIT trap), after committing partials.
#
# Required pod env, passed via the RunPod API at pod creation (NEVER committed,
# never echoed):
#   RUNPOD_API_KEY   - for self-termination
#   HF_TOKEN         - checkpoint second home (Rishabh117117/ghost-ckpts, private)
#   GIT_PUSH_TOKEN   - GitHub token with push access to Rishabh117117/ghost
#   RUNPOD_POD_ID    - set automatically by RunPod
set -u -o pipefail

BRANCH=contrastive-sweep
REPO=github.com/Rishabh117117/ghost
WORK=/workspace/ghost

terminate_pod() {
  echo "[pod_run] terminating pod ${RUNPOD_POD_ID}" >&2
  curl -s -X POST "https://api.runpod.io/graphql" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
    -d "{\"query\":\"mutation { podTerminate(input: {podId: \\\"${RUNPOD_POD_ID}\\\"}) }\"}" \
    >/dev/null || true
}

commit_push() {  # $1 = message; tolerant: empty commits and races must not kill the trap
  cd "$WORK" 2>/dev/null || return 0
  # NB: one `git add` with a missing pathspec stages NOTHING (exit 128) - the
  # first pod died leaving zero trace partly because of that. *.pt and
  # *.safetensors are gitignored, so a blanket add is safe (no fat checkpoints).
  git add -A 2>/dev/null || true
  git commit -m "$1" >/dev/null 2>&1 || true
  for i in 1 2 3 4; do
    git push -u origin "$BRANCH" >/dev/null 2>&1 && return 0
    sleep $((2 ** i))
  done
  echo "[pod_run] WARN: git push failed after retries" >&2
}

on_exit() {
  code=$?
  kill "${PUSHER_PID:-0}" 2>/dev/null || true
  commit_push "sweep: artifacts at exit (code ${code})"
  terminate_pod
}
trap on_exit EXIT

# ---- boot: clone (public repo, no token), wire authed push, leave a marker ---
if [ ! -d "$WORK/.git" ]; then
  git clone --branch "$BRANCH" "https://${REPO}.git" "$WORK"
fi
cd "$WORK"
git config user.email "pod@runpod.local"
git config user.name "sweep pod"
# pushes need auth even though the repo is public; token never hits the branch
git remote set-url origin \
  "https://x-access-token:${GIT_PUSH_TOKEN:-}@${REPO}.git"
mkdir -p status results

# Boot marker FIRST: if anything later dies, the branch still shows the pod
# booted and which env vars existed (names only, never values).
MISSING=""
for v in RUNPOD_API_KEY HF_TOKEN GIT_PUSH_TOKEN RUNPOD_POD_ID; do
  eval "val=\${$v:-}"
  [ -n "$val" ] || MISSING="$MISSING $v"
done
echo "$(date -u +%FT%TZ) pod ${RUNPOD_POD_ID:-unknown} boot; missing env:${MISSING:-' none'}" \
  >> status/boot.log
commit_push "sweep: pod boot marker"
if [ -n "$MISSING" ]; then
  echo "[pod_run] FATAL: missing env:${MISSING}" >&2
  exit 9
fi

python -m pip install -r requirements_pod.txt > status/pip.log 2>&1 \
  || { tail -5 status/pip.log >&2; echo "[pod_run] FATAL: pip install failed" >&2; exit 10; }

python - <<'EOF' \
  || { echo "[pod_run] FATAL: HF preflight failed" >&2; exit 11; }
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
print("HF auth ok:", api.whoami()["name"], flush=True)
api.create_repo("Rishabh117117/ghost-ckpts", private=True, exist_ok=True)
EOF

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee status/gpu.txt
echo "$(date -u +%FT%TZ) pod ${RUNPOD_POD_ID} preflight ok" >> status/heartbeat.log
commit_push "sweep: pod preflight ok"

# ---- background: push status/ to the branch every 15 min ---------------------
(
  while true; do
    sleep 900
    cd "$WORK" || exit 0
    git add status results 2>/dev/null || true
    git commit -m "sweep: status heartbeat" >/dev/null 2>&1 || true
    git push -u origin "$BRANCH" >/dev/null 2>&1 || true
  done
) &
PUSHER_PID=$!

# ---- the sweep ----------------------------------------------------------------
python sweep_ccat50.py 2>&1 | tee -a status/run.log
exit "${PIPESTATUS[0]}"   # EXIT trap commits artifacts + terminates the pod
