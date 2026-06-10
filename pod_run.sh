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
  git add -A status results results.json SWEEP_CCAT50.md 2>/dev/null || true
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

# ---- preflight: all secrets present, tokens actually work --------------------
: "${RUNPOD_API_KEY:?missing}" "${HF_TOKEN:?missing}" \
  "${GIT_PUSH_TOKEN:?missing}" "${RUNPOD_POD_ID:?missing}"

if [ ! -d "$WORK/.git" ]; then
  git clone --branch "$BRANCH" \
    "https://x-access-token:${GIT_PUSH_TOKEN}@${REPO}.git" "$WORK"
fi
cd "$WORK"
git config user.email "pod@runpod.local"
git config user.name "sweep pod"
mkdir -p status results

python -m pip install -q -r requirements_pod.txt

python - <<'EOF'   # HF token must be live before any GPU time is spent
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
