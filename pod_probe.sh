#!/usr/bin/env bash
# pod_probe.sh - 2-cent diagnostic: prove dockerArgs executes, env injects,
# and git push-back works, on the cheapest GPU. Pushes evidence, then dies.
set -u
BRANCH=contrastive-sweep
REPO=github.com/Rishabh117117/ghost
WORK=/w

cd "$WORK" || exit 1
git config user.email "probe@runpod.local"
git config user.name "probe pod"
git remote set-url origin "https://x-access-token:${GIT_PUSH_TOKEN:-}@${REPO}.git"
mkdir -p status

{
  echo "$(date -u +%FT%TZ) PROBE ALIVE on pod ${RUNPOD_POD_ID:-unknown}"
  for v in RUNPOD_API_KEY HF_TOKEN GIT_PUSH_TOKEN RUNPOD_POD_ID; do
    eval "val=\${$v:-}"
    echo "  env $v: $([ -n "$val" ] && echo present || echo MISSING)"
  done
  echo "  gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>&1 | head -1)"
  echo "  python: $(python --version 2>&1)"
} >> status/probe.log

git add -A && git commit -m "probe: dockerArgs executed, env presence logged" \
  && for i in 1 2 3 4; do
       git pull --rebase origin "$BRANCH" >/dev/null 2>&1
       git push origin "$BRANCH" >/dev/null 2>&1 && break
       sleep $((2 ** i))
     done

sleep 180   # stay alive so the sandbox can read live telemetry via the API

curl -s -X POST "https://api.runpod.io/graphql" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${RUNPOD_API_KEY:-}" \
  -d "{\"query\":\"mutation { podTerminate(input: {podId: \\\"${RUNPOD_POD_ID:-}\\\"}) }\"}" \
  >/dev/null || true
