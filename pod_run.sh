#!/usr/bin/env bash
# pod_run.sh - RunPod startup for engram-v6.5 (allocation, the knee, composition; inference-only).
#
# TRANSPORT: Hugging Face ONLY. The GitHub PAT in this environment cannot push
# (fine-grained token without Contents:write - proven 2026-06-11), so the pod
# publishes ALL evidence and results to the private HF repo
# Spartan117Ri/ghost-ckpts. The sandbox supervisor mirrors them into git.
#
# Layout in the HF repo:
#   crumbs/<pod>_<stage>.txt        breadcrumbs (curl-only, work pre-deps)
#   runs/<pod>/stages.log|*.log     stage trail + log tails
#   runs/<pod>/DONE                 exit code, written by the EXIT trap
#   runs/arms_engram_v6_5/arm_*.json  per-arm results (resume keys, per-experiment)
#   runs/final/                     results.json + ENGRAM_V6_5.md + per-arm JSONs
#   sweep-ccat50/arm_*/ghost.pt     checkpoints (uploaded by sweep itself)
#
# Stages: boot -> deps -> hf -> gpu -> smoke-gate -> sweep -> DONE.
# Every fatal path uploads its log tail first; the pod always self-terminates.
set -u -o pipefail

BRANCH=engram-v6_5
WORK=/workspace/ghost
HFREPO=Spartan117Ri/ghost-ckpts
POD="${RUNPOD_POD_ID:-unknown}"

terminate_pod() {
  echo "[pod_run] terminating pod ${POD}" >&2
  curl -s -X POST "https://api.runpod.io/graphql" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer ${RUNPOD_API_KEY:-}" \
    -d "{\"query\":\"mutation { podTerminate(input: {podId: \\\"${POD}\\\"}) }\"}" \
    >/dev/null || true
}

hf_curl_up() {  # $1 = local file, $2 = repo path; curl-only (works pre-deps)
  [ -f "$1" ] || return 0
  B64=$(base64 -w0 < "$1")
  printf '{"key":"header","value":{"summary":"pod %s: %s"}}\n{"key":"file","value":{"path":"%s","content":"%s","encoding":"base64"}}\n' \
    "$POD" "$2" "$2" "$B64" \
  | curl -s -m 30 -X POST \
      "https://huggingface.co/api/models/${HFREPO}/commit/main" \
      -H "Authorization: Bearer ${HF_TOKEN:-}" \
      -H "Content-Type: application/x-ndjson" --data-binary @- >/dev/null 2>&1 || true
}

crumb() {  # $1 = stage tag; tiny existence marker
  echo "crumb pod=$POD stage=$1 date=$(date -u +%FT%TZ)" > /tmp/crumb.txt
  hf_curl_up /tmp/crumb.txt "crumbs/${POD}_$1.txt"
}

stage() {  # $1 = stage name; append trail + publish it
  echo "$(date -u +%FT%TZ) stage: $1" >> "$WORK/status/stages.log" 2>/dev/null || true
  hf_curl_up "$WORK/status/stages.log" "runs/${POD}/stages.log"
}

publish_evidence() {  # logs + arm results; cheap enough to call often
  cd "$WORK" 2>/dev/null || return 0
  for f in status/boot.log status/gpu.txt status/heartbeat.log; do
    hf_curl_up "$f" "runs/${POD}/$(basename "$f")"
  done
  for f in status/pip.log status/smoke.log status/run.log; do
    [ -f "$f" ] && { tail -c 200000 "$f" > /tmp/tail.txt; hf_curl_up /tmp/tail.txt "runs/${POD}/$(basename "$f")"; }
  done
  # per-arm results are the resume keys - publish to the SHARED prefix
  if ls results/engram_v6_5/arm_*.json >/dev/null 2>&1; then
    python pod_hf.py updir results/engram_v6_5 runs/arms_engram_v6_5 >/dev/null 2>&1 || \
      for f in results/engram_v6_5/arm_*.json; do hf_curl_up "$f" "runs/arms_engram_v6_5/$(basename "$f")"; done
  fi
  [ -f status/ABORT.json ] && hf_curl_up status/ABORT.json "runs/${POD}/ABORT.json"
}

on_exit() {
  code=$?
  kill "${PUSHER_PID:-0}" 2>/dev/null || true
  echo "$(date -u +%FT%TZ) exit code ${code}" >> "$WORK/status/stages.log" 2>/dev/null || true
  publish_evidence
  hf_curl_up "$WORK/status/stages.log" "runs/${POD}/stages.log"
  # final artifacts (real run writes them at repo root)
  cd "$WORK" 2>/dev/null && {
    [ -f results.json ]      && python pod_hf.py up results.json "runs/final/results.json" 2>/dev/null
    [ -f ENGRAM_V6_5.md ]      && python pod_hf.py up ENGRAM_V6_5.md "runs/final/ENGRAM_V6_5.md" 2>/dev/null
    [ -d results/engram_v6_5 ] && python pod_hf.py updir results/engram_v6_5 runs/final/engram 2>/dev/null
  }
  echo "$code" > /tmp/done.txt; hf_curl_up /tmp/done.txt "runs/${POD}/DONE"
  terminate_pod
}
trap on_exit EXIT

# ---- boot (clone happened in dockerArgs; make sure, idempotently) -------------
if [ ! -d "$WORK/.git" ]; then
  git clone --branch "$BRANCH" "https://github.com/Rishabh117117/ghost.git" "$WORK"
fi
cd "$WORK"
mkdir -p status results

MISSING=""
for v in RUNPOD_API_KEY HF_TOKEN RUNPOD_POD_ID; do
  eval "val=\${$v:-}"
  [ -n "$val" ] || MISSING="$MISSING $v"
done
echo "$(date -u +%FT%TZ) pod ${POD} boot; missing env:${MISSING:-' none'}" >> status/boot.log
crumb "podrun-start"
stage "boot"
hf_curl_up status/boot.log "runs/${POD}/boot.log"
if [ -n "$MISSING" ]; then
  echo "[pod_run] FATAL: missing env:${MISSING}" >&2
  exit 9
fi

# ---- deps ---------------------------------------------------------------------
python -m pip install -r requirements_pod.txt > status/pip.log 2>&1 \
  || { tail -5 status/pip.log >&2; stage "deps-FAILED"; exit 10; }
stage "deps"

# ---- HF preflight + arm-resume restore -----------------------------------------
python - <<'EOF' \
  || { echo "[pod_run] FATAL: HF preflight failed" >&2; stage "hf-FAILED"; exit 11; }
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
print("HF auth ok:", api.whoami()["name"], flush=True)
api.create_repo("Spartan117Ri/ghost-ckpts", private=True, exist_ok=True)
EOF
mkdir -p results/engram_v6_5
rm -f results/engram_v6_5/arm_*.json results/engram_v6_5/interference.json  # only HF may seed resume (v2 lesson)
python pod_hf.py down runs/arms_engram_v6_5 results/engram_v6_5 || true   # completed arms skip

# ---- data: regenerate the synthetic biographies (deterministic, gitignored) --
python engram_data.py > status/data.log 2>&1 \
  && python engram_data.py --check >> status/data.log 2>&1 \
  || { tail -15 status/data.log >&2; stage "data-FAILED"; exit 14; }
stage "data" 
stage "hf"

# ---- GPU check ------------------------------------------------------------------
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee status/gpu.txt
python - <<'EOF' \
  || { echo "[pod_run] FATAL: GPU check failed" >&2; stage "gpu-FAILED"; exit 12; }
import torch
assert torch.cuda.is_available(), "torch sees no CUDA"
x = torch.randn(64, 64, dtype=torch.bfloat16, device="cuda")
(x @ x).float().sum().item()
print("CUDA ok:", torch.cuda.get_device_name(0), "| torch", torch.__version__, flush=True)
EOF
stage "gpu"

# ---- background evidence pusher (every 10 min) ----------------------------------
(
  while true; do
    sleep 600
    publish_evidence
  done
) &
PUSHER_PID=$!

# ---- SMOKE GATE on this GPU ------------------------------------------------------
python engram_v6_5.py --smoke > status/smoke.log 2>&1 \
  || { tail -15 status/smoke.log >&2; stage "smoke-gate-FAILED"; exit 13; }
stage "smoke-gate-GREEN"

# ---- the real engram-v3 run ----------------------------------------------------------
stage "engram-start"
python engram_v6_5.py 2>&1 | tee -a status/run.log
exit "${PIPESTATUS[0]}"   # EXIT trap publishes artifacts + DONE + terminates
