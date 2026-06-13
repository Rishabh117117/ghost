#!/usr/bin/env bash
# pod_boot.sh - first thing a pod runs, fetched raw from GitHub by dockerArgs:
#   bash -c curl${IFS}-sSf${IFS}<raw-url>/pod_boot.sh|bash
# The dockerArgs payload is a single token (no spaces/quotes) so it survives
# any server-side splitting of the start command. This script does the real
# shell work: clone the branch and hand off to the pipeline (or the probe).
set -u
BRANCH=engram-v6_5
WORK=/workspace/ghost

if [ ! -d "$WORK/.git" ]; then
  git clone --branch "$BRANCH" "https://github.com/Rishabh117117/ghost.git" "$WORK"
fi

# PROBE=1 in pod env -> run the cheap diagnostic instead of the sweep
if [ "${PROBE:-0}" = "1" ]; then
  mkdir -p /w && cp -r "$WORK/." /w/ 2>/dev/null || true
  exec bash /w/pod_probe.sh
fi
exec bash "$WORK/pod_run.sh"
