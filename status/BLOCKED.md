# Status: Phase 0–1 COMPLETE, Phase 2 BLOCKED (sandbox network policy)

2026-06-10 — contrastive-sweep branch

## Done
- `sweep_ccat50.py`: 13-arm lambda x negative-type factorial, per spec
  (DISPATCH_sweep_ccat50.md). Reuses contrastive_ccat50.py machinery;
  architecture untouched. Resumable per arm; heartbeat to status/; cost guard;
  per-arm HF checkpoint push; report generator (grid, factorial main effects,
  retention-vs-leak CSV+PNG, results.json).
- `--smoke` proof on CPU (tiny random LlamaConfig base, scratch tokenizer,
  fake data, arms 0+12, 2 epochs): GREEN — complete grid + report under
  results/smoke/. Rerun correctly skips completed arms (resume proof).
- `pod_run.sh` (clone -> install -> sweep -> push -> self-terminate, EXIT trap
  commits partials first; 15-min status pusher; HF preflight before GPU time)
  and `runpod_launch.py` (GPU pick A100-80GB-preferred/H100-if-cheaper,
  on-demand deploy, status/pod.json, status/terminate subcommands).

## Blocked: Phase 2 cannot start from this sandbox
1. **Network allowlist**: api.runpod.io, rest.runpod.io and huggingface.co all
   return 403 ("Host not in allowlist") from this environment. The RunPod and
   HF keys cannot even be *verified* here, and no pod can be created/monitored/
   terminated. Fix: add `api.runpod.io` and `huggingface.co` to the
   environment's network policy (Claude Code on the web -> environment ->
   network access), or switch the environment to full network access.
2. **GIT_PUSH_TOKEN**: the pod must push status/ and results to this branch,
   but the sandbox's git credential is a session-local proxy that does not
   extend to the pod. A GitHub token with push access to Rishabh117117/ghost
   must be supplied as `GIT_PUSH_TOKEN` for the pod env.

No secret values appear in this file, the code, or the git history.
