# Status: Phase 2 step 1 CLEARED — secrets verified live, ready for step 2

2026-06-11 — new session re-ran HANDOFF.md Phase 2 step 1 in the updated
environment. The earlier allowlist block (api.runpod.io / huggingface.co 403)
is GONE here. Live checks (values never printed):

- RUNPOD_API_KEY: OK — POST api.runpod.io/graphql `myself` HTTP 200,
  clientBalance = $10.
- HF_TOKEN: OK — GET huggingface.co/api/whoami-v2 confirms canReadGatedRepos.
- GIT_PUSH_TOKEN: OK — GitHub API confirms push=True, admin=True on
  Rishabh117117/ghost.
- Pod deps reachable from sandbox: archive.ics.uci.edu (CCAT50 zip) HTTP 200;
  HF ckpt repo Rishabh117117/ghost-ckpts is 404 (not created yet — pod_run.sh
  creates it in HF preflight before GPU time).

Next: Phase 2 step 2 — `python runpod_launch.py create` (deploys an A100-80GB
or H100 on-demand SECURE pod, ~$2–5 / ~2.5 h, spends real balance), then commit
status/pod.json and monitor via git per the handoff. Awaiting go-ahead before
spending GPU money.
