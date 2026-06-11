# Status: Phase 2 BLOCKED at step 1 — RunPod/HF hosts still not in allowlist

2026-06-11 — new session ran HANDOFF.md Phase 2 step 1 (live secret verification).
Results (values never printed):

- GIT_PUSH_TOKEN: OK — GitHub API confirms push=True, admin=True on
  Rishabh117117/ghost. Pod-side push path is good.
- RUNPOD_API_KEY: could NOT be verified — POST api.runpod.io/graphql returns
  HTTP 403 "Host not in allowlist".
- HF_TOKEN: could NOT be verified — GET huggingface.co/api/whoami-v2 returns
  HTTP 403 "Host not in allowlist".

The environment network policy still blocks api.runpod.io and huggingface.co.
`runpod_launch.py create` needs api.runpod.io; the pod needs huggingface.co
(checkpoint home) — so the GPU run cannot start. Stopped here per the handoff's
step-1 rule ("stop and report, don't improvise").

Unblock: add api.runpod.io and huggingface.co (and the pod also needs
archive.ics.uci.edu for the CCAT50 zip) to the environment's network allowlist,
then re-run HANDOFF.md from Phase 2 step 1.
