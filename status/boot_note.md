# 01:45Z note — pod ao2uyrctwyphqb slow image pull, holding

Rented 01:22:09Z; at +20 min container not yet started (uptime 0, no ports).
Docker Hub shows the pinned tag runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel
is ACTIVE and last_pulled 01:43Z — i.e. this pod's host is actively pulling
the 7.4 GB image, just slowly. Not a retired tag (pod 1's silent death was the
dockerArgs/zero-trace issue, fixed in 48d456e). Holding rather than relaunching
to avoid restarting the pull; ~$0.41 burned idle so far. Will relaunch if the
container still hasn't started by ~01:55Z.

# 01:58Z — pod 2 never started its container (35 min, uptime stuck ≤0); terminated.
# Attempt 3: pod bxn9q01tzvzzdr on a fresh host. Balance $9.09.

# 04:45Z — ROOT CAUSE FOUND: GIT_PUSH_TOKEN cannot git-push (fine-grained PAT
# missing Contents: Read & Write). Proven by dry-run push 403 from sandbox and
# by HF crumbs showing every pod boots, clones and runs pod_run.sh fine
# (dockerargs-start -> podrun-start -> exit crumbs) while zero git markers land.
# This silenced every pod since launch 1. All earlier fixes (idempotent clone,
# image refresh, HF namespace) were real but secondary. Waiting on regenerated
# PAT, then: supervise -> boot marker -> smoke gate -> 13-arm sweep.
