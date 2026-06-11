"""
runpod_launch.py - sandbox-side RunPod orchestration for the CCAT50 sweep.

  python runpod_launch.py create      # pick GPU (A100 80GB preferred, H100 if
                                      # cheaper or A100 unavailable), deploy
                                      # on-demand pod, write status/pod.json
  python runpod_launch.py status      # desiredStatus/runtime of the pod
  python runpod_launch.py terminate   # explicit kill (pod_run.sh normally
                                      # self-terminates)

Secrets are read from env (RUNPOD_API_KEY, HF_TOKEN, GIT_PUSH_TOKEN) and passed
to the pod as RunPod env vars via the API only - never written to disk, status
files, or stdout.
"""
import json
import os
import sys
import urllib.request

API = "https://api.runpod.io/graphql"
HERE = os.path.dirname(os.path.abspath(__file__))
POD_JSON = os.path.join(HERE, "status", "pod.json")
BRANCH = "contrastive-sweep"
# Current stable template line (Dec 2025) - widely cached on hosts, unlike the
# retired 2024 tag whose 7.4 GB cold pull stalled/killed pods 1-3.
IMAGE = "runpod/pytorch:1.0.3-cu1281-torch280-ubuntu2204"
GPU_PREFERRED = ["NVIDIA A100 80GB PCIe", "NVIDIA A100-SXM4-80GB"]
GPU_FALLBACK = ["NVIDIA H100 80GB HBM3", "NVIDIA H100 PCIe"]
# Boot command, hardened by the T-series probes (see status/ notes):
#  - crumb FIRST: pure-curl file commit to the private HF repo proves the
#    container started even if everything after dies (2nd evidence channel).
#  - clone is IDEMPOTENT (T7d: on container restart a bare `clone && run`
#    fails "already exists" and crash-loops forever without ever re-running
#    the script).
#  - repo is public: anonymous clone, no secret expansion in dockerArgs;
#    pod_run.sh wires the authenticated push URL from GIT_PUSH_TOKEN.
_CRUMB = (
    'B64=$(echo "crumb pod=$RUNPOD_POD_ID stage=dockerargs-start '
    'date=$(date -u +%FT%TZ)" | base64 -w0); '
    'printf "{\\"key\\":\\"header\\",\\"value\\":{\\"summary\\":\\"crumb '
    '$RUNPOD_POD_ID dockerargs-start\\"}}\\n{\\"key\\":\\"file\\",\\"value\\":'
    '{\\"path\\":\\"crumbs/${RUNPOD_POD_ID}_dockerargs-start.txt\\",'
    '\\"content\\":\\"$B64\\",\\"encoding\\":\\"base64\\"}}\\n" '
    '| curl -s -m 25 -X POST '
    'https://huggingface.co/api/models/Spartan117Ri/ghost-ckpts/commit/main '
    '-H "Authorization: Bearer $HF_TOKEN" '
    '-H "Content-Type: application/x-ndjson" --data-binary @- >/dev/null 2>&1'
)
DOCKER_ARGS = (
    "bash -c '" + _CRUMB + "; "
    "[ -d /workspace/ghost/.git ] || git clone --branch " + BRANCH +
    " https://github.com/Rishabh117117/ghost.git /workspace/ghost; "
    "bash /workspace/ghost/pod_run.sh'"
)


def gql(query, variables=None):
    key = os.environ["RUNPOD_API_KEY"]
    body = json.dumps({"query": query, "variables": variables or {}}).encode()
    req = urllib.request.Request(API, data=body, headers={
        "Content-Type": "application/json", "Authorization": f"Bearer {key}",
        # RunPod's Cloudflare edge 403s the default Python-urllib User-Agent.
        "User-Agent": "ghost-sweep/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        out = json.loads(r.read())
    if out.get("errors"):
        raise RuntimeError(out["errors"])
    return out["data"]


def gpu_prices():
    data = gql("""query { gpuTypes { id memoryInGb
        lowestPrice(input:{gpuCount:1}) { uninterruptablePrice } } }""")
    return {g["id"]: (g["lowestPrice"] or {}).get("uninterruptablePrice")
            for g in data["gpuTypes"]}


def pick_gpu():
    prices = gpu_prices()
    a100 = [(g, prices[g]) for g in GPU_PREFERRED if prices.get(g)]
    h100 = [(g, prices[g]) for g in GPU_FALLBACK if prices.get(g)]
    a100 = min(a100, key=lambda t: t[1]) if a100 else None
    h100 = min(h100, key=lambda t: t[1]) if h100 else None
    if a100 and (not h100 or a100[1] <= h100[1]):
        return a100
    if h100:
        return h100
    raise RuntimeError(f"no A100-80GB or H100 available; prices seen: "
                       f"{ {k: v for k, v in prices.items() if v} }")


def create():
    for k in ("RUNPOD_API_KEY", "HF_TOKEN", "GIT_PUSH_TOKEN"):
        assert os.environ.get(k), f"{k} missing from env"
    gpu, price = pick_gpu()
    print(f"deploying on-demand: {gpu} @ ${price}/hr", flush=True)
    env_keys = ["RUNPOD_API_KEY", "HF_TOKEN", "GIT_PUSH_TOKEN"]
    data = gql("""mutation($in: PodFindAndDeployOnDemandInput!) {
        podFindAndDeployOnDemand(input: $in) {
            id costPerHr machine { gpuDisplayName } } }""", {"in": {
        "cloudType": "SECURE",
        "gpuCount": 1,
        "gpuTypeId": gpu,
        "name": "ghost-sweep-ccat50",
        "imageName": IMAGE,
        "containerDiskInGb": 60,
        "volumeInGb": 0,
        "dockerArgs": DOCKER_ARGS,
        "env": [{"key": k, "value": os.environ[k]} for k in env_keys],
    }})
    pod = data["podFindAndDeployOnDemand"]
    os.makedirs(os.path.dirname(POD_JSON), exist_ok=True)
    record = {"pod_id": pod["id"], "gpu": pod["machine"]["gpuDisplayName"],
              "cost_per_hr": pod["costPerHr"], "image": IMAGE, "branch": BRANCH}
    with open(POD_JSON, "w") as f:
        json.dump(record, f, indent=1)
    print(json.dumps(record, indent=1), flush=True)


def pod_id():
    return json.load(open(POD_JSON))["pod_id"]


def status():
    data = gql("""query($id: String!) { pod(input:{podId:$id}) {
        id desiredStatus costPerHr runtime { uptimeInSeconds } } }""",
               {"id": pod_id()})
    print(json.dumps(data, indent=1), flush=True)


def terminate():
    gql("""mutation($id: String!) { podTerminate(input:{podId:$id}) }""",
        {"id": pod_id()})
    print("terminate requested", flush=True)


# ---- supervise: own the whole lifecycle, no human babysitting ----------------
import subprocess
import time

BOOT_DEADLINE_S = 12 * 60      # boot-marker commit must appear on the branch
STALL_RELAUNCH_S = 40 * 60     # alive pod but no commits this long -> replace
DEAD_GRACE_S = 3 * 60          # pod gone + no commits this long -> relaunch
MAX_LAUNCHES = 4
MAX_ALLOC_FAILS = 8
WALL_CLOCK_S = 5 * 3600
MIN_BALANCE = 1.50


def sh(*args):
    r = subprocess.run(args, cwd=HERE, capture_output=True, text=True)
    return r.returncode, (r.stdout + r.stderr).strip()


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fetch():
    sh("git", "fetch", "-q", "origin", BRANCH)


def tip():
    return sh("git", "rev-parse", f"origin/{BRANCH}")[1][:9]


def last_commit_age_s():
    _, ct = sh("git", "log", "-1", "--format=%ct", f"origin/{BRANCH}")
    return time.time() - int(ct)


def tree_has(path):
    _, out = sh("git", "ls-tree", "--name-only", f"origin/{BRANCH}", path)
    return out.strip() == path


def stages_tail():
    code, out = sh("git", "show", f"origin/{BRANCH}:status/stages.log")
    return out.splitlines()[-1] if code == 0 and out else ""


def safe_push(msg, *paths):
    sh("git", "add", *paths)
    sh("git", "commit", "-m", msg)
    for i in range(4):
        sh("git", "pull", "--rebase", "origin", BRANCH)
        if sh("git", "push", "origin", BRANCH)[0] == 0:
            return
        time.sleep(2 ** i)
    log("WARN: sandbox push failed after retries")


def pod_state(pid):
    """'no-pod' | 'booting' | 'up'"""
    try:
        d = gql("""query($id: String!) { pod(input:{podId:$id}) {
            desiredStatus runtime { uptimeInSeconds } } }""", {"id": pid})
    except Exception as e:
        log(f"pod query error (treating as transient): {e}")
        return "booting"
    p = d.get("pod")
    if not p or p["desiredStatus"] in ("EXITED", "TERMINATED"):
        return "no-pod"
    rt = p.get("runtime") or {}
    return "up" if (rt.get("uptimeInSeconds") or 0) > 0 else "booting"


def balance():
    return gql("query { myself { clientBalance } }")["myself"]["clientBalance"]


def done():
    return tree_has("SWEEP_CCAT50.md") and tree_has("results.json")


def aborted():
    return tree_has("status/ABORT.json")


def supervise():
    t0, launches, alloc_fails = time.time(), 0, 0
    while time.time() - t0 < WALL_CLOCK_S:
        fetch()
        if done():
            break
        bal = balance()
        if bal < MIN_BALANCE:
            log(f"STOP: balance ${bal:.2f} below floor ${MIN_BALANCE}")
            return 2
        if launches >= MAX_LAUNCHES:
            log(f"STOP: {launches} launches without completion - needs a human")
            return 2
        try:
            create()
        except Exception as e:
            alloc_fails += 1
            log(f"create failed ({alloc_fails}/{MAX_ALLOC_FAILS}): {e}")
            if alloc_fails >= MAX_ALLOC_FAILS:
                return 2
            time.sleep(180)
            continue
        launches += 1
        pid = pod_id()
        safe_push(f"supervise: launch {launches} pod {pid}", "status/pod.json")
        fetch()
        base = tip()
        log(f"launch {launches}: pod {pid}, balance ${bal:.2f}, waiting on boot marker")

        # ---- boot watch: marker commit must land within the deadline --------
        boot_t = time.time()
        booted = False
        while time.time() - boot_t < BOOT_DEADLINE_S:
            time.sleep(45)
            fetch()
            if tip() != base:
                booted = True
                break
            if pod_state(pid) == "no-pod":
                log("pod vanished pre-boot")
                break
        if not booted:
            log(f"no boot marker in {int(time.time()-boot_t)}s - replacing host")
            try:
                terminate()
            except Exception:
                pass
            time.sleep(20)
            continue
        log(f"booted: branch moved {base} -> {tip()}")

        # ---- progress watch --------------------------------------------------
        while time.time() - t0 < WALL_CLOCK_S:
            time.sleep(60)
            fetch()
            if done():
                log("results landed: SWEEP_CCAT50.md + results.json on branch")
                break
            if aborted():
                log("cost-guard ABORT.json on branch - stopping (no relaunch)")
                return 1
            gap, st = last_commit_age_s(), pod_state(pid)
            if st == "no-pod" and gap > DEAD_GRACE_S:
                log(f"pod gone, last commit {int(gap/60)}m ago "
                    f"(last stage: '{stages_tail()}') - relaunching, arms resume")
                break
            if gap > STALL_RELAUNCH_S:
                log(f"stalled {int(gap/60)}m with pod {st} - replacing pod")
                try:
                    terminate()
                except Exception:
                    pass
                time.sleep(20)
                break
        if done():
            break

    fetch()
    if not done():
        log("STOP: wall clock exhausted without completion")
        return 3
    # ---- completion: make sure nothing is still billing ----------------------
    if pod_state(pod_id()) != "no-pod":
        log("results in but pod still alive - terminating")
        try:
            terminate()
        except Exception:
            pass
    log(f"DONE in {int((time.time()-t0)/60)} min over {launches} launch(es); "
        f"final balance ${balance():.2f}; last stage: '{stages_tail()}'")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    fns = {"create": create, "status": status, "terminate": terminate,
           "supervise": lambda: sys.exit(supervise())}
    fns[cmd]()
