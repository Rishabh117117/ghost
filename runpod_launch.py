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
BRANCH = "engram-v2"
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
        "name": "ghost-engram-v2",
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
# Transport is HF-ONLY (the pod's PAT can't git-push). The pod publishes crumbs,
# stage logs, arm JSONs and final artifacts to the private HF repo; the
# supervisor (sandbox, which CAN git-push) detects state there and mirrors the
# final results into the branch.
import datetime
import subprocess
import time

HF_REPO = "Spartan117Ri/ghost-ckpts"
HF_API = f"https://huggingface.co/api/models/{HF_REPO}"
HF_RESOLVE = f"https://huggingface.co/{HF_REPO}/resolve/main"

BOOT_DEADLINE_S = 12 * 60      # podrun-start crumb must land on HF by now
STALL_RELAUNCH_S = 40 * 60     # alive pod, no HF commit this long -> replace
DEAD_GRACE_S = 3 * 60          # pod gone + no HF commit this long -> relaunch
MAX_LAUNCHES = 4
MAX_ALLOC_FAILS = 8
WALL_CLOCK_S = 5 * 3600
MIN_BALANCE = 1.50


def sh(*args):
    r = subprocess.run(args, cwd=HERE, capture_output=True, text=True)
    return r.returncode, (r.stdout + r.stderr).strip()


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def hf_req(url, raw=False):
    req = urllib.request.Request(url, headers={
        "Authorization": f"Bearer {os.environ['HF_TOKEN']}",
        "User-Agent": "ghost-sweep/1.0"})
    with urllib.request.urlopen(req, timeout=40) as r:
        data = r.read()
    return data if raw else json.loads(data)


def hf_tree(path=""):
    url = HF_API + "/tree/main" + (f"/{path}" if path else "")
    try:
        return hf_req(url)
    except Exception:
        return []


def hf_has(path):
    parent = path.rsplit("/", 1)[0] if "/" in path else ""
    return any(f.get("path") == path for f in hf_tree(parent))


def hf_text(path):
    try:
        return hf_req(f"{HF_RESOLVE}/{path}", raw=True).decode()
    except Exception:
        return ""


def hf_to_file(path, dst):
    data = hf_req(f"{HF_RESOLVE}/{path}", raw=True)
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    with open(dst, "wb") as f:
        f.write(data)


def hf_last_commit_age_s():
    """Seconds since the HF repo's most recent commit = last pod activity."""
    try:
        c = hf_req(f"https://huggingface.co/api/models/{HF_REPO}/commits/main")
        when = c[0]["date"] if isinstance(c, list) else c["commits"][0]["date"]
        dt = datetime.datetime.fromisoformat(when.replace("Z", "+00:00"))
        return (datetime.datetime.now(datetime.timezone.utc) - dt).total_seconds()
    except Exception:
        return 1e9


def booted(pid):
    return hf_has(f"crumbs/{pid}_podrun-start.txt") or hf_has(f"runs/{pid}/stages.log")


def stages_tail(pid):
    t = hf_text(f"runs/{pid}/stages.log").strip()
    return t.splitlines()[-1] if t else ""


def done():
    return hf_has("runs/final/results.json") and hf_has("runs/final/ENGRAM_V2.md")


def aborted(pid):
    return hf_has(f"runs/{pid}/ABORT.json")


def safe_push(msg, *paths):
    sh("git", "add", *paths)
    sh("git", "commit", "-m", msg)
    for i in range(4):
        sh("git", "pull", "--rebase", "origin", BRANCH)
        if sh("git", "push", "origin", BRANCH)[0] == 0:
            return
        time.sleep(2 ** i)
    log("WARN: sandbox push failed after retries")


def mirror_final_to_branch(pid):
    """Pull pod's HF artifacts down and commit them into the branch."""
    got = []
    for rp, dst in (("runs/final/results.json", "results.json"),
                    ("runs/final/ENGRAM_V2.md", "ENGRAM_V2.md"),
                    (f"runs/{pid}/stages.log", "status/stages.log"),
                    (f"runs/{pid}/run.log", "status/run.log")):
        try:
            hf_to_file(rp, os.path.join(HERE, dst))
            got.append(dst)
        except Exception:
            pass
    # per-arm JSONs / interference, best effort
    for f in hf_tree("runs/final/engram"):
        p = f.get("path", "")
        if p.endswith((".json", ".png", ".csv")):
            try:
                hf_to_file(p, os.path.join(HERE, "results", "engram",
                                           os.path.basename(p)))
                got.append(p)
            except Exception:
                pass
    if got:
        safe_push(f"engram results mirrored from HF (pod {pid})",
                  "results.json", "ENGRAM_V2.md", "status", "results")
    return got


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


def clear_final():
    """Delete any prior-run final artifacts so done() can't trip at startup on
    a stale ENGRAM_V*.md from a previous run (a 0-launch false 'DONE')."""
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=os.environ["HF_TOKEN"])
        stale = [f for f in api.list_repo_files(HF_REPO, repo_type="model")
                 if f.startswith("runs/final/")]
        for f in stale:
            api.delete_file(f, HF_REPO, repo_type="model")
        if stale:
            log(f"cleared {len(stale)} stale runs/final artifacts before launch")
    except Exception as e:
        log(f"clear_final skipped: {e}")


def supervise():
    clear_final()
    t0, launches, alloc_fails = time.time(), 0, 0
    while time.time() - t0 < WALL_CLOCK_S:
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
        log(f"launch {launches}: pod {pid}, balance ${bal:.2f}, waiting on HF boot crumb")

        # ---- boot watch: podrun-start crumb must land on HF -----------------
        boot_t = time.time()
        is_booted = False
        while time.time() - boot_t < BOOT_DEADLINE_S:
            time.sleep(45)
            if booted(pid):
                is_booted = True
                break
            if pod_state(pid) == "no-pod":
                log("pod vanished pre-boot")
                break
        if not is_booted:
            log(f"no HF boot crumb in {int(time.time()-boot_t)}s - replacing host")
            try:
                terminate()
            except Exception:
                pass
            time.sleep(20)
            continue
        log(f"booted: HF crumb present for {pid} (stage: '{stages_tail(pid)}')")

        # ---- progress watch --------------------------------------------------
        while time.time() - t0 < WALL_CLOCK_S:
            time.sleep(60)
            if done():
                log("final artifacts on HF: results.json + SWEEP_CCAT50.md")
                break
            if aborted(pid):
                log("cost-guard ABORT.json on HF - stopping (no relaunch)")
                mirror_final_to_branch(pid)
                return 1
            gap, st = hf_last_commit_age_s(), pod_state(pid)
            if st == "no-pod" and gap > DEAD_GRACE_S:
                log(f"pod gone, last HF commit {int(gap/60)}m ago "
                    f"(stage: '{stages_tail(pid)}') - relaunching, arms resume")
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

    if not done():
        log("STOP: wall clock exhausted without completion")
        return 3
    # ---- completion: mirror to branch + make sure nothing is still billing ---
    pid = pod_id()
    got = mirror_final_to_branch(pid)
    if pod_state(pid) != "no-pod":
        log("results in but pod still alive - terminating")
        try:
            terminate()
        except Exception:
            pass
    log(f"DONE in {int((time.time()-t0)/60)} min over {launches} launch(es); "
        f"final balance ${balance():.2f}; mirrored: {got}")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    fns = {"create": create, "status": status, "terminate": terminate,
           "supervise": lambda: sys.exit(supervise())}
    fns[cmd]()
