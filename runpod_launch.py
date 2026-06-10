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
IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
GPU_PREFERRED = ["NVIDIA A100 80GB PCIe", "NVIDIA A100-SXM4-80GB"]
GPU_FALLBACK = ["NVIDIA H100 80GB HBM3", "NVIDIA H100 PCIe"]
DOCKER_ARGS = (
    "bash -c 'git clone --branch " + BRANCH +
    " https://x-access-token:${GIT_PUSH_TOKEN}@github.com/Rishabh117117/ghost.git"
    " /workspace/ghost && bash /workspace/ghost/pod_run.sh'"
)


def gql(query, variables=None):
    key = os.environ["RUNPOD_API_KEY"]
    body = json.dumps({"query": query, "variables": variables or {}}).encode()
    req = urllib.request.Request(API, data=body, headers={
        "Content-Type": "application/json", "Authorization": f"Bearer {key}"})
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


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    {"create": create, "status": status, "terminate": terminate}[cmd]()
