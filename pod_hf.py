"""
pod_hf.py - tiny HF-transport CLI for the pod (and sandbox mirroring).

The pod's GitHub token cannot push (fine-grained PAT without Contents:write),
so the private HF repo is the pod's ONLY upstream channel:

  python pod_hf.py up    <local_file> <repo_path>     # one file
  python pod_hf.py updir <local_dir>  <repo_prefix>   # recursive small files
  python pod_hf.py down  <repo_prefix> <local_dir>    # restore (arm resume)

Uses HF_TOKEN from env. Tolerant: failures print a warning and exit 0 unless
PODHF_STRICT=1, because evidence uploads must never kill the pipeline.
"""
import os
import sys

from huggingface_hub import HfApi

REPO = "Spartan117Ri/ghost-ckpts"


def api():
    return HfApi(token=os.environ["HF_TOKEN"])


def up(local, repo_path):
    api().upload_file(path_or_fileobj=local, path_in_repo=repo_path,
                      repo_id=REPO, repo_type="model")


def updir(local_dir, prefix):
    a = api()
    for root, _, files in os.walk(local_dir):
        for f in files:
            p = os.path.join(root, f)
            rel = os.path.relpath(p, local_dir)
            a.upload_file(path_or_fileobj=p,
                          path_in_repo=f"{prefix}/{rel}",
                          repo_id=REPO, repo_type="model")


def down(prefix, local_dir):
    a = api()
    os.makedirs(local_dir, exist_ok=True)
    try:
        names = [f for f in a.list_repo_files(REPO)
                 if f.startswith(prefix.rstrip("/") + "/")]
    except Exception:
        names = []
    for n in names:
        dst = os.path.join(local_dir, os.path.relpath(n, prefix))
        os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
        a.hf_hub_download(REPO, n, local_dir="/tmp/_hfdl")
        src = os.path.join("/tmp/_hfdl", n)
        os.replace(src, dst)
        print("restored", dst)


if __name__ == "__main__":
    cmd, args = sys.argv[1], sys.argv[2:]
    try:
        {"up": up, "updir": updir, "down": down}[cmd](*args)
    except Exception as e:  # tolerant by default - evidence must not kill runs
        print(f"[pod_hf] WARN {cmd} failed: {e}", file=sys.stderr)
        if os.environ.get("PODHF_STRICT") == "1":
            sys.exit(1)
