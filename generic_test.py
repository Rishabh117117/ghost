"""
generic_test.py - generic-vs-specific diagnostic for the ghost bank.

Question: how much of a ghost's perplexity benefit is domain-AGNOSTIC? Evaluate the
EXISTING ghosts (no training) on a third domain unrelated to the user's data
(wikitext-2-raw-v1), chat-masked identically to Stage 1/2, and compare each ghost's
relative improvement on {its own domain, the other same-export domain, the unrelated
domain3}.

  SPECIALISTS  -> a ghost's win stays confined to its domain (domain3 improvement ~ 0)
  GENERALISTS  -> a ghost recovers much of its win on unrelated domain3 too
  (likely graded -> report the fractions)

Architecture untouched: everything imported from ghost.py / chat_test.py / bank.py.
"""
import os
import math

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ghost import (
    GhostModel, MODEL_NAME, DEVICE, SEED, MAX_LEN,
    CORPUS_PATH as VOICE_PATH, load_corpus, split_corpus, base_fingerprint,
)
from chat_test import build_examples, build_chat_example, _im_end_id
from bank import (
    GHOST_A_CKPT, GHOST_B_CKPT, CORPUS_B_PATH,
    per_example_losses, agg_ppl, keep_awake,
)

HERE = os.path.dirname(os.path.abspath(__file__))
DOMAIN3_FALLBACK = os.path.join(HERE, "data", "domain3.txt")
N = 80                      # per-domain held-out cap (same as Stage 2)
MIN_CHARS = 40


def load_domain3():
    """Neutral public corpus unrelated to the user's data: wikitext-2-raw-v1[test].
    Fallback: data/domain3.txt. Returns (lines, source_label)."""
    try:
        from datasets import load_dataset
        ds = None
        last_err = None
        for repo in ("Salesforce/wikitext", "wikitext"):   # canonical first, legacy second
            try:
                ds = load_dataset(repo, "wikitext-2-raw-v1", split="test")
                break
            except Exception as e:
                last_err = e
        if ds is None:
            raise last_err
        lines = [t.strip() for t in ds["text"] if len(t.strip()) >= MIN_CHARS]
        lines = lines[:300]
        if len(lines) >= 50:
            return lines, "wikitext-2-raw-v1[test]"
        raise RuntimeError("too few lines from wikitext")
    except Exception as e:
        print(f"(datasets unavailable: {e}; trying fallback)", flush=True)
        if os.path.isfile(DOMAIN3_FALLBACK):
            raw = open(DOMAIN3_FALLBACK, encoding="utf-8").read()
            lines = [l.strip() for l in raw.splitlines() if len(l.strip()) >= MIN_CHARS]
            return lines, f"fallback {os.path.basename(DOMAIN3_FALLBACK)}"
        raise RuntimeError("no domain3 source (no datasets, no data/domain3.txt)")


def load_ghost(model, ckpt):
    model.ghost.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.ghost.alpha.fill_(1.0)


if __name__ == "__main__":
    torch.manual_seed(SEED)
    keep_awake()
    print(f"loading {MODEL_NAME} on {DEVICE} ...", flush=True)
    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    try:
        base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype="auto").to(DEVICE)
    except TypeError:
        base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype="auto").to(DEVICE)
    model = GhostModel(base).to(DEVICE)
    assert not any(p.requires_grad for p in model.base.parameters()), "base must be frozen"
    im_end_id = _im_end_id(tok)
    fp0 = base_fingerprint(model)

    # domains: voice + tool (same held-out split as Stage 2) + domain3 (unrelated)
    A_train, A_val = split_corpus(load_corpus(VOICE_PATH))
    B_train, B_val = split_corpus(load_corpus(CORPUS_B_PATH))
    d3_lines, d3_src = load_domain3()
    print(f"\ndomain3 source: {d3_src}  ({len(d3_lines)} lines)", flush=True)

    voice_ex = build_examples(tok, A_val[:N], im_end_id)
    tool_ex  = build_examples(tok, B_val[:N], im_end_id)
    d3_ex    = build_examples(tok, d3_lines[:N], im_end_id)
    print(f"chat examples: voice {len(voice_ex)} | tool {len(tool_ex)} | domain3 {len(d3_ex)}", flush=True)

    # ---- ONE worked domain3 example (trust the masking on the new corpus) ----
    ex = build_chat_example(tok, d3_lines[0], im_end_id)
    assert ex is not None
    ids, labels = ex
    scored = [t for t, l in zip(ids[0].tolist(), labels[0].tolist()) if l != -100]
    load_ghost(model, GHOST_B_CKPT)
    with torch.no_grad():
        _, lb = model(ids.to(DEVICE), labels=labels.to(DEVICE), use_ghost=False)
        _, lg = model(ids.to(DEVICE), labels=labels.to(DEVICE), use_ghost=True)
    print("\n" + "=" * 78)
    print("WORKED DOMAIN3 EXAMPLE")
    print("=" * 78)
    print(f"line (truncated): {d3_lines[0][:140]!r}")
    print(f"SCORED span (labels != -100, should be only the line's tokens):")
    print(f"  {tok.decode(scored, skip_special_tokens=False)[:160]!r}")
    print(f"base loss = {lb.item():.4f}   +GhostB loss = {lg.item():.4f}")
    print("=" * 78, flush=True)

    domains = {"voice": voice_ex, "tool": tool_ex, "domain3": d3_ex}

    # base (ghost-independent): use_ghost=False
    base_ppl = {d: agg_ppl(per_example_losses(model, ex, use_ghost=False)) for d, ex in domains.items()}
    fp_base = base_fingerprint(model)

    load_ghost(model, GHOST_A_CKPT)
    A_ppl = {d: agg_ppl(per_example_losses(model, ex, use_ghost=True)) for d, ex in domains.items()}
    fp_A = base_fingerprint(model)

    load_ghost(model, GHOST_B_CKPT)
    B_ppl = {d: agg_ppl(per_example_losses(model, ex, use_ghost=True)) for d, ex in domains.items()}
    fp_B = base_fingerprint(model)

    def impr(d, g):  # relative improvement over base
        return (base_ppl[d] - g[d]) / base_ppl[d] * 100.0

    # ---- report ----
    print("\n" + "#" * 78)
    print("RESULTS  (chat-masked held-out perplexity, assistant tokens only)")
    print("#" * 78)
    print(f"{'domain':<9} {'base':>10} {'+GhostA':>10} {'+GhostB':>10}")
    for d in domains:
        print(f"{d:<9} {base_ppl[d]:>10.2f} {A_ppl[d]:>10.2f} {B_ppl[d]:>10.2f}")

    print("\nRELATIVE IMPROVEMENT over base  (positive = ghost helps):")
    print(f"{'domain':<9} {'GhostA %':>10} {'GhostB %':>10}")
    for d in domains:
        print(f"{d:<9} {impr(d, A_ppl):>10.1f} {impr(d, B_ppl):>10.1f}")

    print(f"\nPROBE 2 (base frozen): fp deltas  base={fp_base-fp0:.2e}  "
          f"A={fp_A-fp0:.2e}  B={fp_B-fp0:.2e}  (must be 0)")

    # ---- verdict: how much of each ghost's OWN-domain win carries to unrelated domain3 ----
    A_own, A_unrel = impr("voice", A_ppl), impr("domain3", A_ppl)
    B_own, B_unrel = impr("tool",  B_ppl), impr("domain3", B_ppl)
    A_frac = A_unrel / A_own if A_own > 0 else 0.0
    B_frac = B_unrel / B_own if B_own > 0 else 0.0
    print(f"\ngeneralization fraction (domain3 improvement / own-domain improvement):")
    print(f"  GhostA: own(voice)={A_own:.1f}%  domain3={A_unrel:.1f}%  fraction={A_frac:.2f}")
    print(f"  GhostB: own(tool)={B_own:.1f}%   domain3={B_unrel:.1f}%  fraction={B_frac:.2f}")

    fracs = [f for f in (A_frac, B_frac)]
    if max(fracs) < 0.25:
        verdict = "SPECIALISTS"
    elif min(fracs) > 0.60:
        verdict = "GENERALISTS"
    else:
        verdict = "MIXED"
    print(f"\nVERDICT: {verdict}")
