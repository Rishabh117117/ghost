"""
sweep_ccat50.py - contrastive sweep on CCAT50: lambda x negative-type factorial.

CONTRASTIVE_CCAT50.md (58feaa8): leak REPLICATED at 0.471; the lambda=1.0
self-paraphrase hinge cure FAILED (leak 0.332, missed both bars) while retaining
86% own-win. Open question is two-dimensional: MAGNITUDE (lambda too small) vs
NEGATIVES-DIVERSITY (self-paraphrases too narrow). This sweep is the factorial
that disentangles them, and the Stage-2 gate: no arm passes -> CE+hinge is
structurally insufficient at this scale.

Arms (13, sequential, ONE base load for the whole run):
  arm 0:     plain ghost (lambda=0) - fresh anchor on this hardware
  arms 1-12: lambda in {0.5, 1, 2, 4} x negatives in {self, author, both}
    self:   neutral self-paraphrases of the target's train docs (exact recipe
            from contrastive_ccat50.make_pairs; regenerated on the pod)
    author: 50 docs from the 10 seen-contrast authors' C50train splits
            (5/author, seed=42); CE_base precomputed once
    both:   hinge = mean of the two hinge terms (lambda applied once)

Eval grid (rows = base, style-prompt, all arms; identical token spans per col):
  A_target:   target C50test (own-win)
  B_seen:     C50test of the 10 authors whose C50train fed negatives
  C_unseen:   C50test of the NEXT 10 authors alphabetically (never in training)
  D_wikitext: wikitext-2 (same filter as generic_test.load_domain3)

Machinery reused byte-for-byte from contrastive_ccat50.py (download, masking,
220-token pre-truncation, agg_ppl eval, v03 recipe). Architecture untouched.
--smoke: tiny random LlamaConfig base + scratch tokenizer + fake data, 2 arms x
2 epochs, CPU - proves the full plumbing (grid + report) with zero downloads.
"""
import argparse
import json
import math
import os
import random
import threading
import time
from datetime import datetime, timezone

import torch

from ghost import (
    GhostModel, GhostStream, MODEL_NAME, DEVICE, SEED, MAX_LEN, D_GHOST,
    ALPHA_INIT, LR, WEIGHT_DECAY, MAX_EPOCHS, PATIENCE,
    base_fingerprint, param_counts,
)
from chat_test import _im_end_id, build_chat_example, NEUTRAL_PROMPT
from bank import per_example_losses, agg_ppl, keep_awake
from generic_test import load_domain3
from contrastive_ccat50 import (
    download_ccat50, list_authors, read_docs, build_example_prompted,
    make_pairs, split_idx, masked_ce,
    TARGET_AUTHOR, N_CONTRAST, N_CONTRAST_DOCS, ALPHA_SERVE,
)

HERE = os.path.dirname(os.path.abspath(__file__))

LAMBDAS          = (0.5, 1.0, 2.0, 4.0)
NEG_TYPES        = ("self", "author", "both")
N_NEG_PER_AUTHOR = 5          # author-negative docs sampled per seen author
NEG_SAMPLE_SEED  = 42
HUB_REPO         = "Rishabh117117/ghost-ckpts"
HUB_PREFIX       = "sweep-ccat50"
BUDGET_SECONDS   = 13 * 10 * 60   # ~10 min/arm on A100 -> ~2.2 h
COST_GUARD_X     = 3              # abort if projection > 3x budget ...
COST_GUARD_ARM   = 3              # ... measured once arm index >= 3 is done

# pass bars (per re-dispatch / DISPATCH_sweep_ccat50.md Phase 3)
BAR_ABS_LEAK  = 0.10   # seen-leak fraction <= 0.10 absolute
BAR_REL_LEAK  = 0.5    # seen-leak fraction <= 0.5 x arm-0 leak fraction
BAR_RETENTION = 0.70   # own-win >= 70% of arm-0 own-win


def all_arms():
    arms = [{"i": 0, "lam": 0.0, "neg": "none"}]
    i = 1
    for lam in LAMBDAS:
        for neg in NEG_TYPES:
            arms.append({"i": i, "lam": lam, "neg": neg})
            i += 1
    return arms


ARMS = all_arms()


def arm_name(a):
    return f"arm_{a['i']}_{a['lam']:g}_{a['neg']}"


# ---- heartbeat (status/ is the pod's only window to the sandbox) -------------
STATUS = {"phase": "init", "arm": None, "epoch": None, "step": None}
_STOP = threading.Event()


def start_heartbeat(status_dir, period=60):
    os.makedirs(status_dir, exist_ok=True)

    def loop():
        while True:
            now = datetime.now(timezone.utc).isoformat()
            with open(os.path.join(status_dir, "heartbeat.log"), "a") as f:
                f.write(f"{now} phase={STATUS['phase']} arm={STATUS['arm']} "
                        f"epoch={STATUS['epoch']} step={STATUS['step']}\n")
            with open(os.path.join(status_dir, "current.json"), "w") as f:
                json.dump({"ts": now, **STATUS}, f, indent=1)
            if _STOP.wait(period):
                return

    t = threading.Thread(target=loop, daemon=True)
    t.start()
    return t


# ---- negatives ----------------------------------------------------------------
def chat_ex_aligned(tok, docs, im_end_id):
    """build_chat_example over docs, KEEPING None at dropped indices so negative
    sets stay index-aligned with the train docs (the hinge pairs by index)."""
    return [build_chat_example(tok, d, im_end_id, max_len=MAX_LEN) if d else None
            for d in docs]


def sample_author_negatives(tok, seen_authors, n_per_author=N_NEG_PER_AUTHOR,
                            seed=NEG_SAMPLE_SEED):
    """5 docs/author from the seen authors' C50train splits, seed=42 -> 50 docs.
    No paraphrase generation; these are real other-author texts."""
    rng = random.Random(seed)
    out = []
    for a in seen_authors:
        docs = read_docs(a, "C50train", tok)
        out.extend(rng.sample(docs, n_per_author))
    return out


def precompute_base_ce(model, ex_list):
    out = []
    with torch.no_grad():
        for e in ex_list:
            out.append(masked_ce(model, e, use_ghost=False).item() if e else None)
    return out


# ---- training (v03 recipe + factorial hinge) -----------------------------------
def fresh_ghost(model, d_ghost):
    d_model = model.base.config.hidden_size
    n_taps = model.base.config.num_hidden_layers + 1
    model.ghost = GhostStream(d_model, n_taps, d_ghost).to(DEVICE, dtype=model.base.dtype)
    model.ghost.alpha.fill_(ALPHA_INIT)


def train_arm_sweep(model, doc_ex, negs, ckpt, lam, neg_type, d_ghost,
                    max_epochs=MAX_EPOCHS, lr=LR, weight_decay=WEIGHT_DECAY,
                    patience=PATIENCE, seed=SEED):
    """Resumable trainer, same shape as contrastive_ccat50.train_arm but the
    hinge draws from configurable negative sets. negs: {name: (ex_list, ce_list)},
    index-aligned with doc_ex. Per doc i:
      L_i = CE_g(doc_i) + lam * mean_over_active_sets( max(0, CE_b(n_i) - CE_g(n_i)) )
    'both' -> mean of the two hinge terms (lambda applied once, not doubled)."""
    active = {"none": (), "self": ("self",), "author": ("author",),
              "both": ("self", "author")}[neg_type]
    done_flag = ckpt + ".done"
    tr_idx, va_idx = split_idx(len(doc_ex))
    tr_idx = [i for i in tr_idx if doc_ex[i] is not None]
    va_idx = [i for i in va_idx if doc_ex[i] is not None]

    def hinge(i):
        terms = []
        for name in active:
            ex_list, ce_list = negs[name]
            if ex_list[i] is not None and ce_list[i] is not None:
                g = masked_ce(model, ex_list[i], use_ghost=True)
                terms.append(torch.clamp(
                    torch.tensor(ce_list[i], device=DEVICE) - g, min=0.0))
        if not terms:
            return None
        return sum(terms) / len(terms)

    @torch.no_grad()
    def val_loss():
        model.eval()
        tot = 0.0
        for i in va_idx:
            tot += masked_ce(model, doc_ex[i], use_ghost=True).item()
            if lam > 0:
                h = hinge(i)
                if h is not None:
                    tot += lam * h.item()
        return tot / max(len(va_idx), 1)

    if os.path.isfile(done_flag) and os.path.isfile(ckpt):
        model.ghost.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        model.ghost.alpha.fill_(1.0)
        print(f"arm already converged: {os.path.basename(ckpt)}", flush=True)
        return {"resumed_done": True}
    if os.path.isfile(ckpt):
        model.ghost.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        model.ghost.alpha.fill_(1.0)
        best = val_loss()
        print(f"resuming from checkpoint (val {best:.4f})", flush=True)
    else:
        fresh_ghost(model, d_ghost)
        best = float("inf")
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                            lr=lr, weight_decay=weight_decay)
    rng = random.Random(seed)
    bad, epochs_run = 0, 0
    order = tr_idx[:]
    for epoch in range(max_epochs):
        model.train()
        rng.shuffle(order)
        running, nb = 0.0, 0
        STATUS["epoch"] = epoch
        for step, i in enumerate(order):
            STATUS["step"] = step
            loss = masked_ce(model, doc_ex[i], use_ghost=True)
            if lam > 0:
                h = hinge(i)
                if h is not None:
                    loss = loss + lam * h
            opt.zero_grad(); loss.backward(); opt.step()
            running += loss.item(); nb += 1
        epochs_run = epoch + 1
        vl = val_loss()
        print(f"epoch {epoch:3d} | train {running/max(nb,1):.4f} | val {vl:.4f}", flush=True)
        if vl < best - 1e-4:
            best, bad = vl, 0
            model.ghost.alpha.fill_(1.0)
            torch.save(model.ghost.state_dict(), ckpt)
            print(f"  checkpointed -> {os.path.basename(ckpt)}", flush=True)
        else:
            bad += 1
            if bad >= patience:
                print(f"early stop (best val {best:.4f})", flush=True)
                break
    open(done_flag, "w").close()
    model.ghost.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.ghost.alpha.fill_(1.0)
    return {"best_val": best, "epochs_run": epochs_run}


# ---- checkpoint second home (private HF repo) ----------------------------------
def push_ckpt_to_hub(ckpt, arm, smoke):
    """Upload the arm's best checkpoint to the private hub repo. Retries with
    backoff; on persistent failure records the error and lets the sweep continue
    (GPU time > one delayed copy; the failure is surfaced in the arm JSON)."""
    if smoke:
        return "skipped (smoke)"
    token = os.environ.get("HF_TOKEN")
    if not token:
        return "FAILED: HF_TOKEN not set"
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    last = None
    for attempt in range(4):
        try:
            api.create_repo(HUB_REPO, private=True, exist_ok=True)
            api.upload_file(path_or_fileobj=ckpt, repo_id=HUB_REPO,
                            path_in_repo=f"{HUB_PREFIX}/{arm_name(arm)}/ghost.pt")
            return "ok"
        except Exception as e:
            last = e
            time.sleep(2 ** (attempt + 1))
    return f"FAILED: {last}"


# ---- smoke fixtures (tiny random base, scratch tokenizer, fake data) ------------
SMOKE_VOCAB = {
    "target":  "market trade bank rates pressman online privacy crypto policy net".split(),
    "seen":    "oil metal mining shares profit czech prague tokyo bonds yen".split(),
    "unseen":  "rugby cricket match coach season league score title cup win".split(),
    "wiki":    "history century village river castle museum painting opera novel king".split(),
}


def smoke_sentences(rng, flavor, n, words=24):
    pool = SMOKE_VOCAB[flavor] + "the a of in and to was for with on".split()
    return [" ".join(rng.choice(pool) for _ in range(words)) for _ in range(n)]


def smoke_tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast
    rng = random.Random(SEED)
    corpus = [s for fl in SMOKE_VOCAB for s in smoke_sentences(rng, fl, 20)]
    corpus.append(NEUTRAL_PROMPT)
    tk = Tokenizer(models.BPE(unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tk.train_from_iterator(corpus, trainers.BpeTrainer(
        vocab_size=600,
        special_tokens=["<unk>", "<|endoftext|>", "<|im_start|>", "<|im_end|>"]))
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="<unk>",
                                  eos_token="<|endoftext|>", pad_token="<|endoftext|>")
    tok.add_special_tokens({"additional_special_tokens": ["<|im_start|>", "<|im_end|>"]})
    # minimal Qwen-shaped template: build_chat_example's LCP boundary trick needs
    # only that the empty-assistant render be a prefix of the full render
    tok.chat_template = (
        "{% for message in messages %}<|im_start|>{{ message['role'] }}\n"
        "{{ message['content'] }}<|im_end|>\n{% endfor %}"
        "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")
    return tok


def smoke_base(vocab_size):
    from transformers import LlamaConfig, LlamaForCausalLM
    cfg = LlamaConfig(vocab_size=vocab_size, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=4, max_position_embeddings=2048)
    torch.manual_seed(SEED)
    return LlamaForCausalLM(cfg)


# ---- eval grid ------------------------------------------------------------------
def build_cols(tok, im_end_id, docs_by_col, style_prompt):
    cols = {}
    for c, docs in docs_by_col.items():
        neutral = [build_example_prompted(tok, NEUTRAL_PROMPT, d, im_end_id, MAX_LEN)
                   for d in docs]
        style = [build_example_prompted(tok, style_prompt, d, im_end_id, 2048)
                 for d in docs]
        cols[c] = {"neutral": [e for e in neutral if e is not None],
                   "style":   [e for e in style if e is not None]}
    return cols


def eval_row(model, cols, variant, use_ghost):
    return {c: agg_ppl(per_example_losses(model, v[variant], use_ghost=use_ghost))
            for c, v in cols.items()}


# ---- report ----------------------------------------------------------------------
COLS = ("A_target", "B_seen", "C_unseen", "D_wikitext")


def _rel(base_ppl, row_ppl, col):
    return (base_ppl[col] - row_ppl[col]) / base_ppl[col] * 100.0


def _frac(num, den):
    return num / den if den and den > 0 else float("nan")


def analyze(base_rows, arm_results):
    """Everything derived: per-arm rel%, leak fractions, bars vs arm 0, verdict."""
    base_ppl = base_rows["base"]
    arm0 = next((r for r in arm_results if r["i"] == 0), None)
    out = []
    a0_own = _rel(base_ppl, arm0["ppl"], "A_target") if arm0 else float("nan")
    a0_leak_b = _frac(_rel(base_ppl, arm0["ppl"], "B_seen"), a0_own) if arm0 else float("nan")
    a0_leak_c = _frac(_rel(base_ppl, arm0["ppl"], "C_unseen"), a0_own) if arm0 else float("nan")
    for r in arm_results:
        own = _rel(base_ppl, r["ppl"], "A_target")
        leak_b = _frac(_rel(base_ppl, r["ppl"], "B_seen"), own)
        leak_c = _frac(_rel(base_ppl, r["ppl"], "C_unseen"), own)
        retention = _frac(own, a0_own)
        d = dict(r)
        d.update(own_win=own, leak_b=leak_b, leak_c=leak_c, retention=retention,
                 rel={c: _rel(base_ppl, r["ppl"], c) for c in COLS})
        if r["i"] == 0:
            d.update(bars=None, b_pass=None, c_pass=None, verdict="anchor")
        else:
            bars = {
                "abs_leak":  leak_b <= BAR_ABS_LEAK,
                "rel_leak":  leak_b <= BAR_REL_LEAK * a0_leak_b,
                "retention": retention >= BAR_RETENTION,
            }
            b_pass = all(bars.values())
            # unseen mirror of bars 1-2: a B-pass with C-fail = author
            # memorization, flagged and counted as FAIL per spec
            c_pass = (leak_c <= BAR_ABS_LEAK and leak_c <= BAR_REL_LEAK * a0_leak_c)
            verdict = ("PASS" if (b_pass and c_pass)
                       else "MEMORIZES (B-pass, C-fail -> FAIL)" if b_pass
                       else "FAIL")
            d.update(bars=bars, b_pass=b_pass, c_pass=c_pass, verdict=verdict)
        out.append(d)
    passers = [d for d in out if d["verdict"] == "PASS"]
    best = min(passers, key=lambda d: d["leak_b"]) if passers else None
    return {"arms": out, "arm0_own": a0_own, "arm0_leak_b": a0_leak_b,
            "arm0_leak_c": a0_leak_c, "passers": [p["i"] for p in passers],
            "best": best["i"] if best else None}


def main_effects(arms):
    """Factorial main effects over arms 1-12 (mean leak_b / leak_c / retention)."""
    hinged = [a for a in arms if a["i"] > 0]
    def agg(group):
        if not group:
            return None
        return {k: sum(g[k] for g in group) / len(group)
                for k in ("leak_b", "leak_c", "retention")}
    by_lam = {f"{lam:g}": agg([a for a in hinged if a["lam"] == lam]) for lam in LAMBDAS}
    by_neg = {neg: agg([a for a in hinged if a["neg"] == neg]) for neg in NEG_TYPES}
    cells = {f"{lam:g}/{neg}": agg([a for a in hinged
                                    if a["lam"] == lam and a["neg"] == neg])
             for lam in LAMBDAS for neg in NEG_TYPES}
    return {"by_lambda": by_lam, "by_neg": by_neg, "cells": cells}


def write_report(out_root, smoke, meta):
    arm_files = [f for f in os.listdir(out_root)
                 if f.startswith("arm_") and f.endswith(".json")]
    arm_results = sorted((json.load(open(os.path.join(out_root, f))) for f in arm_files),
                         key=lambda r: r["i"])
    base_rows = json.load(open(os.path.join(out_root, "base_rows.json")))
    az = analyze(base_rows, arm_results)
    fx = main_effects(az["arms"])

    results = {"meta": meta, "base_rows": base_rows, "analysis": az,
               "main_effects": fx, "pass_bars": {
                   "abs_leak": BAR_ABS_LEAK, "rel_leak_x_arm0": BAR_REL_LEAK,
                   "retention_of_arm0": BAR_RETENTION}}
    results_path = os.path.join(out_root if smoke else HERE, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=1)

    # retention-vs-leak tradeoff (CSV always; PNG best-effort)
    csv_path = os.path.join(out_root, "retention_vs_leak.csv")
    with open(csv_path, "w") as f:
        f.write("arm,lambda,neg,own_win_pct,retention,leak_b,leak_c,verdict\n")
        for a in az["arms"]:
            f.write(f"{a['i']},{a['lam']:g},{a['neg']},{a['own_win']:.2f},"
                    f"{a['retention']:.3f},{a['leak_b']:.3f},{a['leak_c']:.3f},"
                    f"{a['verdict']}\n")
    png_path = os.path.join(out_root, "retention_vs_leak.png")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 5))
        colors = {"none": "k", "self": "tab:blue", "author": "tab:orange",
                  "both": "tab:green"}
        for a in az["arms"]:
            ax.scatter(a["retention"], a["leak_b"], s=30 + 25 * a["lam"],
                       c=colors[a["neg"]],
                       marker="*" if a["verdict"] == "PASS" else "o")
            ax.annotate(f"{a['i']}", (a["retention"], a["leak_b"]), fontsize=7)
        ax.axhline(BAR_ABS_LEAK, ls="--", c="r", lw=0.8, label="abs leak bar 0.10")
        ax.axvline(BAR_RETENTION, ls="--", c="g", lw=0.8, label="retention bar 0.70")
        ax.set_xlabel("own-win retention vs arm 0")
        ax.set_ylabel("seen-leak fraction (col B)")
        ax.set_title("sweep_ccat50: retention vs leak (size~lambda, color=neg type)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(png_path, dpi=120)
    except Exception as e:
        print(f"(PNG skipped: {e})", flush=True)
        png_path = None

    # ---- markdown ----
    if az["best"] is not None:
        b = next(a for a in az["arms"] if a["i"] == az["best"])
        verdict = f"**CURE PASS (arm {b['i']}: lambda={b['lam']:g}, neg={b['neg']})**"
    else:
        verdict = "**CURE FAIL — structural** (no arm met all bars)"
    if smoke:
        verdict = f"SMOKE RUN — plumbing only, numbers meaningless. {verdict}"

    L = []
    L.append("# Sweep: contrastive lambda x negative-type factorial on CCAT50\n")
    L.append(f"{verdict}\n")
    L.append(f"_Generated {datetime.now(timezone.utc).isoformat()} | "
             f"arms completed: {len(arm_results)}/13 | meta: {json.dumps(meta)}_\n")
    L.append("## Verdict table (chat-masked held-out ppl; rel% vs base)\n")
    hdr = "| arm | lambda | neg | " + " | ".join(COLS) + " | own-win | retention | leak B | leak C | verdict |"
    L.append(hdr)
    L.append("|" + "---|" * (len(COLS) + 8))
    base_ppl = base_rows["base"]
    L.append("| base | — | — | " + " | ".join(f"{base_ppl[c]:.2f}" for c in COLS)
             + " | — | — | — | — | — |")
    sp = base_rows["style_prompt"]
    L.append("| style-prompt | — | — | "
             + " | ".join(f"{sp[c]:.2f} ({_rel(base_ppl, sp, c):+.1f}%)" for c in COLS)
             + " | — | — | — | — | — |")
    for a in az["arms"]:
        cells = " | ".join(f"{a['ppl'][c]:.2f} ({a['rel'][c]:+.1f}%)" for c in COLS)
        L.append(f"| {a['i']} | {a['lam']:g} | {a['neg']} | {cells} | "
                 f"{a['own_win']:+.1f}% | {a['retention']:.2f} | {a['leak_b']:.3f} | "
                 f"{a['leak_c']:.3f} | {a['verdict']} |")
    L.append(f"\nPass bars: leak B <= {BAR_ABS_LEAK} abs, leak B <= "
             f"{BAR_REL_LEAK} x arm-0 ({az['arm0_leak_b']:.3f}), retention >= "
             f"{BAR_RETENTION}. Unseen (C) mirror of bars 1-2 flags author "
             f"memorization (counts as FAIL).\n")
    L.append("## Factorial analysis (means over hinged arms)\n")
    for title, table in (("Main effect of lambda", fx["by_lambda"]),
                         ("Main effect of negative type", fx["by_neg"]),
                         ("Interaction (lambda x neg cells)", fx["cells"])):
        L.append(f"### {title}\n")
        L.append("| level | leak B | leak C | retention |")
        L.append("|---|---|---|---|")
        for k, v in table.items():
            if v:
                L.append(f"| {k} | {v['leak_b']:.3f} | {v['leak_c']:.3f} | "
                         f"{v['retention']:.2f} |")
        L.append("")
    L.append("## Tradeoff curve\n")
    L.append(f"- CSV: `{os.path.relpath(csv_path, HERE)}`")
    if png_path:
        L.append(f"- PNG: `{os.path.relpath(png_path, HERE)}`")
    L.append(f"\nAll raw numbers: `{os.path.relpath(results_path, HERE)}`\n")

    md_path = os.path.join(out_root, "SWEEP_CCAT50.md") if smoke \
        else os.path.join(HERE, "SWEEP_CCAT50.md")
    with open(md_path, "w") as f:
        f.write("\n".join(L))
    print(f"report -> {md_path}\nresults -> {results_path}", flush=True)
    return az


# ---- main -------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="tiny random base + fake data, 2 arms x 2 epochs, CPU")
    ap.add_argument("--report-only", action="store_true",
                    help="rebuild SWEEP_CCAT50.md/results.json from arm JSONs")
    args = ap.parse_args()
    smoke = args.smoke

    out_root = os.path.join(HERE, "results", "smoke" if smoke else "sweep_ccat50")
    status_dir = os.path.join(out_root, "status") if smoke else os.path.join(HERE, "status")
    ckpt_dir = os.path.join(HERE, "ghosts", "sweep_smoke" if smoke else "sweep")
    os.makedirs(out_root, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    meta = {"smoke": smoke, "model": "tiny-random-llama" if smoke else MODEL_NAME,
            "device": DEVICE, "alpha_serve": ALPHA_SERVE, "seed": SEED}
    if args.report_only:
        write_report(out_root, smoke, meta)
        return

    torch.manual_seed(SEED)
    keep_awake()
    start_heartbeat(status_dir)
    t_start = time.time()

    arm_list = [ARMS[0], ARMS[12]] if smoke else ARMS
    max_epochs = 2 if smoke else MAX_EPOCHS
    patience = 1 if smoke else PATIENCE
    d_ghost = 16 if smoke else D_GHOST

    # ---- data + ONE base load for the whole run ----
    STATUS["phase"] = "load"
    if smoke:
        rng = random.Random(SEED)
        tok = smoke_tokenizer()
        base = smoke_base(len(tok)).to(DEVICE)
        train_docs = smoke_sentences(rng, "target", 10)
        test_docs = smoke_sentences(rng, "target", 6)
        seen_docs = smoke_sentences(rng, "seen", 6)
        unseen_docs = smoke_sentences(rng, "unseen", 6)
        d3_lines = smoke_sentences(rng, "wiki", 6)
        # self "paraphrases": shuffled doc words; author negs: more seen-flavor text
        self_neg_docs = [" ".join(sorted(d.split(), key=lambda w: rng.random()))
                         for d in train_docs]
        author_neg_docs = smoke_sentences(rng, "seen", len(train_docs))
        d3_src = "smoke-fake"
    else:
        download_ccat50()
        print(f"loading {MODEL_NAME} on {DEVICE} ...", flush=True)
        if DEVICE == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(MODEL_NAME)
        try:
            base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype="auto").to(DEVICE)
        except TypeError:
            base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype="auto").to(DEVICE)
        authors = list_authors()
        assert authors[0] == TARGET_AUTHOR, f"expected {TARGET_AUTHOR}, got {authors[0]}"
        seen_authors = authors[1:1 + N_CONTRAST]
        unseen_authors = authors[1 + N_CONTRAST:1 + 2 * N_CONTRAST]
        print(f"target: {TARGET_AUTHOR}\nseen:   {', '.join(seen_authors)}\n"
              f"unseen: {', '.join(unseen_authors)}", flush=True)
        train_docs = read_docs(TARGET_AUTHOR, "C50train", tok)
        test_docs = read_docs(TARGET_AUTHOR, "C50test", tok)
        seen_docs = [d for a in seen_authors
                     for d in read_docs(a, "C50test", tok, limit=N_CONTRAST_DOCS)]
        unseen_docs = [d for a in unseen_authors
                       for d in read_docs(a, "C50test", tok, limit=N_CONTRAST_DOCS)]
        d3_lines, d3_src = load_domain3()
        d3_lines = d3_lines[:80]
        author_neg_docs = sample_author_negatives(tok, seen_authors)

    model = GhostModel(base, d_ghost=d_ghost).to(DEVICE)
    assert not any(p.requires_grad for p in model.base.parameters()), "base must be frozen"
    im_end_id = _im_end_id(tok)
    fp0 = base_fingerprint(model)
    b_n, g_n = param_counts(model)
    print(f"ghost/base = {100 * g_n / b_n:.3f}% | domain3: {d3_src}", flush=True)

    # ---- negatives (self regenerated via the exact existing recipe) ----
    STATUS["phase"] = "negatives"
    if smoke:
        pair_by_doc = dict(zip(train_docs, self_neg_docs))
    else:
        pairs = make_pairs(base, tok, train_docs)
        pair_by_doc = {p["doc"]: p["para"] for p in pairs}
    doc_ex = chat_ex_aligned(tok, train_docs, im_end_id)
    self_ex = chat_ex_aligned(tok, [pair_by_doc.get(d) for d in train_docs], im_end_id)
    author_ex = chat_ex_aligned(tok, author_neg_docs, im_end_id)
    assert len(author_ex) == len(doc_ex), "author negatives must align by index"
    print("precomputing base CE on negatives ...", flush=True)
    negs = {"self": (self_ex, precompute_base_ce(model, self_ex)),
            "author": (author_ex, precompute_base_ce(model, author_ex))}
    n_live = {k: sum(1 for e in v[0] if e) for k, v in negs.items()}
    print(f"train docs {sum(1 for e in doc_ex if e)} | live negatives {n_live}", flush=True)

    # ---- eval columns (identical token spans across all rows) ----
    STATUS["phase"] = "eval-setup"
    style_samples = [d[:600] for d in train_docs[:3]]
    style_prompt = ("Here are samples of the author's writing:\n\n"
                    + "\n\n".join(style_samples) + "\n\nWrite in this author's style.")
    cols = build_cols(tok, im_end_id, {
        "A_target": test_docs, "B_seen": seen_docs,
        "C_unseen": unseen_docs, "D_wikitext": d3_lines}, style_prompt)
    for c, v in cols.items():
        print(f"  {c}: {len(v['neutral'])} examples", flush=True)

    # base + style-prompt rows (once, cached for resume)
    base_rows_path = os.path.join(out_root, "base_rows.json")
    if os.path.isfile(base_rows_path):
        print("base rows cached", flush=True)
    else:
        STATUS["phase"] = "base-rows"
        base_rows = {"base": eval_row(model, cols, "neutral", use_ghost=False),
                     "style_prompt": eval_row(model, cols, "style", use_ghost=False),
                     "n_examples": {c: len(v["neutral"]) for c, v in cols.items()}}
        with open(base_rows_path, "w") as f:
            json.dump(base_rows, f, indent=1)

    # ---- arms (sequential; resumable; cost-guarded) ----
    fresh_times = []
    for arm in arm_list:
        name = arm_name(arm)
        arm_json = os.path.join(out_root, f"{name}.json")
        if os.path.isfile(arm_json):
            print(f"\n{name}: done (skipping)", flush=True)
            continue
        print(f"\n=== {name} (lambda={arm['lam']:g}, neg={arm['neg']}) ===", flush=True)
        STATUS.update(phase="train", arm=name, epoch=None, step=None)
        t_arm = time.time()
        ckpt = os.path.join(ckpt_dir, f"{name}.pt")
        info = train_arm_sweep(model, doc_ex, negs, ckpt, arm["lam"], arm["neg"],
                               d_ghost, max_epochs=max_epochs, patience=patience)
        STATUS["phase"] = "eval"
        model.ghost.alpha.fill_(ALPHA_SERVE)
        ppl = eval_row(model, cols, "neutral", use_ghost=True)
        fp_delta = base_fingerprint(model) - fp0
        STATUS["phase"] = "hub-push"
        hub = push_ckpt_to_hub(ckpt, arm, smoke)
        seconds = time.time() - t_arm
        fresh_times.append(seconds)
        with open(arm_json, "w") as f:
            json.dump({**arm, "ppl": ppl, "train": info, "seconds": round(seconds, 1),
                       "fp_delta": fp_delta, "hub_push": hub, "ckpt": ckpt}, f, indent=1)
        print(f"{name}: {seconds:.0f}s | hub: {hub} | fp_delta {fp_delta:.2e}", flush=True)

        # cost guard: once arm index >= 3 done, projection from this run's pace
        if not smoke and arm["i"] >= COST_GUARD_ARM and fresh_times:
            done_n = sum(1 for a in ARMS
                         if os.path.isfile(os.path.join(out_root, f"{arm_name(a)}.json")))
            per_arm = sum(fresh_times) / len(fresh_times)
            projection = per_arm * len(ARMS)
            if projection > COST_GUARD_X * BUDGET_SECONDS:
                msg = (f"cost guard tripped after {name}: {per_arm:.0f}s/arm -> "
                       f"projected {projection/3600:.1f}h > "
                       f"{COST_GUARD_X}x budget {BUDGET_SECONDS/3600:.1f}h "
                       f"({done_n}/13 arms committed)")
                print(f"\nABORT: {msg}", flush=True)
                with open(os.path.join(status_dir, "ABORT.json"), "w") as f:
                    json.dump({"reason": msg,
                               "ts": datetime.now(timezone.utc).isoformat()}, f)
                write_report(out_root, smoke, meta)
                raise SystemExit(3)

    # ---- report ----
    STATUS["phase"] = "report"
    write_report(out_root, smoke, meta)
    STATUS["phase"] = "done"
    print(f"\ntotal wall time {time.time() - t_start:.0f}s", flush=True)
    _STOP.set()


if __name__ == "__main__":
    main()
