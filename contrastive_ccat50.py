"""
contrastive_ccat50.py - contrastive author-isolation on CCAT50 (two-arm experiment).

Question: the generic diagnostic showed the voice ghost LEAKS (0.32 of its own-domain
win transfers to unrelated wikitext). Hypothesis: shortcut learning - plain next-token
CE buys generic prose fluency, not author style. Test on a public, controlled corpus:
CCAT50 (Reuter_50_50, UCI id 217) - 50 authors, same CCAT register by design, so the
49 non-target authors are a sharp same-topic control.

  Arm 1 (plain):       exact v03 recipe on one target author. Does the leak replicate?
  Arm 2 (contrastive): same + hinge penalty when the ghost improves NEUTRAL PARAPHRASES
                       of the same docs:  L = CE_g(doc) + LAMBDA * max(0, CE_b(n) - CE_g(n)).
                       Does the leak collapse while the author win survives?

Falsifiable: can return NOT-REPLICATED (no leak here) or FAIL (cure kills the win).

Architecture untouched - GhostModel/GhostStream/compressor imported from ghost.py.
Masking byte-identical to Stage 1/2 (doc as assistant turn after the fixed neutral
prompt; assistant tokens scored, -100 elsewhere). Resumable checkpoints (Modern-Standby
laptop). Data under data/ccat50/ is gitignored; reproducibility via download_ccat50().
"""
import json
import math
import os
import random
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ghost import (
    GhostModel, MODEL_NAME, DEVICE, SEED, MAX_LEN,
    LR, WEIGHT_DECAY, MAX_EPOCHS, PATIENCE, VAL_FRAC,
    base_fingerprint, param_counts,
)
from chat_test import _flat_ids, _im_end_id, build_chat_example, reinit_ghost, NEUTRAL_PROMPT
from bank import per_example_losses, agg_ppl, keep_awake
from generic_test import load_domain3

HERE          = os.path.dirname(os.path.abspath(__file__))
CCAT_DIR      = os.path.join(HERE, "data", "ccat50")
PAIRS_PATH    = os.path.join(CCAT_DIR, "pairs_target.jsonl")
PLAIN_CKPT    = os.path.join(HERE, "ghosts", "ghost_ccat_plain.pt")
CONTRAST_CKPT = os.path.join(HERE, "ghosts", "ghost_ccat_contrastive.pt")

TARGET_AUTHOR     = "AaronPressman"   # alphabetically first; overridable
N_CONTRAST        = 10                # next N authors alphabetically
N_CONTRAST_DOCS   = 20                # C50test docs per contrast author
DOC_TOKENS        = 220               # pre-truncate docs so scaffold+doc fits MAX_LEN=256
LAMBDA            = 1.0               # hinge weight (single value, no sweep)
ALPHA_SERVE       = 0.5               # established serving point
PARA_INSTRUCTION  = ("Rewrite the following text in plain, neutral English, "
                     "preserving all meaning:")


# ---- Phase 0: data ----------------------------------------------------------
def download_ccat50(dest=CCAT_DIR):
    """Fetch CCAT50. ucimlrepo can't serve id=217 (text dataset), so pull the UCI
    static zip. Idempotent: skips if already extracted."""
    if os.path.isdir(os.path.join(dest, "C50train")):
        return
    os.makedirs(dest, exist_ok=True)
    zip_path = os.path.join(dest, "c50.zip")
    if not os.path.isfile(zip_path):
        import urllib.request
        url = "https://archive.ics.uci.edu/static/public/217/reuter+50+50.zip"
        print(f"downloading {url} ...", flush=True)
        urllib.request.urlretrieve(url, zip_path)
    import zipfile
    zipfile.ZipFile(zip_path).extractall(dest)


def list_authors(split="C50train"):
    return sorted(os.listdir(os.path.join(CCAT_DIR, split)))


def read_docs(author, split, tok, limit=None):
    """One doc = one example, pre-truncated to DOC_TOKENS tokens (text-level) so the
    full chat-formatted example fits MAX_LEN and the SAME doc text is scored in every
    eval row (scaffold length differs per row but scaffold is masked)."""
    d = os.path.join(CCAT_DIR, split, author)
    docs = []
    for fn in sorted(os.listdir(d))[:limit]:
        raw = open(os.path.join(d, fn), encoding="utf-8", errors="replace").read().strip()
        raw = " ".join(raw.split())
        ids = tok(raw, add_special_tokens=False).input_ids[:DOC_TOKENS]
        docs.append(tok.decode(ids))
    return docs


# ---- masking: same scheme as chat_test, but with a configurable user prompt --
def build_example_prompted(tok, user_prompt, turn, im_end_id, max_len):
    """[user: user_prompt, assistant: turn] -> (ids, labels) with ONLY the assistant
    content scored. Same empty-assistant LCP boundary as chat_test.build_chat_example."""
    full = _flat_ids(tok, [{"role": "user", "content": user_prompt},
                           {"role": "assistant", "content": turn}], add_gen=False)
    empt = _flat_ids(tok, [{"role": "user", "content": user_prompt},
                           {"role": "assistant", "content": ""}], add_gen=False)
    boundary = 0
    for x, y in zip(empt, full):
        if x == y:
            boundary += 1
        else:
            break
    end = len(full)
    for i in range(boundary, len(full)):
        if full[i] == im_end_id:
            end = i
            break
    full = full[:max_len]
    end = min(end, len(full))
    if boundary >= end:
        return None
    labels = [-100] * len(full)
    for i in range(boundary, end):
        labels[i] = full[i]
    return (torch.tensor([full], dtype=torch.long),
            torch.tensor([labels], dtype=torch.long))


def neutral_examples(tok, docs, im_end_id):
    out = []
    for d in docs:
        ex = build_chat_example(tok, d, im_end_id, max_len=MAX_LEN)
        if ex is not None:
            out.append(ex)
    return out


# ---- Phase 0b: neutral paraphrases with the frozen base ----------------------
def strip_think(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


@torch.no_grad()
def paraphrase(base, tok, doc, temperature, seed):
    msgs = [{"role": "user", "content": f"{PARA_INSTRUCTION}\n\n{doc}"}]
    try:
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)
    except TypeError:
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt").input_ids.to(DEVICE)
    n_doc = len(tok(doc, add_special_tokens=False).input_ids)
    torch.manual_seed(seed)
    out = base.generate(ids, do_sample=True, temperature=temperature, top_p=0.9,
                        max_new_tokens=n_doc + 32, pad_token_id=tok.eos_token_id)
    text = tok.decode(out[0, ids.size(1):], skip_special_tokens=True)
    return strip_think(text)


def make_pairs(base, tok, docs):
    """One neutral paraphrase per train doc; keep if non-empty and token-ratio in
    [0.5, 2.0]; one retry at temp 0.7; else drop. Persisted to PAIRS_PATH (resume)."""
    if os.path.isfile(PAIRS_PATH):
        pairs = [json.loads(l) for l in open(PAIRS_PATH, encoding="utf-8")]
        print(f"pairs loaded from cache: {len(pairs)}", flush=True)
        return pairs
    pairs, dropped = [], 0
    for i, doc in enumerate(docs):
        n_doc = len(tok(doc, add_special_tokens=False).input_ids)
        kept = None
        for attempt, temp in enumerate((0.3, 0.7)):
            p = paraphrase(base, tok, doc, temp, seed=SEED * 1000 + i * 10 + attempt)
            n_p = len(tok(p, add_special_tokens=False).input_ids)
            if p and 0.5 <= n_p / max(n_doc, 1) <= 2.0:
                kept = p
                break
        if kept is None:
            dropped += 1
        else:
            pairs.append({"doc": doc, "para": kept})
        if (i + 1) % 10 == 0:
            print(f"  paraphrased {i+1}/{len(docs)} (dropped {dropped})", flush=True)
    with open(PAIRS_PATH, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"pairs: {len(pairs)} kept / {dropped} dropped -> {PAIRS_PATH}", flush=True)
    return pairs


# ---- training (resumable, chat-masked; contrastive optional) -----------------
def split_idx(n, val_frac=VAL_FRAC, seed=SEED):
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    n_val = max(1, int(round(n * val_frac)))
    return idx[n_val:], idx[:n_val]


def masked_ce(model, ex, use_ghost):
    ids, labels = ex
    _, loss = model(ids.to(DEVICE), labels=labels.to(DEVICE), use_ghost=use_ghost)
    return loss


def train_arm(model, tok, doc_ex, pair_ex, base_ce_pairs, ckpt, lam=0.0,
              max_epochs=MAX_EPOCHS, lr=LR, weight_decay=WEIGHT_DECAY,
              patience=PATIENCE, seed=SEED):
    """Resumable trainer. lam=0 -> Arm 1 (plain CE). lam>0 -> Arm 2: per doc i,
    L_i = CE_g(doc_i) + lam * max(0, CE_b(n_i) - CE_g(n_i)). Early-stop on the val
    value of the SAME objective being optimized. Keeps best-val weights."""
    done_flag = ckpt + ".done"
    tr_idx, va_idx = split_idx(len(doc_ex))

    @torch.no_grad()
    def val_loss():
        model.eval()
        tot = 0.0
        for i in va_idx:
            tot += masked_ce(model, doc_ex[i], use_ghost=True).item()
            if lam > 0 and pair_ex[i] is not None:
                g = masked_ce(model, pair_ex[i], use_ghost=True).item()
                tot += lam * max(0.0, base_ce_pairs[i] - g)
        return tot / max(len(va_idx), 1)

    if os.path.isfile(done_flag) and os.path.isfile(ckpt):
        model.ghost.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        model.ghost.alpha.fill_(1.0)
        print(f"arm already converged: {os.path.basename(ckpt)}", flush=True)
        return
    if os.path.isfile(ckpt):
        model.ghost.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        model.ghost.alpha.fill_(1.0)
        best = val_loss()
        print(f"resuming from checkpoint (val {best:.4f})", flush=True)
    else:
        reinit_ghost(model)
        best = float("inf")
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                            lr=lr, weight_decay=weight_decay)
    rng = random.Random(seed)
    bad = 0
    order = tr_idx[:]
    for epoch in range(max_epochs):
        model.train()
        rng.shuffle(order)
        running, nb = 0.0, 0
        for i in order:
            loss = masked_ce(model, doc_ex[i], use_ghost=True)
            if lam > 0 and pair_ex[i] is not None:
                g = masked_ce(model, pair_ex[i], use_ghost=True)
                hinge = torch.clamp(torch.tensor(base_ce_pairs[i], device=DEVICE) - g, min=0.0)
                loss = loss + lam * hinge
            opt.zero_grad(); loss.backward(); opt.step()
            running += loss.item(); nb += 1
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


# ---- main --------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(SEED)
    keep_awake()

    download_ccat50()
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
    b_n, g_n = param_counts(model)
    print(f"ghost/base = {100*g_n/b_n:.3f}%", flush=True)

    authors = list_authors()
    assert authors[0] == TARGET_AUTHOR, f"expected {TARGET_AUTHOR} first, got {authors[0]}"
    contrast_authors = [a for a in authors if a != TARGET_AUTHOR][:N_CONTRAST]
    print(f"target: {TARGET_AUTHOR} | contrast: {', '.join(contrast_authors)}", flush=True)

    train_docs = read_docs(TARGET_AUTHOR, "C50train", tok)            # 50
    test_docs  = read_docs(TARGET_AUTHOR, "C50test", tok)             # 50
    contrast_docs = [d for a in contrast_authors
                     for d in read_docs(a, "C50test", tok, limit=N_CONTRAST_DOCS)]  # 200
    d3_lines, d3_src = load_domain3()
    print(f"docs: target train {len(train_docs)} / test {len(test_docs)} | "
          f"contrast {len(contrast_docs)} | domain3 {d3_src}", flush=True)

    # Phase 0b: neutral paraphrase pairs (frozen base; cached)
    print("\nPhase 0b: neutral paraphrases ...", flush=True)
    pairs = make_pairs(base, tok, train_docs)
    pair_by_doc = {p["doc"]: p["para"] for p in pairs}

    # build chat-masked examples (train + pair aligned by index)
    doc_ex  = neutral_examples(tok, train_docs, im_end_id)
    pair_ex = []
    for d in train_docs:
        p = pair_by_doc.get(d)
        pair_ex.append(build_chat_example(tok, p, im_end_id, max_len=MAX_LEN) if p else None)
    n_live_pairs = sum(1 for e in pair_ex if e is not None)
    print(f"train examples: {len(doc_ex)} docs, {n_live_pairs} live pairs", flush=True)

    # worked masking example (counts only - no article text committed anywhere)
    ids0, lab0 = doc_ex[0]
    n_scored = int((lab0 != -100).sum().item())
    print(f"worked masking example: doc 0 -> {ids0.size(1)} tokens total, "
          f"{n_scored} scored (assistant span), {ids0.size(1)-n_scored} masked scaffold", flush=True)

    # precompute CE_base on paraphrases (frozen base, once)
    print("precomputing base CE on paraphrases ...", flush=True)
    base_ce_pairs = []
    with torch.no_grad():
        for e in pair_ex:
            base_ce_pairs.append(masked_ce(model, e, use_ghost=False).item() if e else None)

    # ---- Arm 1: plain ----
    print("\nArm 1: plain ghost (exact v03 recipe) ...", flush=True)
    train_arm(model, tok, doc_ex, pair_ex, base_ce_pairs, PLAIN_CKPT, lam=0.0)
    fp_plain = base_fingerprint(model)

    # ---- Arm 2: contrastive ----
    print(f"\nArm 2: contrastive ghost (LAMBDA={LAMBDA}) ...", flush=True)
    train_arm(model, tok, doc_ex, pair_ex, base_ce_pairs, CONTRAST_CKPT, lam=LAMBDA)
    fp_contrast = base_fingerprint(model)

    # ---- Phase 3: eval grid ----
    print("\nPhase 3: eval grid ...", flush=True)
    style_samples = [d[:600] for d in train_docs[:3]]
    style_prompt = ("Here are samples of the author's writing:\n\n"
                    + "\n\n".join(style_samples)
                    + "\n\nWrite in this author's style.")

    def col_examples(docs, prompt=NEUTRAL_PROMPT, max_len=MAX_LEN):
        out = []
        for d in docs:
            ex = build_example_prompted(tok, prompt, d, im_end_id, max_len)
            if ex is not None:
                out.append(ex)
        return out

    cols = {
        "A_target":   {"neutral": col_examples(test_docs),
                       "style":   col_examples(test_docs, style_prompt, max_len=2048)},
        "B_contrast": {"neutral": col_examples(contrast_docs),
                       "style":   col_examples(contrast_docs, style_prompt, max_len=2048)},
        "C_wikitext": {"neutral": col_examples(d3_lines[:80]),
                       "style":   col_examples(d3_lines[:80], style_prompt, max_len=2048)},
    }
    for c, v in cols.items():
        print(f"  {c}: {len(v['neutral'])} examples", flush=True)

    grid = {}
    # row 1: base
    grid["base"] = {c: agg_ppl(per_example_losses(model, v["neutral"], use_ghost=False))
                    for c, v in cols.items()}
    # row 2: base + style prompt (no ghost)
    grid["style_prompt"] = {c: agg_ppl(per_example_losses(model, v["style"], use_ghost=False))
                            for c, v in cols.items()}
    # rows 3-4: ghosts at serving alpha
    for row, ckpt in (("ghost_plain", PLAIN_CKPT), ("ghost_contrastive", CONTRAST_CKPT)):
        model.ghost.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        model.ghost.alpha.fill_(ALPHA_SERVE)
        grid[row] = {c: agg_ppl(per_example_losses(model, v["neutral"], use_ghost=True))
                     for c, v in cols.items()}
    fp_end = base_fingerprint(model)

    # ---- report ----
    def rel(row, col):
        return (grid["base"][col] - grid[row][col]) / grid["base"][col] * 100.0

    print("\n" + "#" * 78)
    print("RESULTS  (chat-masked held-out ppl; rel% = improvement over base)")
    print("#" * 78)
    print(f"{'row':<20} {'A_target':>16} {'B_contrast':>16} {'C_wikitext':>16}")
    for row in ("base", "style_prompt", "ghost_plain", "ghost_contrastive"):
        cells = []
        for c in ("A_target", "B_contrast", "C_wikitext"):
            r = f" ({rel(row, c):+.1f}%)" if row != "base" else ""
            cells.append(f"{grid[row][c]:>8.2f}{r:<8}")
        print(f"{row:<20} {''.join(cells)}")

    plain_own, plain_leak_b = rel("ghost_plain", "A_target"), rel("ghost_plain", "B_contrast")
    cont_own,  cont_leak_b  = rel("ghost_contrastive", "A_target"), rel("ghost_contrastive", "B_contrast")
    plain_frac = plain_leak_b / plain_own if plain_own > 0 else float("nan")
    cont_frac  = cont_leak_b / cont_own if cont_own > 0 else float("nan")
    print(f"\nheadlines:")
    print(f"  ghost_plain:       own-win {plain_own:+.1f}%  leak(B) {plain_leak_b:+.1f}%  leak fraction {plain_frac:.3f}")
    print(f"  ghost_contrastive: own-win {cont_own:+.1f}%  leak(B) {cont_leak_b:+.1f}%  leak fraction {cont_frac:.3f}")
    print(f"  wikitext leak: plain {rel('ghost_plain','C_wikitext'):+.1f}%  "
          f"contrastive {rel('ghost_contrastive','C_wikitext'):+.1f}%")

    print(f"\nPROBE 2 (base frozen): fp deltas plain={fp_plain-fp0:.2e} "
          f"contrastive={fp_contrast-fp0:.2e} end={fp_end-fp0:.2e} (must be 0)")

    replicated = plain_frac >= 0.15
    not_repl   = plain_frac < 0.10
    cure_pass = (cont_own >= 0.6 * plain_own
                 and cont_frac <= 0.5 * plain_frac
                 and cont_frac <= 0.10)
    style_beats = (grid["style_prompt"]["A_target"] <= grid["ghost_plain"]["A_target"]
                   or grid["style_prompt"]["A_target"] <= grid["ghost_contrastive"]["A_target"])

    print(f"\nLEAK: {'REPLICATED' if replicated else ('NOT-REPLICATED' if not_repl else 'BORDERLINE')}"
          f"  (plain leak fraction {plain_frac:.3f}; replicate>=0.15, not-replicated<0.10)")
    print(f"CURE: {'PASS' if cure_pass else 'FAIL'}"
          f"  (own-win retention {cont_own/plain_own if plain_own else float('nan'):.2f} need >=0.6; "
          f"frac ratio {cont_frac/plain_frac if plain_frac else float('nan'):.2f} need <=0.5; "
          f"abs frac {cont_frac:.3f} need <=0.10)")
    if style_beats:
        print("FLAG: style-prompt baseline matches/beats a ghost on A_target - "
              "the cheap baseline wins; do not bury this.")
    print(f"\nVERDICT: leak {'REPLICATED' if replicated else ('NOT-REPLICATED' if not_repl else 'BORDERLINE')} "
          f"| cure {'PASS' if cure_pass else 'FAIL'}")
