"""
chat_test.py - the "is it true" test: does the ghost's perplexity win survive
in the CHAT-MODE serving path (Qwen's <|im_start|> chat template), not just raw
continuation?

The ghost architecture is imported verbatim from ghost.py (frozen base +
GhostStream + compressor + `lm_head(repr + alpha*compressed)` readout). NOTHING
about the architecture changes here. This module only adds:
  - a chat-format corpus loader (each turn in the assistant slot after a fixed
    neutral user prompt),
  - assistant-ONLY label masking (score his words, -100 over all chat scaffolding),
  - a masked train loop + masked perplexity eval (same knobs as ghost.train),
  - a chat-mode alpha sweep and a chat-mode generation A/B.

ghost.py's raw path is left intact so raw-mode is the reference.

Falsifiable: PROBE 1 (chat) can come back FAIL. If it does, we say so.
"""
import os
import sys
import math
import random

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# import the architecture + helpers from ghost.py (do NOT fork it)
from ghost import (
    GhostModel, GhostStream,
    MODEL_NAME, DEVICE, D_GHOST, ALPHA_INIT, SEED,
    VAL_FRAC, LR, WEIGHT_DECAY, MAX_EPOCHS, PATIENCE, MAX_LEN,
    load_corpus, split_corpus, mean_loss, base_fingerprint, param_counts,
)

NEUTRAL_PROMPT = "Share your thoughts."     # one FIXED user prompt for every example, train == eval
RAW_CKPT  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ghosts", "ghost_voice_02_compressor.pt")
CHAT_CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ghosts", "ghost_voice_03_chat.pt")
ALPHA_GRID = [0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0]
# generation A/B prompts (generic, NOT from the corpus)
AB_PROMPTS = [
    "Write a short message asking a professor for a deadline extension.",
    "Reflect briefly on what you worked on this week.",
    "Explain, casually, your idea for giving an AI persistent memory.",
]


def _im_end_id(tok):
    i = tok.convert_tokens_to_ids("<|im_end|>")
    return i if isinstance(i, int) and i >= 0 else tok.eos_token_id


def _stop_ids(tok):
    s = set()
    if tok.eos_token_id is not None:
        s.add(tok.eos_token_id)
    e = tok.convert_tokens_to_ids("<|im_end|>")
    if isinstance(e, int) and e >= 0:
        s.add(e)
    return s


def _flat_ids(tok, msgs, add_gen):
    """Flat python list of token ids (transformers 5.x returns a dict otherwise)."""
    enc = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=add_gen,
                                  return_dict=True, return_tensors="pt")
    return enc["input_ids"][0].tolist()


def build_chat_example(tok, turn, im_end_id, max_len=MAX_LEN):
    """
    Format one turn as [user: NEUTRAL_PROMPT, assistant: turn] through the chat
    template, and return (ids, labels) where labels == -100 everywhere EXCEPT the
    assistant-content span (his words). Boundary = longest common prefix between
    the full render and the (user + generation-prompt) render -> robust to the
    closed empty <think></think> block Qwen3 injects (it stays masked as
    scaffolding). Trailing <|im_end|> is masked too. Returns None if nothing
    scoreable remains.
    """
    full = _flat_ids(tok, [{"role": "user", "content": NEUTRAL_PROMPT},
                           {"role": "assistant", "content": turn}], add_gen=False)
    # empty-assistant render shares user + assistant header + the injected
    # <think></think> block EXACTLY; LCP(empty, full) lands at his content start,
    # so all scaffolding (including the think block) is masked.
    empt = _flat_ids(tok, [{"role": "user", "content": NEUTRAL_PROMPT},
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


def sanity_print(tok, turn, im_end_id):
    """Print ONE fully-formatted example + which tokens carry loss. Eyeball gate."""
    ex = build_chat_example(tok, turn, im_end_id)
    assert ex is not None, "example produced no scoreable tokens"
    ids, labels = ex
    ids_l, lab_l = ids[0].tolist(), labels[0].tolist()
    scored   = [t for t, l in zip(ids_l, lab_l) if l != -100]
    masked   = [t for t, l in zip(ids_l, lab_l) if l == -100]
    print("=" * 78)
    print("SANITY: one fully-formatted chat training example")
    print("=" * 78)
    print("--- full formatted (with special tokens) ---")
    print(tok.decode(ids_l, skip_special_tokens=False))
    print(f"\n--- SCORED span (labels != -100; should be ONLY his words) [{len(scored)} tok] ---")
    print(repr(tok.decode(scored, skip_special_tokens=False)))
    print(f"\n--- MASKED span (labels == -100; should be ALL scaffolding) [{len(masked)} tok] ---")
    print(repr(tok.decode(masked, skip_special_tokens=False)))
    print("=" * 78)


def build_examples(tok, turns, im_end_id):
    out = []
    for t in turns:
        ex = build_chat_example(tok, t, im_end_id)
        if ex is not None:
            out.append(ex)
    return out


@torch.no_grad()
def chat_perplexity(model, examples, use_ghost):
    """Token-weighted perplexity over assistant-content tokens only."""
    model.eval()
    total, ntok = 0.0, 0
    for ids, labels in examples:
        ids, labels = ids.to(DEVICE), labels.to(DEVICE)
        n = int((labels[:, 1:] != -100).sum().item())
        if n == 0:
            continue
        _, loss = model(ids, labels=labels, use_ghost=use_ghost)
        total += loss.item() * n
        ntok += n
    return math.exp(total / max(ntok, 1))


def chat_train(model, tok, train_ex, val_ex, max_epochs=MAX_EPOCHS, lr=LR,
               weight_decay=WEIGHT_DECAY, patience=PATIENCE, seed=SEED):
    """Masked next-token training of the ghost only; early-stop on masked val loss.
    Same knobs/structure as ghost.train, but labels are assistant-masked."""
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                            lr=lr, weight_decay=weight_decay)
    rng = random.Random(seed)
    best_val, best_epoch, best_state, bad = float("inf"), -1, None, 0
    order = train_ex[:]
    for epoch in range(max_epochs):
        model.train()
        rng.shuffle(order)
        running, nb = 0.0, 0
        for ids, labels in order:
            ids, labels = ids.to(DEVICE), labels.to(DEVICE)
            if int((labels[:, 1:] != -100).sum().item()) == 0:
                continue
            _, loss = model(ids, labels=labels)
            opt.zero_grad(); loss.backward(); opt.step()
            running += loss.item(); nb += 1
        vl = math.log(chat_perplexity(model, val_ex, use_ghost=True))
        print(f"epoch {epoch:3d} | train loss {running/max(nb,1):.4f} | "
              f"val loss {vl:.4f} | alpha {model.ghost.alpha.item():.3f}", flush=True)
        if vl < best_val - 1e-4:
            best_val, best_epoch, bad = vl, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.ghost.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                print(f"early stop at epoch {epoch} (best epoch {best_epoch}, "
                      f"val loss {best_val:.4f})", flush=True)
                break
    if best_state is not None:
        model.ghost.load_state_dict(best_state)
    return best_epoch, best_val


@torch.no_grad()
def chat_generate(model, tok, user_prompt, alpha, seed, stop_ids,
                  max_new=80, temp=0.8, top_p=0.95):
    model.ghost.alpha.fill_(float(alpha))
    text = tok.apply_chat_template([{"role": "user", "content": user_prompt}],
                                   tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").input_ids.to(DEVICE)
    start = ids.size(1)
    torch.manual_seed(seed)
    for _ in range(max_new):
        logits, _ = model(ids, use_ghost=True)
        logits = logits[:, -1, :].float() / temp
        sl, si = torch.sort(logits, descending=True)
        probs = F.softmax(sl, dim=-1)
        cum = probs.cumsum(dim=-1)
        mask = cum - probs > top_p
        sl = sl.masked_fill(mask, float("-inf"))
        nxt = si.gather(-1, torch.multinomial(F.softmax(sl, dim=-1), 1))
        ids = torch.cat([ids, nxt], dim=1)
        if nxt.item() in stop_ids:
            break
    return tok.decode(ids[0, start:], skip_special_tokens=True).strip()


def reinit_ghost(model):
    """Fresh random ghost (chat ghost trains from scratch, not from voice_02)."""
    d_model = model.base.config.hidden_size
    n_taps = model.base.config.num_hidden_layers + 1
    model.ghost = GhostStream(d_model, n_taps, D_GHOST).to(DEVICE, dtype=model.base.dtype)
    model.ghost.alpha.fill_(ALPHA_INIT)


if __name__ == "__main__":
    torch.manual_seed(SEED)

    # same corpus + same seeded 85/15 split as raw-mode (apples to apples)
    corpus = load_corpus()
    if len(corpus) < 50:
        print(f"[STOP] corpus has only {len(corpus)} turns; need a real corpus.")
        sys.exit(2)
    train_turns, val_turns = split_corpus(corpus)
    print(f"corpus: {len(corpus)} turns -> {len(train_turns)} train / {len(val_turns)} val "
          f"(SEED={SEED}, identical split to raw-mode)\n", flush=True)

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
    stop_ids = _stop_ids(tok)

    # PROBE 3 + base fingerprint (before any training)
    fp_before = base_fingerprint(model)
    b, g = param_counts(model)
    print(f"\nPROBE 3 (tiny ghost): base={b:,}  ghost={g:,}  ghost/base={100*g/b:.3f}%\n", flush=True)

    # SANITY: must look right before we trust any number
    sanity_print(tok, train_turns[0], im_end_id)

    # build masked chat examples
    train_ex = build_examples(tok, train_turns, im_end_id)
    val_ex   = build_examples(tok, val_turns, im_end_id)
    print(f"\nchat examples: {len(train_ex)} train / {len(val_ex)} val (assistant-masked)\n", flush=True)

    # ---- train a FRESH chat-mode ghost from scratch ----
    reinit_ghost(model)
    print(f"training chat-mode ghost (lr={LR}, wd={WEIGHT_DECAY}, patience={PATIENCE}, "
          f"train alpha={ALPHA_INIT}) ...", flush=True)
    best_epoch, best_val = chat_train(model, tok, train_ex, val_ex)
    os.makedirs(os.path.dirname(CHAT_CKPT), exist_ok=True)
    model.ghost.alpha.fill_(ALPHA_INIT)
    torch.save(model.ghost.state_dict(), CHAT_CKPT)
    print(f"saved {CHAT_CKPT}\n", flush=True)

    fp_after = base_fingerprint(model)

    # ---- PROBE 1 (chat-mode): the real test ----
    model.ghost.alpha.fill_(ALPHA_INIT)
    chat_base  = chat_perplexity(model, val_ex, use_ghost=False)
    chat_ghost = chat_perplexity(model, val_ex, use_ghost=True)
    chat_delta = (chat_ghost - chat_base) / chat_base * 100.0
    if chat_ghost < chat_base and abs(chat_delta) >= 3.0:
        verdict = "PASS"
    elif chat_ghost < chat_base:
        verdict = "INCONCLUSIVE"
    else:
        verdict = "FAIL"

    # ---- PROBE 4 (chat-mode): alpha sweep ----
    sweep = []
    for a in ALPHA_GRID:
        model.ghost.alpha.fill_(float(a))
        sweep.append((a, chat_perplexity(model, val_ex, use_ghost=True)))
    best_alpha = min(sweep, key=lambda t: t[1])[0]

    # ---- Voice A/B (chat-mode generation), alpha=0 vs best_alpha ----
    print("\n" + "#" * 78)
    print(f"VOICE A/B (chat-mode generation): alpha=0.0 (base) vs alpha={best_alpha} (base+ghost)")
    print("#" * 78, flush=True)
    for pi, prompt in enumerate(AB_PROMPTS):
        print("\n" + "=" * 78)
        print(f"PROMPT {pi+1}: {prompt}")
        print("=" * 78, flush=True)
        seed = 4321 + pi
        for a in [0.0, best_alpha]:
            out = chat_generate(model, tok, prompt, a, seed, stop_ids)
            tag = "[alpha=0.0  BASE       ]" if a == 0.0 else f"[alpha={best_alpha}  BASE+GHOST ]"
            print(f"\n--- {tag} ---\n{out}", flush=True)

    # ---- raw-mode reference (LAST: overwrites the ghost with voice_02) ----
    raw_base = raw_ghost = None
    if os.path.isfile(RAW_CKPT):
        model.ghost.load_state_dict(torch.load(RAW_CKPT, map_location=DEVICE))
        model.ghost.alpha.fill_(ALPHA_INIT)
        raw_base  = math.exp(mean_loss(model, tok, val_turns, use_ghost=False))
        raw_ghost = math.exp(mean_loss(model, tok, val_turns, use_ghost=True))

    # ---- report ----
    print("\n" + "#" * 78)
    print("RESULTS")
    print("#" * 78)
    print(f"PROBE 1 chat-mode (assistant-only ppl @alpha={ALPHA_INIT}):  "
          f"base={chat_base:.2f}  base+ghost={chat_ghost:.2f}  "
          f"delta={chat_delta:+.1f}%   -> {verdict}")
    if raw_base is not None:
        raw_delta = (raw_ghost - raw_base) / raw_base * 100.0
        print(f"PROBE 1 raw-mode  (reference, voice_02):              "
              f"base={raw_base:.2f}  base+ghost={raw_ghost:.2f}  delta={raw_delta:+.1f}%")
    print(f"PROBE 2 base frozen: fingerprint delta = {fp_after - fp_before:.6e}  (must be 0)")
    print(f"PROBE 3 tiny ghost:  ghost/base = {100*g/b:.3f}%")
    print(f"PROBE 4 chat-mode alpha sweep (smooth & bounded?):")
    for a, p in sweep:
        star = "  <- min" if a == best_alpha else ""
        print(f"    alpha={a:<4}  chat val ppl={p:.2f}{star}")
    print(f"\nbest alpha (chat-mode): {best_alpha}   |   train best epoch: {best_epoch}, "
          f"best val loss: {best_val:.4f}")
    print(f"\nVERDICT: {verdict}")
