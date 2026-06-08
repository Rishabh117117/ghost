"""
bank.py - ghost Stage 2, first increment: a 2-entry BANK + a no-training
embedding-centroid top-1 ROUTER, and a falsifiable test of both halves:
  (a) does the router pick the right ghost?
  (b) does picking right actually matter (four-way per-domain perplexity)?

Architecture is untouched: GhostModel / GhostStream / compressor are imported
from ghost.py. The router uses ONLY the frozen base's own representation
(mean-pooled final hidden state) - no new embedder, no learned routing.

Bank entries:
  A = voice ghost  (ghosts/ghost_voice_03_chat.pt, corpus data/voice.txt)  -> STYLE signal
  B = tool ghost   (ghosts/ghost_b_01.pt,         corpus data/corpus_b.txt) -> TOPIC signal

This is a measurement: it can come back INCONCLUSIVE/FAIL. A diffuse voice
centroid routing poorly is the expected "voice needs a style router" finding,
not a hidden failure.
"""
import os
import math
import random

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from ghost import (
    GhostModel, MODEL_NAME, DEVICE, SEED, MAX_LEN,
    CORPUS_PATH as VOICE_PATH, load_corpus, split_corpus,
    base_fingerprint, param_counts,
)
from chat_test import (
    build_examples, _im_end_id, reinit_ghost,
)

HERE        = os.path.dirname(os.path.abspath(__file__))
CORPUS_B_PATH = os.path.join(HERE, "data", "corpus_b.txt")
GHOST_A_CKPT  = os.path.join(HERE, "ghosts", "ghost_voice_03_chat.pt")
GHOST_B_CKPT  = os.path.join(HERE, "ghosts", "ghost_b_01.pt")

N_ROUTE = 50    # held-out routing inputs per class
N_PPL   = 80    # held-out inputs per domain for the four-way ppl
GHOST_B_DONE = GHOST_B_CKPT + ".done"   # marker: B training converged (early-stopped)


def train_ghost_b_resumable(model, tok, train_turns, val_turns, ckpt, done_flag,
                            max_epochs=None, lr=None, weight_decay=None,
                            patience=None, max_len=MAX_LEN, seed=SEED):
    """
    Crash-resilient Ghost B trainer for this Modern-Standby laptop: same objective
    and knobs as ghost.train (raw next-token, AdamW, early-stop), but checkpoints
    the best-so-far ghost EVERY time val improves and resumes from it on relaunch.
    A death (system standby) costs at most the in-progress epoch.
    """
    from ghost import mean_loss, LR, WEIGHT_DECAY, MAX_EPOCHS, PATIENCE
    lr = LR if lr is None else lr
    weight_decay = WEIGHT_DECAY if weight_decay is None else weight_decay
    max_epochs = MAX_EPOCHS if max_epochs is None else max_epochs
    patience = PATIENCE if patience is None else patience
    if os.path.isfile(ckpt):
        model.ghost.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        model.ghost.alpha.fill_(1.0)
        best_val = mean_loss(model, tok, val_turns, use_ghost=True, max_len=max_len)
        print(f"resuming Ghost B from checkpoint (val loss {best_val:.4f})", flush=True)
    else:
        reinit_ghost(model)
        best_val = float("inf")
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                            lr=lr, weight_decay=weight_decay)
    rng = random.Random(seed)
    bad = 0
    order = train_turns[:]
    for epoch in range(max_epochs):
        model.train(); rng.shuffle(order); running, nb = 0.0, 0
        for text in order:
            ids = tok(text, return_tensors="pt", truncation=True,
                      max_length=max_len).input_ids.to(DEVICE)
            if ids.size(1) < 2:
                continue
            _, loss = model(ids, labels=ids)
            opt.zero_grad(); loss.backward(); opt.step()
            running += loss.item(); nb += 1
        vl = mean_loss(model, tok, val_turns, use_ghost=True, max_len=max_len)
        print(f"epoch {epoch:3d} | train loss {running/max(nb,1):.4f} | val loss {vl:.4f}", flush=True)
        if vl < best_val - 1e-4:
            best_val, bad = vl, 0
            model.ghost.alpha.fill_(1.0)
            torch.save(model.ghost.state_dict(), ckpt)   # checkpoint best-so-far each improvement
            print(f"  checkpointed -> {os.path.basename(ckpt)}", flush=True)
        else:
            bad += 1
            if bad >= patience:
                print(f"early stop (best val {best_val:.4f})", flush=True)
                break
    open(done_flag, "w").close()
    model.ghost.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.ghost.alpha.fill_(1.0)
    return best_val


# ---- embedding in the base's own space (mean-pooled final hidden state) ------
@torch.no_grad()
def embed(model, tok, text, max_len=MAX_LEN):
    ids = tok(text, return_tensors="pt", truncation=True, max_length=max_len).input_ids.to(DEVICE)
    h = model.base.model(ids).last_hidden_state[0]      # [T, d_model]
    return h.mean(0).float()                            # [d_model]


@torch.no_grad()
def centroid(model, tok, turns):
    acc = None
    for t in turns:
        v = embed(model, tok, t)
        acc = v if acc is None else acc + v
    return acc / max(len(turns), 1)


class Router:
    """Embeds an input in the base's space, picks top-1 ghost by cosine to centroids."""
    def __init__(self, model, tok, centroids, names):
        self.model, self.tok = model, tok
        self.centroids = centroids          # list of [d_model] tensors
        self.names = names

    def cosines(self, text):
        v = embed(self.model, self.tok, text)
        return [F.cosine_similarity(v, c, dim=0).item() for c in self.centroids]

    def route(self, text):
        cs = self.cosines(text)
        return int(max(range(len(cs)), key=lambda i: cs[i])), cs


class GhostBank:
    """N ghost checkpoints + per-ghost domain centroid + a top-1 router + serving."""
    def __init__(self, model, tok, entries):
        # entries: list of dict(name, ckpt, train_turns)
        self.model, self.tok = model, tok
        self.entries = entries
        cents = [centroid(model, tok, e["train_turns"]) for e in entries]
        for e, c in zip(entries, cents):
            e["centroid"] = c
        self.router = Router(model, tok, cents, [e["name"] for e in entries])

    def load(self, idx):
        """Load ghost idx's weights into the shared frozen-base model."""
        sd = torch.load(self.entries[idx]["ckpt"], map_location=DEVICE)
        self.model.ghost.load_state_dict(sd)
        self.model.ghost.alpha.fill_(1.0)

    def serve(self, text):
        """Route -> load the selected ghost -> ready to run GhostModel."""
        idx, cs = self.router.route(text)
        self.load(idx)
        return self.entries[idx]["name"], idx, cs


# ---- masked (chat-mode) per-example losses, same masking as the Stage-1 test --
@torch.no_grad()
def per_example_losses(model, examples, use_ghost):
    model.eval()
    out = []
    for ids, labels in examples:
        ids, labels = ids.to(DEVICE), labels.to(DEVICE)
        n = int((labels[:, 1:] != -100).sum().item())
        if n == 0:
            out.append((0.0, 0)); continue
        _, loss = model(ids, labels=labels, use_ghost=use_ghost)
        out.append((loss.item(), n))
    return out


def agg_ppl(pairs):
    tot = sum(l * n for l, n in pairs)
    nt = sum(n for _, n in pairs)
    return math.exp(tot / max(nt, 1))


def keep_awake():
    """Assert that the system stay awake while this process runs (Windows power
    request — OEM power-management can't revert an app assertion the way it
    reverts powercfg settings). Best-effort; no-op off Windows."""
    try:
        import ctypes
        ES_CONTINUOUS = 0x80000000
        ES_SYSTEM_REQUIRED = 0x00000001
        ES_AWAYMODE_REQUIRED = 0x00000040
        ctypes.windll.kernel32.SetThreadExecutionState(
            ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED)
        print("keep-awake: power assertion set (ES_CONTINUOUS|ES_SYSTEM_REQUIRED|ES_AWAYMODE_REQUIRED)", flush=True)
    except Exception as e:
        print(f"keep-awake: could not set power assertion ({e})", flush=True)


if __name__ == "__main__":
    torch.manual_seed(SEED)
    keep_awake()                          # hold the system awake for the whole run
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
    b, g = param_counts(model)

    # corpora + identical seeded 85/15 splits
    voice = load_corpus(VOICE_PATH)
    btxt  = load_corpus(CORPUS_B_PATH)
    assert len(voice) >= 50 and len(btxt) >= 50, "need real corpora for both ghosts"
    A_train, A_val = split_corpus(voice)
    B_train, B_val = split_corpus(btxt)
    print(f"\nA (voice): {len(voice)} turns -> {len(A_train)} train / {len(A_val)} val")
    print(f"B (tool):  {len(btxt)} turns -> {len(B_train)} train / {len(B_val)} val", flush=True)

    # ---- ensure Ghost B exists (resumable training; survives standby deaths) ----
    os.makedirs(os.path.dirname(GHOST_B_CKPT), exist_ok=True)
    if os.path.isfile(GHOST_B_DONE) and os.path.isfile(GHOST_B_CKPT):
        print(f"\nGhost B converged previously; loading checkpoint.", flush=True)
        model.ghost.load_state_dict(torch.load(GHOST_B_CKPT, map_location=DEVICE))
        model.ghost.alpha.fill_(1.0)
    else:
        print("\ntraining Ghost B (tool) — resumable, per-epoch checkpoint, same knobs as ghost.train ...", flush=True)
        bv = train_ghost_b_resumable(model, tok, B_train, B_val, GHOST_B_CKPT, GHOST_B_DONE)
        print(f"Ghost B done (best val loss {bv:.4f})", flush=True)

    # ---- build the bank (centroids from each ghost's TRAIN corpus) ----
    print("\nbuilding bank centroids (base mean-pooled embeddings over train corpora) ...", flush=True)
    bank = GhostBank(model, tok, [
        {"name": "A_voice", "ckpt": GHOST_A_CKPT, "train_turns": A_train},
        {"name": "B_tool",  "ckpt": GHOST_B_CKPT, "train_turns": B_train},
    ])

    # ---- PROBE 2: base frozen when each ghost is loaded through the bank ----
    bank.load(0); fpA = base_fingerprint(model)
    bank.load(1); fpB = base_fingerprint(model)
    print(f"\nPROBE 2 (base frozen): fp delta after load A = {fpA - fp0:.6e} | "
          f"after load B = {fpB - fp0:.6e}  (must be 0)")
    print(f"PROBE 3 (tiny ghost):  ghost/base = {100*g/b:.3f}%", flush=True)

    # ---- ONE fully-worked routing example (before aggregates) ----
    ex_input = B_val[0]
    idx, cs = bank.router.route(ex_input)
    print("\n" + "=" * 78)
    print("WORKED ROUTING EXAMPLE")
    print("=" * 78)
    print(f"input (a held-out tool trace, truncated): {ex_input[:120]!r}")
    print(f"cosine to centroid A_voice = {cs[0]:.4f}")
    print(f"cosine to centroid B_tool  = {cs[1]:.4f}")
    print(f"-> picked: {bank.entries[idx]['name']}  (expected B_tool)")
    print("=" * 78, flush=True)

    # ---- routing probe set (held-out; not in any centroid/train) ----
    rng = random.Random(SEED)
    A_probe = A_val[:N_ROUTE]
    B_probe = B_val[:N_ROUTE]
    print(f"\nrouting probe: {len(A_probe)} voice (->A) + {len(B_probe)} tool (->B) held-out inputs", flush=True)
    a_correct = sum(bank.router.route(t)[0] == 0 for t in A_probe)
    b_correct = sum(bank.router.route(t)[0] == 1 for t in B_probe)
    a_acc = 100 * a_correct / max(len(A_probe), 1)
    b_acc = 100 * b_correct / max(len(B_probe), 1)

    # ---- four-way per-domain ppl (chat-masked) ----
    A_ex = build_examples(tok, A_val[:N_PPL], im_end_id)
    B_ex = build_examples(tok, B_val[:N_PPL], im_end_id)
    routesA = [bank.router.route(t)[0] for t in A_val[:N_PPL]]
    routesB = [bank.router.route(t)[0] for t in B_val[:N_PPL]]

    bank.load(0)                                  # ghost A
    A_underA = per_example_losses(model, A_ex, use_ghost=True)
    B_underA = per_example_losses(model, B_ex, use_ghost=True)
    base_A   = per_example_losses(model, A_ex, use_ghost=False)
    base_B   = per_example_losses(model, B_ex, use_ghost=False)
    bank.load(1)                                  # ghost B
    A_underB = per_example_losses(model, A_ex, use_ghost=True)
    B_underB = per_example_losses(model, B_ex, use_ghost=True)

    def routed(domain_under_A, domain_under_B, routes):
        return [domain_under_B[i] if routes[i] == 1 else domain_under_A[i]
                for i in range(len(routes))]

    # naming: {A_ex=voice | B_ex=tool}_under{A|B}.  voice domain uses voice examples
    # (A_underA oracle, A_underB wrong); tool domain uses tool examples (B_underB
    # oracle, B_underA wrong). routed selects voice/tool examples under the routed ghost.
    voice_tbl = {
        "base":   agg_ppl(base_A),
        "wrong":  agg_ppl(A_underB),
        "routed": agg_ppl(routed(A_underA, A_underB, routesA[:len(A_ex)])),
        "oracle": agg_ppl(A_underA),
    }
    tool_tbl = {
        "base":   agg_ppl(base_B),
        "wrong":  agg_ppl(B_underA),
        "routed": agg_ppl(routed(B_underA, B_underB, routesB[:len(B_ex)])),
        "oracle": agg_ppl(B_underB),
    }

    # ---- report ----
    print("\n" + "#" * 78)
    print("RESULTS")
    print("#" * 78)
    print("ROUTING ACCURACY (chance = 50%):")
    print(f"  B-domain (tool) inputs -> B : {b_acc:.1f}%   ({b_correct}/{len(B_probe)})")
    print(f"  voice inputs           -> A : {a_acc:.1f}%   ({a_correct}/{len(A_probe)})")
    print("\nFOUR-WAY PERPLEXITY (chat-masked, assistant tokens only):")
    print(f"  {'domain':<8} {'base':>10} {'wrong':>10} {'routed':>10} {'oracle':>10}")
    print(f"  {'voice':<8} {voice_tbl['base']:>10.2f} {voice_tbl['wrong']:>10.2f} "
          f"{voice_tbl['routed']:>10.2f} {voice_tbl['oracle']:>10.2f}")
    print(f"  {'tool':<8} {tool_tbl['base']:>10.2f} {tool_tbl['wrong']:>10.2f} "
          f"{tool_tbl['routed']:>10.2f} {tool_tbl['oracle']:>10.2f}")

    # ---- verdict ----
    mech = b_acc >= 80.0
    matters = (tool_tbl['oracle'] < tool_tbl['wrong'] * 0.97) or (voice_tbl['oracle'] < voice_tbl['wrong'] * 0.97)
    if mech and matters:
        verdict = "PASS"
    elif mech or matters:
        verdict = "INCONCLUSIVE"
    else:
        verdict = "FAIL"
    print(f"\nVERDICT: {verdict}")
    print(f"  (mechanism: B routes >=80%? {mech} | routing matters: oracle<<wrong? {matters})")
    print(f"  per domain, expect oracle ~= routed << wrong <= base when routing helps.")
