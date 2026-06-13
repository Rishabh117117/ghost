"""
engram_alloc.py (v6.5) - allocation and composition for selected KV memory.

v6 mechanism findings this module answers to:
  - global top-k starves whole attributes (profession 100% @ k=8 while
    birth_year/quirk die) -> FLOOR-THEN-GREEDY allocator;
  - pos-0 is a verified attention sink (mass keeps it 273-281/288, norm's
    124/288 went +725% ppl) -> the sink is always in the floor;
  - the product bridge needs MULTIPLE entities loaded at once ->
    sequential segment composition with per-slot RoPE offsets (capture each
    segment at its own positions - standard multi-document mechanics).

No training anywhere. Floor metadata (fact-sentence spans) is derived by
construction from the context template - metadata the deployment system
genuinely owns; there is no per-attribute tuning of any kind.
"""
import torch

from engram import context_for

ATTR_ORDER = ["birth_year", "city", "profession", "employer", "quirk"]


def ctx_sentences(e):
    """The 5 fact sentences exactly as context_for() concatenates them."""
    return [f"{e['name']} was born in {e['birth_year']}.",
            f"{e['name']} lives in {e['city']}.",
            f"{e['name']} works as a {e['profession']}.",
            f"{e['name']} is employed by {e['employer']}.",
            f"A quirk of {e['name']}: {e['quirk']}.\n".rstrip("\n")]


def sentence_token_spans(tok, e, max_len=128):
    """Token [start, end) range of each fact sentence inside context_for(e),
    by construction (char offsets -> offset mapping)."""
    ctx = context_for(e)
    sents = ctx_sentences(e)
    char_spans, pos = [], 0
    for s in sents:
        i = ctx.find(s, pos)
        assert i >= 0, f"sentence not found in ctx: {s!r}"
        char_spans.append((i, i + len(s)))
        pos = i + len(s)
    enc = tok(ctx, return_offsets_mapping=True, truncation=True,
              max_length=max_len)
    spans = []
    for (cs, ce) in char_spans:
        toks = [t for t, (a, b) in enumerate(enc["offset_mapping"])
                if b > a and a < ce and b > cs]
        spans.append((min(toks), max(toks) + 1) if toks else (0, 0))
    return spans, len(enc["input_ids"])


def floor_then_greedy(kv, score, k, spans, floor_per_span=2):
    """Keep set per layer per kv-head =
         {position 0}                          (the verified sink)
       u top-`floor_per_span` per fact span    (the starvation floor)
       u top remaining global by mass          (greedy fill to exactly k).
    Equal budget vs global top-k at the same k. Also returns the
    budget-waste audit: how many floor positions duplicate what plain
    greedy would have kept anyway."""
    sel, kept_map = [], []
    n_floor = n_dup = n_lh = 0
    recap_num = recap_den = 0.0
    for l, (kl, vl) in enumerate(kv):
        n_kv, T, hd = kl.shape
        kk = min(k, T)
        ks, vs, pos_l = [], [], []
        for h in range(n_kv):
            s = score[l, h]
            order = torch.argsort(s, descending=True).tolist()
            greedy_set = set(order[:kk])
            span_picks = set()
            for (a, b) in spans:
                if b > a:
                    span_order = sorted(range(a, min(b, T)),
                                        key=lambda p: -float(s[p]))
                    span_picks.update(span_order[:floor_per_span])
            span_picks.discard(0)
            # the sink survives truncation unconditionally; span picks are
            # truncated by mass if the budget is smaller than the floor
            floor_list = [0] + sorted(span_picks, key=lambda p: -float(s[p]))
            floor = set(floor_list[:kk])
            keep = set(floor)
            for p in order:
                if len(keep) >= kk:
                    break
                keep.add(p)
            idx = torch.tensor(sorted(keep), dtype=torch.long)
            ks.append(kl[h, idx])
            vs.append(vl[h, idx])
            pos_l.append(idx.tolist())
            n_floor += len(floor)
            n_dup += len(floor & greedy_set)
            n_lh += 1
            recap_num += float(s[idx].sum())
            recap_den += float(s.sum())
        sel.append((torch.stack(ks), torch.stack(vs)))
        kept_map.append(pos_l)
    audit = {"floor_per_lh": round(n_floor / max(n_lh, 1), 2),
             "floor_dup_of_greedy_per_lh": round(n_dup / max(n_lh, 1), 2),
             "floor_forced_per_lh": round((n_floor - n_dup) / max(n_lh, 1), 2)}
    recap = recap_num / max(recap_den, 1e-9)
    return sel, kept_map, recap, audit


def stack_segments(segments):
    """Concatenate per-layer (k, v) segment lists along the position axis.
    Each segment was captured at its own RoPE offset, so phases are already
    correct - this is plain concatenation."""
    out = []
    n_layers = len(segments[0])
    for l in range(n_layers):
        out.append((torch.cat([seg[l][0] for seg in segments], dim=1),
                    torch.cat([seg[l][1] for seg in segments], dim=1)))
    return out
