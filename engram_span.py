"""
engram_span.py (v4) - char-span -> token-position locator.

The v4 position experiment needs the ENTITY's token positions in every
training/eval sequence. Char spans are known by construction (the dataset is
template-generated; engram_data emits `name_spans`); this module converts them
to token masks via the tokenizer's offset mapping, and can also locate a name
in a freshly built prompt (e.g. the "Question: ...\nAnswer:" framing) where
the char offsets have shifted.

A token is "on the name" if its char range overlaps any name span. Leading
whitespace absorbed into a name's first token counts as on-name (the write
rides the token that CARRIES the name text).
"""
import torch


def char_spans(text, name):
    spans, i = [], text.find(name)
    while i >= 0:
        spans.append((i, i + len(name)))
        i = text.find(name, i + 1)
    return spans


def token_mask(tok, text, name=None, spans=None, max_length=None):
    """Float mask over token positions: 1.0 where the token overlaps a name
    span. Returns (input_ids list, mask list). Requires a fast tokenizer."""
    if spans is None:
        spans = char_spans(text, name)
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=True,
              truncation=max_length is not None, max_length=max_length)
    ids = enc["input_ids"]
    mask = []
    for (a, b) in enc["offset_mapping"]:
        on = b > a and any(a < e and b > s for s, e in spans)
        mask.append(1.0 if on else 0.0)
    return ids, mask


def batch_masks(masks, ml, device):
    """Right-pad per-row token masks to length ml -> float tensor [B, ml]."""
    out = torch.zeros(len(masks), ml)
    for i, m in enumerate(masks):
        out[i, :min(len(m), ml)] = torch.tensor(m[:ml])
    return out.to(device)
