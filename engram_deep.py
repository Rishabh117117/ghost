"""
engram_deep.py (v5) - multi-layer KV presence: cached-KV anchor (E0) and
learned deep KV pairs (E1).

The unifying v1-v4 lesson: every nulled injection was SINGLE-SITE, while
tokens-in-context (94%) are present in K,V at EVERY layer. v5 gives a
parametric write that same multi-layer presence:

- DeepMem wraps EVERY decoder layer's attention with v4's
  KVInjectedAttention (post-rope concat + explicit mask; per-layer
  independent, so layer subsets are trivial - the E2-half probe).
- E0 anchor: capture the entity's real per-layer K,V for its context
  paragraph once, inject them everywhere, ask the question at offset
  positions. Mathematically adjacent to in-context (A1) - the plumbing gate.
- E1: DeepKVBank - m learned pairs per layer per entity, warm-started by
  contiguous mean-pooling of the entity's E0 cache, then trained.
- Positions are pinned: memory occupies positions [0, offset); the text
  (fact sentence at train, question at eval) starts at `offset` via explicit
  position_ids, identical between train and eval.
"""
import torch
import torch.nn as nn

from transformers import DynamicCache

from engram_kv import KVInjectedAttention


class DeepMem:
    """Install KVInjectedAttention on ALL layers; set per-layer memories."""

    def __init__(self, base):
        self.base = base
        self.wrappers = []
        for layer in base.model.layers:
            w = KVInjectedAttention(layer.self_attn)
            layer.self_attn = w
            self.wrappers.append(w)
        self.n_layers = len(self.wrappers)

    def clear(self):
        for w in self.wrappers:
            w.mem = None

    def set_deep(self, K, V, layers=None):
        """K, V: [B, L, n_kv, m, head_dim]; layers: optional subset of layer
        indices to arm (others stay silent) - the E2-half probe."""
        live = set(range(self.n_layers)) if layers is None else set(layers)
        for l, w in enumerate(self.wrappers):
            w.mem = (K[:, l], V[:, l]) if l in live else None

    def set_cached(self, kv_per_layer, layers=None):
        """kv_per_layer: list of (k, v) each [n_kv, T, head_dim] (one entity,
        batch 1) - the E0 cached-KV anchor."""
        live = set(range(self.n_layers)) if layers is None else set(layers)
        for l, w in enumerate(self.wrappers):
            if l in live:
                k, v = kv_per_layer[l]
                w.mem = (k.unsqueeze(0), v.unsqueeze(0))
            else:
                w.mem = None

    def telemetry(self, on):
        for w in self.wrappers:
            w.telemetry = on
            w.last_mass = None

    def mass_per_layer(self):
        return [w.last_mass for w in self.wrappers]

    @torch.no_grad()
    def capture_ctx_ids(self, ids):
        """Capture per-layer post-rope (k, v) [n_kv, T, head_dim] for given
        token ids [1, T]."""
        self.clear()
        for w in self.wrappers:
            w.capture = True
        self.base(ids)
        out = []
        for w in self.wrappers:
            k, v = w.captured
            out.append((k[0], v[0]))
            w.capture = False
            w.captured = None
        return out, ids.size(1)

    @torch.no_grad()
    def capture_ctx(self, tok, text, device, max_len=128):
        """One pass over the entity's context; returns per-layer post-rope
        (k, v) [n_kv, T, head_dim] and the token count T."""
        ids = tok(text, return_tensors="pt", truncation=True,
                  max_length=max_len).input_ids.to(device)
        return self.capture_ctx_ids(ids)

    def remove(self):
        for layer, w in zip(self.base.model.layers, self.wrappers):
            layer.self_attn = w.attn
        self.wrappers = []


class DeepKVBank(nn.Module):
    """Dense per-entity deep KV pairs: [n_ents, L, n_kv, m, head_dim] x2.
    Entity -> row via the committed hash map's dense rank (storage layout
    only; addressing semantics stay v3's deterministic hash)."""

    def __init__(self, n_ents, n_layers, n_kv, m_pairs, head_dim, dtype):
        super().__init__()
        self.m = m_pairs
        self.K = nn.Parameter(torch.zeros(n_ents, n_layers, n_kv, m_pairs, head_dim))
        self.V = nn.Parameter(torch.zeros(n_ents, n_layers, n_kv, m_pairs, head_dim))
        self.to(dtype=dtype)

    @torch.no_grad()
    def warm_start(self, rank, kv_per_layer):
        """Contiguous mean-pool the entity's E0 cache T -> m slots, per layer."""
        for l, (k, v) in enumerate(kv_per_layer):
            chunks = torch.tensor_split(torch.arange(k.size(1)), self.m)
            for j, c in enumerate(chunks):
                if len(c):
                    self.K[rank, l, :, j] = k[:, c].mean(dim=1)
                    self.V[rank, l, :, j] = v[:, c].mean(dim=1)

    def gather(self, ranks):
        idx = torch.tensor(ranks, dtype=torch.long, device=self.K.device)
        return self.K[idx], self.V[idx]

    @torch.no_grad()
    def norm_stats(self):
        kn = self.K.float().norm(dim=-1)
        vn = self.V.float().norm(dim=-1)
        return {"k_mean": round(kn.mean().item(), 6),
                "k_max": round(kn.max().item(), 6),
                "v_mean": round(vn.mean().item(), 6),
                "v_max": round(vn.max().item(), 6)}


@torch.no_grad()
def gen_mem(base, tok, prompt, offset, device, max_new=16):
    """Greedy generation with the text at positions [offset, ...) so injected
    memory occupies the positions before it. Uses a real KV cache; wrappers
    re-prepend memory at every step."""
    ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    cache = DynamicCache()
    pos = torch.arange(offset, offset + ids.size(1), device=device)[None]
    out = base(ids, position_ids=pos, past_key_values=cache, use_cache=True)
    toks = []
    nxt = out.logits[0, -1].argmax()
    for i in range(max_new):
        toks.append(int(nxt))
        if int(nxt) == tok.eos_token_id:
            break
        p = torch.tensor([[offset + ids.size(1) + i]], device=device)
        out = base(nxt.view(1, 1), position_ids=p, past_key_values=cache,
                   use_cache=True)
        nxt = out.logits[0, -1].argmax()
    return tok.decode(toks, skip_special_tokens=True)
