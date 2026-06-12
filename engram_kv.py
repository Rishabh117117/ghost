"""
engram_kv.py (v4, arm D2) - per-entity VIRTUAL KV pairs injected into one
layer's self-attention.

v3 proved the additive-broadcast write is undecodable by frozen layers while
the same facts as TOKENS (A1) read at 94%. D2 tests the ATTENTION CHANNEL
directly: each entity owns m key/value pairs, GQA-shaped
[n_kv_heads, m, head_dim], CONCATENATED into layer-18 K,V at train and eval.
Addressing stays v3's deterministic hash (entity-id at eval = the same
upper-bound convention).

Implementation notes (pinned to transformers==5.9.0, Qwen3 family):
- KVInjectedAttention replicates Qwen3Attention.forward verbatim and concats
  the virtual pairs AFTER RoPE + cache update, so the pairs live in the same
  representation as cached keys (post-q_norm/k_norm, post-rotation).
- The attention mask is ALWAYS materialized when memory is present: torch
  SDPA's `is_causal` aligns top-left, which mis-masks prefix KV when
  kv_len > q_len. We prepend `m` always-visible columns to an explicit
  additive mask instead.
- WARM-START: each entity's pairs are initialized from the REAL layer-18
  K,V activations of its fact sentences (one base-model pass, contiguous
  mean-pool to m slots) - prompt-tuning-style init from real activations.
- Silence is structural: no entity -> no injection -> bit-exact base model.
"""
import torch
import torch.nn as nn

from transformers.models.qwen3.modeling_qwen3 import (
    apply_rotary_pos_emb, eager_attention_forward)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

K_KV_SLOTS = 4096      # hash space for KV banks (500 entities -> 12% load;
                       # smaller than v3's 32768 because bank size is not under
                       # test and KV slots are 16x wider than value slots)


class KVBank(nn.Module):
    """Hash-slot bank of virtual KV pairs. Gather rows by slot id; gradients
    flow only to gathered rows. Zero-init is NOT used here: rows are
    warm-started from real activations before training."""

    def __init__(self, n_kv_heads, m_pairs, head_dim, dtype, k=K_KV_SLOTS):
        super().__init__()
        self.k, self.m = k, m_pairs
        self.K_bank = nn.Parameter(torch.zeros(k, n_kv_heads, m_pairs, head_dim))
        self.V_bank = nn.Parameter(torch.zeros(k, n_kv_heads, m_pairs, head_dim))
        self.to(dtype=dtype)

    @torch.no_grad()
    def warm_start(self, slot, k_act, v_act):
        """k_act/v_act: [n_kv, T, head_dim] real activations (post-rope keys);
        contiguous mean-pool T into m chunks."""
        chunks = torch.tensor_split(torch.arange(k_act.size(1)), self.m)
        for j, c in enumerate(chunks):
            if len(c):
                self.K_bank[slot, :, j] = k_act[:, c].mean(dim=1)
                self.V_bank[slot, :, j] = v_act[:, c].mean(dim=1)

    def gather(self, slots):
        """slots: list of slot ids (one per batch row) -> (vK, vV) each
        [B, n_kv, m, head_dim], differentiable."""
        idx = torch.tensor(slots, dtype=torch.long, device=self.K_bank.device)
        return self.K_bank[idx], self.V_bank[idx]

    @torch.no_grad()
    def norm_stats(self, assigned):
        idx = torch.tensor(sorted(assigned), device=self.K_bank.device)
        kn = self.K_bank[idx].float().norm(dim=-1)   # [n, n_kv, m]
        vn = self.V_bank[idx].float().norm(dim=-1)
        return {"k_mean": round(kn.mean().item(), 6),
                "k_max": round(kn.max().item(), 6),
                "v_mean": round(vn.mean().item(), 6),
                "v_max": round(vn.max().item(), 6)}


class KVInjectedAttention(nn.Module):
    """Drop-in replacement for one Qwen3Attention layer; set `.mem` to a
    (vK, vV) pair ([B, n_kv, m, head_dim]) to inject, None for bit-exact
    passthrough. `.capture = True` records post-rope K,V for warm-start."""

    def __init__(self, attn):
        super().__init__()
        self.attn = attn
        self.mem = None
        self.capture = False
        self.captured = None
        self.telemetry = False   # measure attention mass on memory columns
        self.last_mass = None
        self.mass_cols = None    # accumulated per-column mass [n_kv, m] (v6 ranker)

    def forward(self, hidden_states, position_embeddings, attention_mask,
                past_key_values=None, **kwargs):
        a = self.attn
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, a.head_dim)

        query_states = a.q_norm(a.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = a.k_norm(a.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = a.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(
                key_states, value_states, a.layer_idx)

        if self.capture:
            self.captured = (key_states.detach(), value_states.detach())

        if self.mem is not None:
            vK, vV = self.mem
            B = hidden_states.size(0)
            if vK.size(0) != B:
                vK = vK[:1].expand(B, -1, -1, -1)
                vV = vV[:1].expand(B, -1, -1, -1)
            m = vK.size(2)
            key_states = torch.cat([vK.to(key_states.dtype), key_states], dim=2)
            value_states = torch.cat([vV.to(value_states.dtype), value_states], dim=2)
            q_len, kv_real = query_states.size(2), key_states.size(2) - m
            if attention_mask is not None:
                pad = torch.zeros(attention_mask.shape[:-1] + (m,),
                                  dtype=attention_mask.dtype,
                                  device=attention_mask.device)
                attention_mask = torch.cat([pad, attention_mask], dim=-1)
            else:
                # explicit causal mask over the real keys (SDPA is_causal
                # aligns TOP-LEFT and would mis-mask the prefix), memory
                # columns always visible
                neg = torch.finfo(query_states.dtype).min
                causal = torch.full((q_len, kv_real), neg,
                                    device=query_states.device,
                                    dtype=query_states.dtype)
                causal = torch.triu(causal, diagonal=kv_real - q_len + 1)
                vis = torch.zeros((q_len, m), device=causal.device,
                                  dtype=causal.dtype)
                attention_mask = torch.cat([vis, causal], dim=-1)[None, None]

        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            a.config._attn_implementation, eager_attention_forward)
        attn_output, attn_weights = attention_interface(
            a, query_states, key_states, value_states, attention_mask,
            dropout=0.0 if not self.training else a.attention_dropout,
            scaling=a.scaling, sliding_window=a.sliding_window, **kwargs)

        if self.telemetry and self.mem is not None:
            # "did the reader look": fraction of attention going to the
            # injected pairs, averaged over heads and query positions
            with torch.no_grad():
                m = self.mem[0].size(2)
                groups = a.num_key_value_groups
                kk = key_states.repeat_interleave(groups, dim=1)
                logits = (query_states.float() @ kk.float().transpose(-1, -2)) * a.scaling
                if attention_mask is not None:
                    logits = logits + attention_mask.float()[..., :logits.size(-1)]
                w = torch.softmax(logits, dim=-1)
                self.last_mass = w[..., :m].sum(-1).mean().item()
                # per-column mass per kv-head (v6 selection ranker): group the
                # query heads belonging to each kv head, sum over query
                # positions, mean over groups and batch -> [n_kv, m]
                B, nq, T, _ = w.shape
                wm = w[..., :m].view(B, nq // groups, groups, T, m)
                cols = wm.sum(dim=3).mean(dim=(0, 2)).float().cpu()
                self.mass_cols = cols if self.mass_cols is None else self.mass_cols + cols

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = a.o_proj(attn_output)
        return attn_output, attn_weights


class KVEngramModel(nn.Module):
    """Frozen base + KVInjectedAttention swapped in at the tap layer."""

    def __init__(self, base, device, tap_layer, m_pairs, k_slots=K_KV_SLOTS):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.base.eval()
        cfg = base.config
        n_kv = cfg.num_key_value_heads
        head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
        n_layers = len(base.model.layers)
        self.tap = min(tap_layer, n_layers - 1)
        self.wrap = KVInjectedAttention(base.model.layers[self.tap].self_attn)
        base.model.layers[self.tap].self_attn = self.wrap
        self.bank = KVBank(n_kv, m_pairs, head_dim, base.dtype, k=k_slots).to(device)

    def set_entity(self, slots):
        """slots: list of slot ids (one per row) or None for no injection."""
        self.wrap.mem = self.bank.gather(slots) if slots is not None else None

    def set_mem_raw(self, vK, vV):
        self.wrap.mem = (vK, vV)

    def remove(self):
        self.base.model.layers[self.tap].self_attn = self.wrap.attn
        self.wrap.mem = None

    def logits(self, ids, attn=None):
        return self.base(ids, attention_mask=attn).logits

    def trainable(self):
        return [p for p in self.bank.parameters() if p.requires_grad]


@torch.no_grad()
def capture_warm_start(em, tok, texts, device, max_len=64):
    """Run the base over `texts` (one entity's fact sentences) with the tap
    layer in capture mode; return concatenated post-rope K,V activations
    [n_kv, total_T, head_dim]."""
    em.wrap.mem = None
    em.wrap.capture = True
    ks, vs = [], []
    for t in texts:
        ids = tok(t, return_tensors="pt", truncation=True,
                  max_length=max_len).input_ids.to(device)
        em.base(ids)
        k, v = em.captured_kv()
        ks.append(k[0])
        vs.append(v[0])
    em.wrap.capture = False
    return torch.cat(ks, dim=1), torch.cat(vs, dim=1)


def _captured_kv(self):
    return self.wrap.captured


KVEngramModel.captured_kv = _captured_kv
