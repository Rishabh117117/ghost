"""
engram_hash.py (v3) - DETERMINISTIC hash addressing for a fact-memory bank.

v1/v2 autopsy: a LEARNED router collapsed before the real question could be
tested (v2 B1: 7 unique slots, one slot at 68% share, values ~0). Root cause is
a cold-start trap: zero-init values make all slots identical, so there is no
routing gradient until values differ, but values cannot differ until routing
concentrates. v3 removes the router entirely:

  - Each entity gets a FIXED slot by blake2b(surface form) % K, with collisions
    resolved at ASSIGN time by linear probing; the entity->slot map is committed
    before training. Optional S consecutive probed slots per entity (S pinned).
  - Keys / W_q / softmax routing do not exist. Read = gather the entity's
    assigned slot(s) directly. The cold-start trap cannot occur.
  - Output (v1 silence lesson, no RMSNorm): m = beta * values[slots].mean(0),
    h' = h + m, values zero-init -> exact-zero write at init.
  - The last N_NULL slots are zero-pinned null slots: any sequence without a
    known entity (wikitext replay, drift eval, unknown distractors) reads them
    and gets an exact zero -> the bank is silent off-entity by construction.
  - Telemetry is PERMANENT: value_norm_stats() makes "did anything get
    written" a number, never an inference.
"""
import hashlib

import torch
import torch.nn as nn

K_SLOTS = 32768
N_NULL = 64           # last N_NULL slots: zero-pinned nulls (the silence path)


def entity_home(name, k_usable=K_SLOTS - N_NULL):
    """Deterministic home slot for an entity surface form."""
    h = hashlib.blake2b(name.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") % k_usable


def assign_slots(entities, s_slots, k=K_SLOTS, n_null=N_NULL):
    """Commit the key->slot map. `entities` = [(key, surface_form)] in PINNED
    order (entity_id ascending; key may be an entity_id or "eid|attr" for
    per-fact addressing). Each key owns its first S free slots scanning
    forward from its hash home (linear probe past occupied slots, wrapping
    inside the usable region). `probes` counts occupied slots skipped = the
    collision burden of this key's assignment.
    """
    k_usable = k - n_null
    taken = set()
    mapping = {}
    for eid, name in entities:
        home = entity_home(name, k_usable)
        slots, probes, cur = [], 0, home
        while len(slots) < s_slots:
            if cur in taken:
                probes += 1
            else:
                taken.add(cur)
                slots.append(cur)
            cur = (cur + 1) % k_usable
            if probes > k_usable:
                raise RuntimeError("slot space exhausted")
        mapping[eid] = {"name": name, "home": home,
                        "slots": slots, "probes": probes}
    return mapping


def null_row(s_slots, k=K_SLOTS, n_null=N_NULL):
    """The slot row a sequence with NO known entity reads (zero-pinned)."""
    return [k - n_null + j for j in range(min(s_slots, n_null))]


class HashEngramBank(nn.Module):
    """Value bank with deterministic addressing. No keys, no router.

    Callers set the active slot rows (one row of S slot ids per batch row)
    with set_slots() before each forward; forward returns beta * mean of the
    gathered values, broadcast over the time dimension.
    """

    def __init__(self, d_model, dtype, k=K_SLOTS, n_null=N_NULL):
        super().__init__()
        self.k, self.n_null = k, n_null
        self.values = nn.Parameter(torch.zeros(k, d_model))     # zero-init
        self.beta = nn.Parameter(torch.tensor(1.0))             # learned magnitude
        mask = torch.ones(k, 1)
        mask[k - n_null:] = 0.0                                  # null slots pinned
        self.register_buffer("val_mask", mask)
        self.values.register_hook(lambda g: g * self.val_mask)
        self.to(dtype=dtype)
        self.enabled = True
        self._slot_idx = None

    def set_slots(self, rows):
        """rows: list of slot-id lists, one per batch row (all the same length)."""
        self._slot_idx = torch.tensor(rows, dtype=torch.long,
                                      device=self.values.device)

    def forward(self, h):
        idx = self._slot_idx
        if idx is None:
            raise RuntimeError("HashEngramBank: set_slots() before forward")
        if idx.size(0) != h.size(0):          # e.g. beam/cache batch mismatch
            idx = idx[:1].expand(h.size(0), -1)
        m = self.values[idx].mean(dim=1)                       # [B, d_model]
        return (self.beta * m).unsqueeze(1).to(h.dtype)        # broadcast over T

    @torch.no_grad()
    def value_norm_stats(self, assigned=None):
        """Permanent telemetry: 'did anything get written' as a number."""
        n = self.values.float().norm(dim=-1)                   # [K]
        live = n[: self.k - self.n_null]
        out = {"mean_all": round(live.mean().item(), 6),
               "max_all": round(live.max().item(), 6),
               "nonzero_slots": int((live > 1e-6).sum().item()),
               "null_max": round(n[self.k - self.n_null:].max().item(), 6)}
        if assigned:
            a = n[torch.tensor(sorted(assigned), device=n.device)]
            out["mean_assigned"] = round(a.mean().item(), 6)
            out["max_assigned"] = round(a.max().item(), 6)
        return out


class HashEngramModel(nn.Module):
    """Frozen base + additive mid-layer write from a HashEngramBank."""

    def __init__(self, base, device, tap_layer):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.base.eval()
        self.bank = HashEngramBank(base.config.hidden_size, base.dtype).to(device)
        n_layers = len(base.model.layers)
        self.tap = min(tap_layer, n_layers - 1)
        self._handle = base.model.layers[self.tap].register_forward_hook(self._hook)

    def _hook(self, module, inp, out):
        if not self.bank.enabled:
            return out
        hs = out[0] if isinstance(out, tuple) else out
        hs = hs + self.bank(hs)
        return (hs,) + tuple(out[1:]) if isinstance(out, tuple) else hs

    def remove(self):
        self._handle.remove()

    def logits(self, ids, attn=None):
        return self.base(ids, attention_mask=attn).logits

    def trainable(self):
        return [p for p in self.bank.parameters() if p.requires_grad]
