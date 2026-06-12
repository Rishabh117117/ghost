"""
engram_select.py (v6) - SELECTION compression of per-entity KV memory.

v5: full cached KV (E0) = 93.5%; mean-pooled pairs (E1) = 0.0% at 2.4%
attention mass - averaging post-RoPE keys is a compressor bug. v6 compresses
the way the cache-compression literature says it survives (H2O/SnapKV
class): KEEP top-k REAL positions per layer per kv-head - original RoPE
phases, original values, untouched. Two rankers:

  MASS - attention mass each cached position received during a CALIBRATION
         pass. Calibration queries are the entity's TRAINING fact sentences
         (the train templates). The held-out QA phrasings are NEVER used for
         calibration - no eval leakage.
  NORM - key L2 norm per position (zero-calibration baseline).
"""
import torch

from engram_data import TRAIN_TEMPLATES, ATTRS


def calib_texts(ent):
    """Calibration queries: the entity's training-template fact sentences
    only (never the held-out QA phrasings)."""
    out = []
    for attr in ATTRS:
        for tmpl in TRAIN_TEMPLATES[attr]:
            out.append(tmpl.format(name=ent["name"], value=str(ent[attr])))
    return out


@torch.no_grad()
def calibrate_mass(dm, base, tok, kv, texts, offset, device, max_len=64):
    """Run calibration texts (positions offset past the memory) with the full
    cache injected and per-column telemetry on. Returns mass [L, n_kv, T]."""
    dm.set_cached(kv)
    for w in dm.wrappers:
        w.telemetry = True
        w.mass_cols = None
    for t in texts:
        ids = tok(t, return_tensors="pt", truncation=True,
                  max_length=max_len).input_ids.to(device)
        pos = torch.arange(offset, offset + ids.size(1), device=device)[None]
        base(ids, position_ids=pos)
    out = torch.stack([w.mass_cols for w in dm.wrappers])      # [L, n_kv, T]
    for w in dm.wrappers:
        w.telemetry = False
        w.mass_cols = None
        w.last_mass = None
    dm.clear()
    return out


def norm_score(kv):
    """Zero-calibration ranker: key L2 norm per position. [L, n_kv, T]."""
    return torch.stack([k.float().norm(dim=-1) for k, _ in kv])


def select_topk(kv, score, k):
    """Keep the top-k positions per layer per kv-head (positions may differ
    across layers/heads - H2O convention). Kept entries are REAL cache
    entries. Returns (selected kv list, kept-position map [L][h] -> sorted
    positions, mass recapture fraction)."""
    sel, kept_map = [], []
    recap_num = recap_den = 0.0
    for l, (kl, vl) in enumerate(kv):
        n_kv, T, hd = kl.shape
        kk = min(k, T)
        ks, vs, pos_l = [], [], []
        for h in range(n_kv):
            idx = torch.topk(score[l, h], kk).indices.sort().values
            ks.append(kl[h, idx])
            vs.append(vl[h, idx])
            pos_l.append(idx.tolist())
            recap_num += float(score[l, h][idx].sum())
            recap_den += float(score[l, h].sum())
        sel.append((torch.stack(ks), torch.stack(vs)))
        kept_map.append(pos_l)
    recap = recap_num / max(recap_den, 1e-9)
    return sel, kept_map, recap


def storage_mb(k, n_layers, n_kv, head_dim, bytes_per=2):
    """Kept entries per entity in MB: k positions x heads x dim x (K+V)."""
    return k * n_layers * n_kv * head_dim * 2 * bytes_per / 1e6
