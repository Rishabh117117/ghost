"""Unit tests for the engram-v4 token-span locator (run on CPU, no GPU).

Covers: every training template, every held-out QA phrasing (bare and in the
Question/Answer framing the eval actually uses), double-mention quirk
templates, names at sequence start, and truncation.
"""
import sys

from transformers import AutoTokenizer

from engram_data import TRAIN_TEMPLATES, QA_TEMPLATES, ATTRS, find_spans
from engram_score import qa_prompt
from engram_span import char_spans, token_mask

NAMES = ["Vexa Quorin", "Brivо Sl".replace("о", "o") + "ater", "Zubrenkal Fendarosti"]
VALUES = {"birth_year": "1947", "city": "Tramelo", "profession": "botanist",
          "employer": "Krastel Holdings", "quirk": "collects antique grelvins"}


def covered_text(text, ids, mask, tok):
    """Decode only the masked tokens and strip — should reconstruct the name."""
    keep = [i for i, m in zip(ids, mask) if m > 0]
    return tok.decode(keep).strip()


def run():
    tok = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
    failures = 0

    def check(text, name, what):
        nonlocal failures
        spans = find_spans(text, name)
        assert spans and all(text[s:e] == name for s, e in spans)
        assert char_spans(text, name) == [tuple(s) for s in spans]
        ids, mask = token_mask(tok, text, spans=[tuple(s) for s in spans])
        assert len(ids) == len(mask)
        got = covered_text(text, ids, mask, tok)
        n_mentions = len(spans)
        ok = sum(mask) >= n_mentions and name in got or got.count(name[:4]) >= n_mentions
        # the masked tokens must contain every mention's text
        ok = got.replace(" ", "").count(name.replace(" ", "")) >= n_mentions
        status = "ok" if ok else "FAIL"
        if not ok:
            failures += 1
            print(f"[{status}] {what}: masked={got!r} expected {n_mentions}x {name!r}")
        else:
            print(f"[ok] {what}: {sum(int(m) for m in mask)} on-name tokens")

    for name in NAMES:
        for attr in ATTRS:
            for t, tmpl in enumerate(TRAIN_TEMPLATES[attr]):
                check(tmpl.format(name=name, value=VALUES[attr]), name,
                      f"train {attr}/t{t} {name.split()[0]}")
            q = QA_TEMPLATES[attr].format(name=name)
            check(q, name, f"qa-bare {attr}")
            check(qa_prompt(q), name, f"qa-framed {attr}")

    # truncation: mask must stay aligned with truncated ids
    long = "Filler. " * 50 + "Vexa Quorin lives in Tramelo."
    ids, mask = token_mask(tok, long, name="Vexa Quorin", max_length=32)
    assert len(ids) == len(mask) == 32 and sum(mask) == 0, "truncated-away name must give empty mask"
    print("[ok] truncation alignment")

    if failures:
        print(f"\n{failures} FAILURES")
        sys.exit(1)
    print("\nALL GREEN")


if __name__ == "__main__":
    run()
