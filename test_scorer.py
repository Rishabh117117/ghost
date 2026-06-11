"""Unit tests for the repaired engram-v2 scorer. Run: python test_scorer.py"""
from engram_score import scored_hit, is_confabulation, qa_prompt, first_line

CASES_HIT = [
    # (gen, gold, expected) - documents the v1 failure modes the fix addresses
    ("1985", "1985", True),                                   # exact
    ("The birth year is 1985.", "1985", True),                # value mid-sentence
    ("born in 1985, in the city.", "1985", True),             # digit-robust
    ("new zubu!", "New Zubu", True),                          # case + punctuation
    ("He lives in New Zubu.", "New Zubu", True),              # substring + case
    ("a weaver by trade", "weaver", True),                    # profession substring
    ("1990", "1985", False),                                  # wrong year
    ("I don't know.\n1985", "1985", False),                   # answer not on first line
    ("Acme Industries", "Acme Industries", True),             # multiword employer
    ("", "1985", False),                                      # empty generation
    ("the year", None, False),                                # no gold (distractor)
]

CASES_CONFAB = [
    ("Pretoria.", True),                 # confident fabrication
    ("I don't know.", False),            # hedge
    ("There is no record of that.", False),
    ("Unknown.", False),
    ("1923", True),                      # confident numeric fabrication
    ("", False),                         # empty -> not a confabulation
]


def main():
    fails = 0
    for gen, gold, exp in CASES_HIT:
        got = scored_hit(gen, gold)
        ok = got == exp
        fails += not ok
        print(f"[{'ok' if ok else 'FAIL'}] scored_hit({gen!r},{gold!r})={got} exp={exp}")
    for gen, exp in CASES_CONFAB:
        got = is_confabulation(gen)
        ok = got == exp
        fails += not ok
        print(f"[{'ok' if ok else 'FAIL'}] is_confab({gen!r})={got} exp={exp}")
    assert qa_prompt("What year?", "ctx. ") == "ctx. Question: What year?\nAnswer:"
    assert first_line("\n\nfoo\nbar") == "foo"
    print(f"\n{'ALL GREEN' if not fails else str(fails)+' FAILED'}")
    raise SystemExit(1 if fails else 0)


if __name__ == "__main__":
    main()
