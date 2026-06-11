"""
engram_score.py - the repaired QA/recall matcher for engram-v2.

v1 autopsy: A1 in-context scored 92.8% verbatim but 15.8% QA - impossible with
facts in the prompt if the matcher were sound. Two root causes:
  (1) QA prompts were bare questions; a *base* (non-instruct) model continues a
      question with more text, not the answer. v2 frames QA as
      "Question: ...\nAnswer:" so the model is primed to answer.
  (2) The matcher substring-checked the whole 12-token blob. v2 normalizes
      case/whitespace/punctuation, restricts to the FIRST generated line, and
      adds digit-robust matching for years/numbers.

Hedging rules for the confabulation metric are pinned here, in advance.
"""
import re

# Pinned hedging vocabulary for the confabulation metric (committed in advance).
HEDGES = (
    "i don't know", "i do not know", "unknown", "not sure", "no information",
    "cannot", "can't", "n/a", "unclear", "unsure", "none", "no record",
    "i'm not", "i am not", "there is no", "not available", "do not have",
    "don't have", "no data", "not enough information", "i have no",
)


def normalize(s):
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def first_line(s):
    """The answer is the first non-empty line of the generation."""
    for ln in (s or "").splitlines():
        if ln.strip():
            return ln
    return s or ""


def qa_prompt(question, context=""):
    """Frame a held-out QA question so a base model answers it."""
    return f"{context}Question: {question}\nAnswer:"


def scored_hit(gen, gold):
    """True if the gold answer appears in the first generated line.

    - case/whitespace/punctuation-insensitive substring match
    - digit-robust: years/numbers match on their digit runs even if surrounded
      by words ("born in 1985." vs gold "1985")
    """
    if gold is None or str(gold) == "":
        return False
    g = normalize(first_line(gen))
    v = normalize(str(gold))
    if v and v in g:
        return True
    gold_digits = re.findall(r"\d+", str(gold))
    if gold_digits:
        gen_digits = re.findall(r"\d+", first_line(gen))
        if all(d in gen_digits for d in gold_digits):
            return True
    return False


def is_confabulation(gen):
    """Distractor probe: the model FABRICATES (confident answer) vs hedges.

    Confabulation = first line is non-empty, contains an answer-like token, and
    contains no pinned hedge phrase. Abstention/hedging = not a confabulation.
    """
    line = first_line(gen)
    low = line.lower()
    if any(h in low for h in HEDGES):
        return False
    return len(normalize(line)) >= 2
