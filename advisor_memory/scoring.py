"""Exact scoring against the generator's gold answers, and trap attribution for wrong answers."""

from __future__ import annotations

import datetime as dt
import re

from .data import CONTACT, NUM_WORDS

_ABSTAIN = re.compile(r"\bunknown\b|not (?:been )?(?:mentioned|stated|discussed|provided|specified|disclosed|available|recorded)|"
                      r"no (?:information|record|mention|data)|(?:cannot|can't|can not) (?:be )?(?:determine|determined|tell|find|answer)|"
                      r"\bn/?a\b|don't know|do not know|isn't (?:mentioned|stated)|never (?:mentioned|said|stated)", re.I)
_NUM = re.compile(r"(?<![\w.])-?\$?\s?(\d[\d,]*(?:\.\d+)?)\s*([kK]\b)?")
SYNONYMS = {"text message": ["text message", "texting", "text", "sms"], "phone call": ["phone call", "phone", "call"],
            "video call": ["video call", "video", "zoom"], "email": ["email", "e-mail"]}
MONTHS = "%B %d, %Y|%b %d, %Y|%d %B %Y|%d %b %Y|%B %d %Y|%b %d %Y|%B %dth, %Y|%B %dst, %Y|%B %dnd, %Y|%B %drd, %Y".split("|")


def is_abstain(answer) -> bool:
    return isinstance(answer, str) and bool(_ABSTAIN.search(answer))


def parse_money(text: str) -> float | None:
    m = _NUM.search(str(text))
    if not m:
        return None
    v = float(m.group(1).replace(",", ""))
    return v * 1000 if m.group(2) else v


def parse_int(text: str) -> int | None:
    t = str(text).lower()
    m = re.search(r"\b\d+\b", t)
    if m:
        return int(m.group(0))
    for w, v in NUM_WORDS.items():
        if re.search(rf"\b{w}\b", t):
            return v
    return None


def parse_date(text: str) -> str | None:
    t = str(text)
    m = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", t)
    if m:
        return m.group(0)
    for chunk in re.findall(r"[A-Z][a-z]+\.? \d{1,2}(?:st|nd|rd|th)?,? \d{4}|\d{1,2} [A-Z][a-z]+ \d{4}", t):
        chunk = chunk.replace(".", "")
        for fmt in MONTHS:
            try:
                return dt.datetime.strptime(chunk, fmt).date().isoformat()
            except ValueError:
                continue
    return None


def labels_in(text: str, choices: list[str]) -> set[str]:
    """Closed-set mentions, longest first, so 'moderately conservative' is not also read as 'conservative'."""
    t = " " + re.sub(r"\s+", " ", str(text).lower()) + " "
    found = set()
    terms = []
    for c in choices:
        for s in SYNONYMS.get(c, [c]) if choices == CONTACT else [c]:
            terms.append((s.lower(), c))
    for s, c in sorted(terms, key=lambda x: -len(x[0])):
        pat = r"(?<![a-z])" + re.escape(s) + r"(?![a-z])"
        if re.search(pat, t):
            found.add(c)
            t = re.sub(pat, " ", t)
    return found


def matches(kind: str, answer: str, value, choices=None) -> bool:
    if kind == "money":
        v = parse_money(answer)
        return v is not None and abs(v - float(value)) <= 0.5
    if kind == "int":
        v = parse_int(answer)
        return v is not None and v == int(value)
    if kind == "date":
        return parse_date(answer) == value
    if kind in ("label", "choice"):
        return labels_in(answer, choices) == {value}
    if kind == "name":
        return re.search(r"(?<![A-Za-z])" + re.escape(str(value)) + r"(?![A-Za-z])", str(answer), re.I) is not None
    raise ValueError(kind)


def score(q: dict, answer) -> dict:
    if answer is None or not str(answer).strip():
        return {"correct": False, "status": "no_answer", "trap": None}
    answer = str(answer)
    if q["kind"] == "abstain":
        ok = is_abstain(answer)
        return {"correct": ok, "status": "correct" if ok else "wrong", "trap": None if ok else "answered_unanswerable"}
    if matches(q["kind"], answer, q["gold"], q.get("choices")):
        return {"correct": True, "status": "correct", "trap": None}
    if is_abstain(answer):
        return {"correct": False, "status": "abstained", "trap": "abstained"}
    trap = next((t for t, v in q["traps"].items() if matches(q["kind"], answer, v, q.get("choices"))), None)
    return {"correct": False, "status": "wrong", "trap": trap}
