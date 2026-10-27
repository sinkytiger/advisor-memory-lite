"""Exploratory, not pre-registered: does the gold value reach the reader?

For questions whose gold answer is a value stated in one meeting (single-session, knowledge-update and
decision questions), check whether that value appears in the context each system showed the reader, and, for
Mem0, anywhere in the store. A value can appear for an unrelated reason, so the counts are an upper bound on
"the memory had the answer", not proof of it.

    python scripts/explore_context.py --out reports/context_check.md
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from advisor_memory.data import NUM_WORDS, load  # noqa: E402
from advisor_memory.report import load_runs  # noqa: E402
from advisor_memory.run import WRITER  # noqa: E402
from advisor_memory.scoring import _NUM, labels_in, parse_date  # noqa: E402
from advisor_memory.systems import build_systems  # noqa: E402

CATS = ["single_session", "knowledge_update", "decision"]
SYSTEMS = ["raw_rag", "mem0", "mem0_addonly", "kyc"]
_PREFIX = re.compile(r"^\[[^\]]*\]\s*")


def present(q: dict, text: str) -> bool:
    """The gold value appears somewhere in the text (every number, date or label is checked, not just the first)."""
    kind, gold = q["kind"], q["gold"]
    if kind == "money":
        return any(abs(float(m.group(1).replace(",", "")) * (1000 if m.group(2) else 1) - float(gold)) <= 0.5
                   for m in _NUM.finditer(text))
    if kind == "int":
        t = text.lower()
        return any(int(x) == int(gold) for x in re.findall(r"\b\d+\b", t)) or \
            any(v == int(gold) and re.search(rf"\b{w}\b", t) for w, v in NUM_WORDS.items())
    if kind == "date":
        return gold in text or any(parse_date(line) == gold for line in text.splitlines())
    if kind in ("label", "choice"):
        return gold in labels_in(text, q["choices"])
    if kind == "name":
        return re.search(r"(?<![A-Za-z])" + re.escape(str(gold)) + r"(?![A-Za-z])", text, re.I) is not None
    return False


def strip_dates(text: str) -> str:
    """Drop the '[YYYY-MM-DD]' prefix each Mem0 line carries, so its date is not read as an amount."""
    return "\n".join(_PREFIX.sub("", line) for line in text.splitlines())


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/dataset")
    p.add_argument("--work", default="runs/memory")
    p.add_argument("--runs", default="runs/main")
    p.add_argument("--out", default="reports/context_check.md")
    args = p.parse_args()
    clients = [c for c in load(args.data) if c.split == "test"]
    systems = build_systems(SYSTEMS, WRITER)
    runs = load_runs(args.runs)
    readers = sorted({r for r, _ in runs})
    rows = []   # one per (system, question)
    for c in clients:
        qs = [q for q in c.questions if q["cat"] in CATS]
        for name, sys_ in systems.items():
            sys_.prepare(c, Path(args.work))
            store = None
            if name.startswith("mem0"):
                got = sys_.mems[c.cid].get_all(user_id=c.cid, limit=100_000)
                items = got.get("results", []) if isinstance(got, dict) else got
                store = "\n".join(it.get("memory", "") for it in items)
            for q in qs:
                text = sys_.context(c, q)["text"]
                if name.startswith("mem0"):
                    text = strip_dates(text)
                rows.append({"system": name, "qid": q["qid"], "cat": q["cat"], "in_context": present(q, text),
                             "in_store": present(q, store) if store is not None else None})
        print(f"{c.cid} checked", flush=True)

    out = ["# Does the gold value reach the reader? (exploratory)", "",
           f"Test split, {len({r['qid'] for r in rows})} questions whose answer is a value stated in a meeting "
           f"({', '.join(CATS)}). \"In context\" means the gold value appears in what the reader was shown; for Mem0, "
           "\"in store\" means it appears in any stored memory. A value can appear for an unrelated reason, so these "
           "are upper bounds.", "",
           "| System | n | gold value in context | in Mem0 store | " + " | ".join(f"{r} correct when in context" for r in readers) + " |",
           "|---|---|---|---|" + "---|" * len(readers)]
    by = defaultdict(list)
    for r in rows:
        by[r["system"]].append(r)
    for name in SYSTEMS:
        rs = by[name]
        inc = [r for r in rs if r["in_context"]]
        store = sum(1 for r in rs if r["in_store"]) if name.startswith("mem0") else None
        cells = []
        for reader in readers:
            rec = runs.get((reader, name), {})
            ok = sum(1 for r in inc if rec.get(r["qid"], {}).get("score", {}).get("correct"))
            cells.append(f"{ok}/{len(inc)}")
        out.append(f"| {name} | {len(rs)} | {len(inc)} ({100 * len(inc) / len(rs):.0f}%) | "
                   f"{'-' if store is None else f'{store} ({100 * store / len(rs):.0f}%)'} | " + " | ".join(cells) + " |")
    out += ["", "By category (gold value in context):", "", "| System | " + " | ".join(CATS) + " |", "|---|" + "---|" * len(CATS)]
    for name in SYSTEMS:
        cells = []
        for cat in CATS:
            rs = [r for r in by[name] if r["cat"] == cat]
            cells.append(f"{sum(r['in_context'] for r in rs)}/{len(rs)}")
        out.append(f"| {name} | " + " | ".join(cells) + " |")
    text = "\n".join(out) + "\n"
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text, encoding="utf-8")
    Path(args.out).with_suffix(".jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
