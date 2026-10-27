"""Tables and pre-registered hypothesis checks from run directories."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .data import build_dataset, load
from .systems import SYSTEMS

CATS = ["single_session", "knowledge_update", "temporal", "aggregation", "decision", "abstention"]
B = 10_000


def load_runs(root: str | Path) -> dict[tuple[str, str], dict[str, dict]]:
    """{(reader, system): {qid: record}}"""
    out: dict = defaultdict(dict)
    for f in Path(root).glob("*/*/*.json"):
        r = json.loads(f.read_text(encoding="utf-8"))
        out[(r["reader"], r["system"])][r["qid"]] = r
    return out


def boot_ci(per_client: dict[str, np.ndarray], seed: int = 0) -> tuple[float, float, float]:
    """Mean and 95% interval, resampling clients and then questions within each client."""
    rng = np.random.default_rng(seed)
    keys = [k for k, v in per_client.items() if len(v)]
    if not keys:
        return float("nan"), float("nan"), float("nan")
    allv = np.concatenate([per_client[k] for k in keys])
    stats = np.empty(B)
    for b in range(B):
        pick = rng.integers(0, len(keys), len(keys))
        parts = [per_client[keys[i]][rng.integers(0, len(per_client[keys[i]]), len(per_client[keys[i]]))] for i in pick]
        stats[b] = np.concatenate(parts).mean()
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(allv.mean()), float(lo), float(hi)


def _pct(x: float) -> str:
    return "n/a" if x != x else f"{100 * x:.1f}"


def report(root: str, data: str, split: str = "test") -> str:
    clients = load(data) if Path(data, "clients.jsonl").exists() else build_dataset()
    qs = {q["qid"]: q for c in clients for q in c.questions if split == "all" or c.split == split}
    runs = load_runs(root)
    readers = sorted({r for r, _ in runs})
    lines = [f"# Results ({split} split, {len(qs)} questions, {len({q['cid'] for q in qs.values()})} clients)", ""]

    def recs(reader, system):
        got = runs.get((reader, system), {})
        return {qid: got[qid] for qid in qs if qid in got}

    for reader in readers:
        systems = [s for s in SYSTEMS if recs(reader, s)]
        lines += [f"## Reader: {reader}", "", "| System | n | Accuracy % (95% CI) | " + " | ".join(CATS) + " | Context tokens |",
                  "|---|---|---|" + "---|" * len(CATS) + "---|"]
        for s in systems:
            r = recs(reader, s)
            per = defaultdict(list)
            for qid, rec in r.items():
                per[rec["cid"]].append(float(rec["score"]["correct"]))
            m, lo, hi = boot_ci({k: np.asarray(v) for k, v in per.items()})
            cats = []
            for cat in CATS:
                v = [rec["score"]["correct"] for rec in r.values() if rec["cat"] == cat]
                cats.append(f"{100 * np.mean(v):.0f}" if v else "-")
            ctx = np.mean([rec["context_tokens"] for rec in r.values()])
            lines.append(f"| {s} | {len(r)} | {_pct(m)} ({_pct(lo)}–{_pct(hi)}) | " + " | ".join(cats) + f" | {ctx:,.0f} |")
        lines += ["", "**Wrong answers by trap** (count of wrong answers that equal a planted wrong value)", "",
                  "| System | stale | initial | hypothetical | advisor_proposal | spouse | misspoke | by_session_year | other trap | abstained | answered_unanswerable | no trap |",
                  "|---|" + "---|" * 11]
        for s in systems:
            r = recs(reader, s)
            c = Counter(rec["score"]["trap"] for rec in r.values() if not rec["score"]["correct"])
            named = ["stale", "initial", "hypothetical", "advisor_proposal", "spouse", "misspoke", "by_session_year"]
            other = sum(v for k, v in c.items() if k not in named + ["abstained", "answered_unanswerable", None])
            lines.append(f"| {s} | " + " | ".join(str(c.get(k, 0)) for k in named) +
                         f" | {other} | {c.get('abstained', 0)} | {c.get('answered_unanswerable', 0)} | {c.get(None, 0)} |")
        lines += ["", hypotheses(reader, recs), "", knowledge_updates(reader, recs, systems), "",
                  empty_answers(reader, recs, systems), ""]
    lines += [mem0_events(root, sorted({q["cid"] for q in qs.values()})), ""]
    return "\n".join(lines)


def _paired(reader, recs, a: str, b: str, pred=lambda q, r: float(r["score"]["correct"]), filt=lambda q: True,
            keep=lambda ra, rb: True) -> str:
    ra, rb = recs(reader, a), recs(reader, b)
    per = defaultdict(list)
    for qid in sorted(set(ra) & set(rb)):
        q = ra[qid]
        if filt(q) and keep(ra[qid], rb[qid]):
            per[q["cid"]].append(pred(q, ra[qid]) - pred(q, rb[qid]))
    if not per:
        return "not run"
    m, lo, hi = boot_ci({k: np.asarray(v) for k, v in per.items()})
    verdict = "no detectable difference" if lo <= 0 <= hi else ("higher" if m > 0 else "lower")
    return f"{a} − {b}: {100 * m:+.1f} pts (95% CI {100 * lo:+.1f} to {100 * hi:+.1f}, n={sum(len(v) for v in per.values())}) → {verdict}"


def hypotheses(reader, recs) -> str:
    out = ["### Pre-registered hypotheses", ""]
    out.append("**H1** (Raw-RAG at least as accurate as agentic memory)")
    for b in ("mem0", "mem0_addonly"):
        out.append(f"- {_paired(reader, recs, 'raw_rag', b)}")
    stale = lambda q, r: float(r["score"]["trap"] == "stale")
    out.append("")
    out.append("**H2** (on knowledge-update questions, Raw-RAG gives a superseded value more often than Mem0)")
    out.append(f"- stale-answer rate, {_paired(reader, recs, 'raw_rag', 'mem0', pred=stale, filt=lambda q: q['cat'] == 'knowledge_update')}")
    out.append("")
    out.append("**H3** (Mem0 add-only at least as accurate as default Mem0)")
    out.append(f"- {_paired(reader, recs, 'mem0_addonly', 'mem0')}")
    out.append("")
    r = recs(reader, "raw_rag")
    wrong = [x for x in r.values() if not x["score"]["correct"] and x["cat"] != "abstention" and "coverage" in x]
    if wrong:
        shown = sum(1 for x in wrong if (x["coverage"]["all_evidence_shown"] if x["cat"] == "aggregation" else x["coverage"]["answer_turn_shown"]))
        out.append(f"**H4** (Raw-RAG misses are mostly use, not coverage): the gold evidence was in the context for "
                   f"{shown} of {len(wrong)} wrong answers ({100 * shown / len(wrong):.0f}%). "
                   "(Aggregation questions count as covered only when every contribution turn was shown.)")
    return "\n".join(out)


def knowledge_updates(reader, recs, systems) -> str:
    """Exploratory: H2's stale rate counts abstentions as not stale, so show what each system did instead."""
    out = ["### Knowledge-update questions in full (exploratory, not pre-registered)", "",
           "| System | n | correct | stale value | other wrong value | abstained | stale share of answers given |",
           "|---|---|---|---|---|---|---|"]
    for s in systems:
        r = [x for x in recs(reader, s).values() if x["cat"] == "knowledge_update"]
        ok = sum(x["score"]["correct"] for x in r)
        stale = sum(x["score"]["trap"] == "stale" for x in r)
        abst = sum(x["score"]["trap"] == "abstained" for x in r)
        given = len(r) - abst
        out.append(f"| {s} | {len(r)} | {ok} | {stale} | {len(r) - ok - stale - abst} | {abst} | "
                   f"{f'{100 * stale / given:.0f}%' if given else 'n/a'} |")
    return "\n".join(out)


def empty_answers(reader, recs, systems) -> str:
    """Exploratory: the pre-registered scoring counts an empty answer field as wrong. Show how many there were and
    whether the hypothesis comparisons change when questions with an empty answer on either side are left out."""
    empty = lambda x: x["score"]["status"] == "no_answer"
    out = ["### Empty answer fields (exploratory, not pre-registered)", "",
           "| System | empty answers | of which dollar amounts | accuracy on the rest % |", "|---|---|---|---|"]
    for s in systems:
        r = list(recs(reader, s).values())
        e = [x for x in r if empty(x)]
        rest = [x for x in r if not empty(x)]
        out.append(f"| {s} | {len(e)} | {sum(x['kind'] == 'money' for x in e)} | "
                   f"{100 * np.mean([x['score']['correct'] for x in rest]):.1f} |")
    both = lambda x, y: not empty(x) and not empty(y)
    out += ["", "Paired comparisons without those questions:", ""]
    for a, b in (("raw_rag", "mem0"), ("raw_rag", "mem0_addonly"), ("mem0_addonly", "mem0"), ("oracle", "raw_rag")):
        out.append(f"- {_paired(reader, recs, a, b, keep=both)}")
    return "\n".join(out)


def mem0_events(root: str, cids: list[str]) -> str:
    """Changes Mem0 applied, summed over the split's clients (Mem0 does not report NONE decisions)."""
    base = Path(root).parent / "memory"
    rows = []
    for name in ("mem0", "mem0_addonly"):
        c = Counter()
        n = 0
        for cid in cids:
            f = base / name / cid / "DONE"
            if f.exists():
                c.update(json.loads(f.read_text(encoding="utf-8")).get("events", {}))
                n += 1
        if n:
            rows.append(f"| {name} | {n} | " + " | ".join(str(c.get(k, 0)) for k in ("ADD", "UPDATE", "DELETE", "failed_messages")) + " |")
    if not rows:
        return ""
    return "\n".join(["## Mem0 write decisions", "", "| Store | clients | ADD | UPDATE | DELETE | messages skipped |",
                      "|---|---|---|---|---|---|"] + rows)
