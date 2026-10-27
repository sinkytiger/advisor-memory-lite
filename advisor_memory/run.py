"""Run loop: build each memory once per client, read every question with the same prompt, score, cache.

    python -m advisor_memory build
    python -m advisor_memory run --reader qwen3:4b --split dev --out runs/main
    python -m advisor_memory report --runs runs/main --out reports/results.md
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

from .data import FIRM, build_dataset, digest, load, save
from .models import Chat, ModelUnavailable
from .scoring import score
from .systems import SYSTEMS, answer_turns, build_systems, n_tokens

READER_SYSTEM = (f"You answer questions about a client of {FIRM} for the client's financial advisor, using only the "
                 "memory provided. Today's date is {today}.")
READER_USER = """Memory (excerpts from past meetings, with dates):
{context}

Question: {question}

Instructions:
1. Read all of the memory before answering, and combine entries when the question needs it.
2. Pay attention to dates. If entries disagree, prefer the most recent one, unless the question asks about an earlier time.
3. Convert relative time references ("last Friday", "yesterday") into dates using the date of the entry.
4. If the memory does not contain the answer, answer "unknown".
5. Keep the answer short: a dollar amount as a number, a date as YYYY-MM-DD, or a few words.
6. Keep the reasoning brief (under 100 words)."""
ANSWER_SCHEMA = {"type": "object", "properties": {"reasoning": {"type": "string", "maxLength": 2000},
                                                  "answer": {"type": "string", "maxLength": 120}},
                 "required": ["reasoning", "answer"]}
# Every memory that needs an LLM to write it uses this model, whatever the reader. qwen3:4b cannot follow Mem0's
# update protocol: once the store is non-empty it returns only the old memories and drops every new fact
# (dev probe, client c01: one memory kept from two sessions). llama3.2:3b follows it.
WRITER = "llama3.2:3b"


def clean_answer(ans):
    """Strip stray JSON or tag characters a small model sometimes wraps around the answer ('}$12,500{')."""
    if not isinstance(ans, str):
        return ans
    ans = re.sub(r"(?:\s*(?:</?\w*>|assistant))+\s*$", "", ans.strip())
    return ans.strip(" \t\n{}<>/:,") or None


def read(chat: Chat, client, q: dict, ctx: str) -> tuple[str | None, dict, str]:
    out, usage = chat.json(READER_SYSTEM.format(today=client.today),
                           READER_USER.format(context=ctx, question=q["text"]), ANSWER_SCHEMA, max_tokens=900)
    raw = out.get("answer") if isinstance(out, dict) else None
    usage["answer_raw"] = raw
    return clean_answer(raw), usage, str(out.get("reasoning", ""))[:2000]


def run(args) -> int:
    clients = load(args.data) if Path(args.data, "clients.jsonl").exists() else build_dataset()
    clients = [c for c in clients if args.split == "all" or c.split == args.split]
    if args.clients:
        clients = [c for c in clients if c.cid in args.clients]
    work = Path(args.work)
    systems = build_systems(args.systems, WRITER)
    chat = Chat(args.reader)
    out = Path(args.out) / args.reader.replace(":", "_")
    t0 = time.time()
    n = ok = 0
    for c in clients:
        for name, sys_ in systems.items():
            d = out / name
            d.mkdir(parents=True, exist_ok=True)
            todo = [q for q in c.questions if not (d / f"{q['qid']}.json").exists()]
            if not todo:
                continue
            ts = time.time()
            try:
                sys_.prepare(c, work)
            except ModelUnavailable as exc:
                print(f"model unavailable while building {name} for {c.cid}: {exc}; stopping (finished work is kept)")
                return 2
            if time.time() - ts > 5:
                print(f"[{name}] built memory for {c.cid} in {time.time() - ts:.0f}s", flush=True)
            for q in todo:
                ctx = sys_.context(c, q)
                try:
                    ans, usage, why = read(chat, c, q, ctx["text"])
                except ModelUnavailable as exc:
                    print(f"model unavailable: {exc}; stopping (finished answers are kept)")
                    return 2
                s = score(q, ans)
                rec = {"qid": q["qid"], "cid": c.cid, "cat": q["cat"], "kind": q["kind"], "system": name, "reader": args.reader,
                       "answer": ans, "gold": q["gold"], "score": s, "reasoning": why, "usage": usage,
                       "context_tokens": n_tokens(ctx["text"])}
                if ctx["turn_ids"] is not None:
                    shown = set(ctx["turn_ids"])
                    gold = set(answer_turns(c, q))
                    rec["coverage"] = {"answer_turn_shown": bool(gold & shown) if gold else None,
                                       "all_evidence_shown": set(q["evidence"]) <= shown if q["evidence"] else None}
                (d / f"{q['qid']}.json").write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
                n += 1
                ok += s["correct"]
                if args.verbose:
                    print(f"[{name}] {q['qid']} {q['cat']:16s} {'OK' if s['correct'] else 'x '} ans={ans!r} gold={q['gold']!r} "
                          f"trap={s['trap']} ctx={rec['context_tokens']}", flush=True)
            done = [json.loads((d / f"{q['qid']}.json").read_text(encoding="utf-8")) for q in c.questions]
            print(f"[{name}] {c.cid} done ({sum(r['score']['correct'] for r in done)}/{len(done)} correct) "
                  f"{time.time() - t0:.0f}s elapsed", flush=True)
    print(f"{n} new answers, {ok} correct, {time.time() - t0:.0f}s")
    return 0


def _prepare_job(job: tuple[str, str, str, str]) -> str:
    name, cid, data, work = job
    clients = load(data) if Path(data, "clients.jsonl").exists() else build_dataset()
    c = next(c for c in clients if c.cid == cid)
    t = time.time()
    build_systems([name], WRITER)[name].prepare(c, Path(work))
    return f"[{name}] {cid} memory built in {time.time() - t:.0f}s"


def prepare(args) -> int:
    """Build LLM-written memories ahead of the readers, several clients at a time (Ollama serves the
    requests in parallel, which roughly doubles throughput on a small GPU)."""
    from concurrent.futures import ProcessPoolExecutor, as_completed
    clients = load(args.data) if Path(args.data, "clients.jsonl").exists() else build_dataset()
    clients = [c for c in clients if (args.split == "all" or c.split == args.split) and (not args.clients or c.cid in args.clients)]
    jobs = [(n, c.cid, args.data, args.work) for n in args.systems for c in clients]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(_prepare_job, j) for j in jobs]
        for f in as_completed(futs):
            try:
                print(f.result(), flush=True)
            except ModelUnavailable as exc:
                print(f"model unavailable: {exc}; rerun prepare to resume")
                return 2
    return 0


def rerank(args) -> int:
    """Precompute Raw-RAG's reranked windows on the CPU while a reader uses the GPU."""
    clients = load(args.data) if Path(args.data, "clients.jsonl").exists() else build_dataset()
    clients = [c for c in clients if args.split == "all" or c.split == args.split]
    rag = build_systems(["raw_rag"], WRITER)["raw_rag"]
    for c in clients:
        rag.prepare(c, Path(args.work))
        t = time.time()
        for q in c.questions:
            rag.order(c, q)
        print(f"[rerank] {c.cid} {len(c.questions)} questions in {time.time() - t:.0f}s", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="advisor_memory", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("build", help="generate the synthetic conversations and questions")
    sp.add_argument("--seed", type=int, default=5)
    sp.add_argument("--out", default="data/dataset")
    sp = sub.add_parser("run", help="build memories and answer questions")
    sp.add_argument("--data", default="data/dataset")
    sp.add_argument("--work", default="runs/memory", help="memory stores, shared by every reader")
    sp.add_argument("--out", default="runs/main")
    sp.add_argument("--reader", default="qwen3:4b")
    sp.add_argument("--systems", nargs="+", default=SYSTEMS, choices=SYSTEMS)
    sp.add_argument("--split", default="dev", choices=["dev", "test", "all"])
    sp.add_argument("--clients", nargs="*")
    sp.add_argument("-v", "--verbose", action="store_true")
    sp = sub.add_parser("prepare", help="build LLM-written memories in parallel before reading")
    sp.add_argument("--data", default="data/dataset")
    sp.add_argument("--work", default="runs/memory")
    sp.add_argument("--systems", nargs="+", default=["mem0", "mem0_addonly", "kyc"], choices=["mem0", "mem0_addonly", "kyc"])
    sp.add_argument("--split", default="test", choices=["dev", "test", "all"])
    sp.add_argument("--clients", nargs="*")
    sp.add_argument("--workers", type=int, default=3)
    sp = sub.add_parser("rerank", help="precompute Raw-RAG's reranked windows (CPU)")
    sp.add_argument("--data", default="data/dataset")
    sp.add_argument("--work", default="runs/memory")
    sp.add_argument("--split", default="test", choices=["dev", "test", "all"])
    sp = sub.add_parser("report", help="markdown tables from run directories")
    sp.add_argument("--runs", default="runs/main")
    sp.add_argument("--data", default="data/dataset")
    sp.add_argument("--split", default="test", choices=["dev", "test", "all"])
    sp.add_argument("--out")
    args = p.parse_args(argv)
    if args.cmd == "build":
        clients = build_dataset(args.seed)
        save(clients, args.out)
        print(f"{len(clients)} clients, {sum(len(c.questions) for c in clients)} questions, sha {digest(clients)} -> {args.out}")
        return 0
    if args.cmd == "run":
        return run(args)
    if args.cmd == "prepare":
        return prepare(args)
    if args.cmd == "rerank":
        return rerank(args)
    from .report import report
    text = report(args.runs, args.data, args.split)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
