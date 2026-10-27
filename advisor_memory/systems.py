"""Memory systems. Each builds a per-client store once and returns a context for a question.

* raw_rag / raw_rag_norerank: the Raw-RAG pipeline (Bao et al., arXiv:2610.30755) with the paper's profile:
  256-token windows, 32-token overlap, 128 dense candidates, rerank to 15, windows shown as whole turns.
* recent: the most recent turns that fit the same token budget (a "just keep the latest" memory).
* oracle: the gold evidence turns; an upper bound on what the reader can do with perfect retrieval.
* mem0 / mem0_addonly: Mem0 1.0.11 (the version pinned by the paper's code), default and with UPDATE turned
  into ADD and DELETE into NONE, as in the paper's ablation.
* kyc: a structured know-your-client profile. Its fields and its rule to skip suggestions and hypotheticals
  were written with knowledge of the question types, so it shows what domain structure can buy, not a
  neutral competitor.
"""

from __future__ import annotations

import contextvars
import json
import os
import re
import shutil
from pathlib import Path

import numpy as np
import tiktoken

from .data import FIRM, Client, money
from .models import EMBED_MODEL, HOST, NUM_CTX, Chat, Embedder, Reranker, thinks

ENC = tiktoken.get_encoding("cl100k_base")
WINDOW, OVERLAP, CANDIDATES, SELECTED = 256, 32, 128, 15
BUDGET = WINDOW * SELECTED
MEM0_BATCH, MEM0_TOP_K = 5, 20
# Mem0 asks for free-form JSON ("format": "json"); qwen3:4b sometimes loops until Ollama aborts. These schemas
# are the shapes Mem0's own prompts request, with length caps, so a small model cannot run away.
MEM0_FACTS = {"type": "object", "properties": {"facts": {"type": "array", "maxItems": 20,
                                                         "items": {"type": "string", "maxLength": 200}}}, "required": ["facts"]}
MEM0_ACTIONS = {"type": "object", "properties": {"memory": {"type": "array", "maxItems": 40, "items": {
    "type": "object", "properties": {"id": {"type": "string", "maxLength": 40}, "text": {"type": "string", "maxLength": 300},
                                     "event": {"type": "string", "enum": ["ADD", "UPDATE", "DELETE", "NONE"]},
                                     "old_memory": {"type": "string", "maxLength": 300}},
    "required": ["id", "text", "event"]}}}, "required": ["memory"]}


def n_tokens(text: str) -> int:
    return len(ENC.encode(text))


class System:
    name = ""
    turn_level = True   # whether the context can be traced back to turn ids

    def prepare(self, client: Client, work: Path) -> None:
        pass

    def context(self, client: Client, q: dict) -> dict:
        raise NotImplementedError


# ---------------------------------------------------------------- raw history

class RawIndex:
    """Token windows over one client's serialized history, embedded once and cached on disk."""

    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder
        self.data: dict[str, tuple] = {}

    def build(self, client: Client, work: Path) -> tuple:
        if client.cid in self.data:
            return self.data[client.cid]
        turns = client.turns
        lines = [t.line() for t in turns]
        toks, owner = [], []
        for i, line in enumerate(lines):
            ids = ENC.encode(line + "\n")
            toks += ids
            owner += [i] * len(ids)
        windows = []
        for a in range(0, len(toks), WINDOW - OVERLAP):
            b = min(a + WINDOW, len(toks))
            windows.append((sorted(set(owner[a:b])), ENC.decode(toks[a:b])))
            if b == len(toks):
                break
        cache = work / "raw_index" / f"{client.cid}.npy"
        emb = np.load(cache) if cache.exists() else None
        if emb is None or len(emb) != len(windows):
            emb = self.embedder.documents([w[1] for w in windows])
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache, emb)
        self.data[client.cid] = (turns, lines, windows, emb)
        return self.data[client.cid]


class RawRAG(System):
    def __init__(self, index: RawIndex, reranker: Reranker | None) -> None:
        self.index = index
        self.reranker = reranker
        self.name = "raw_rag" if reranker else "raw_rag_norerank"

    def prepare(self, client: Client, work: Path) -> None:
        self.index.build(client, work)
        self.work = work

    def order(self, client: Client, q: dict) -> list[int]:
        """The selected window ids. Retrieval does not depend on the reader, so the reranked order is cached
        per question and shared by every reader (the CPU cross-encoder is the slow step)."""
        cache = self.work / ("rerank" if self.reranker else "dense") / client.cid / f"{q['qid']}.json"
        if cache.exists():
            try:
                return json.loads(cache.read_text(encoding="utf-8"))
            except ValueError:
                pass   # a power loss can leave the file full of NUL bytes; recompute it (the order is deterministic)
        turns, lines, windows, emb = self.index.data[client.cid]
        sims = emb @ self.index.embedder.query(q["text"])
        cand = [int(i) for i in np.argsort(-sims)[:CANDIDATES]]
        if self.reranker:
            sc = np.asarray(self.reranker.scores(q["text"], [windows[i][1] for i in cand]))
            order = [cand[j] for j in np.argsort(-sc, kind="stable")[:SELECTED]]
        else:
            order = cand[:SELECTED]
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(order), encoding="utf-8")
        os.replace(tmp, cache)   # atomic, so a parallel precompute and a reader never see half a file
        return order

    def context(self, client: Client, q: dict) -> dict:
        turns, lines, windows, emb = self.index.data[client.cid]
        order = self.order(client, q)
        # shown in rank order, each window expanded to its whole turns (no de-duplication, the paper's default)
        blocks = ["\n".join(lines[j] for j in windows[i][0]) for i in order]
        text = "\n\n".join(f"[Memory {k + 1}]\n{b}" for k, b in enumerate(blocks))
        shown = sorted({turns[j].tid for i in order for j in windows[i][0]})
        return {"text": text, "turn_ids": shown}


class Recent(System):
    name = "recent"

    def context(self, client: Client, q: dict) -> dict:
        picked, used = [], 0
        for t in reversed(client.turns):
            k = n_tokens(t.line()) + 1
            if used + k > BUDGET:
                break
            picked.append(t)
            used += k
        picked.reverse()
        return {"text": "\n".join(t.line() for t in picked), "turn_ids": [t.tid for t in picked]}


class Oracle(System):
    name = "oracle"

    def context(self, client: Client, q: dict) -> dict:
        ev = set(q["evidence"])
        picked = [t for t in client.turns if t.tid in ev]
        text = "\n".join(t.line() for t in picked) or "(no conversation excerpt mentions this)"
        return {"text": text, "turn_ids": [t.tid for t in picked]}


# ---------------------------------------------------------------- Mem0

os.environ.setdefault("MEM0_TELEMETRY", "False")   # keep runs offline; Mem0 otherwise reports to PostHog
_SESSION_DATE: contextvars.ContextVar[str | None] = contextvars.ContextVar("session_date", default=None)


def _patch_mem0_prompt_date() -> None:
    """Mem0 bakes the wall-clock date into its extraction prompt at import. Replaying a 2025-26 history,
    each session must see its own date as 'today', as a live deployment would."""
    import mem0.memory.main as mm
    if getattr(mm, "_advisor_memory_patched", False):
        return
    orig = mm.get_fact_retrieval_messages

    def patched(message, is_agent_memory=False):
        system, user = orig(message, is_agent_memory)
        d = _SESSION_DATE.get()
        if d:
            system = re.sub(r"Today's date is \d{4}-\d{2}-\d{2}", f"Today's date is {d}", system)
        return system, user

    mm.get_fact_retrieval_messages = patched
    mm._advisor_memory_patched = True


def rewrite_add_only(response):
    """UPDATE -> ADD and DELETE -> NONE on Mem0's second LLM response, before Mem0 acts on it.
    Adapted from the paper's Mem0 adapter (Litmeb/RAW-RAG, MIT)."""
    if isinstance(response, str):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", response.strip(), flags=re.I)
        try:
            payload = json.loads(text)
        except (TypeError, ValueError):
            return response
        serialize = True
    elif isinstance(response, dict):
        payload, serialize = response, False
    else:
        return response
    actions = payload.get("memory") if isinstance(payload, dict) else None
    if not isinstance(actions, list):
        return response
    out = []
    for a in actions:
        if not isinstance(a, dict):
            out.append(a)
            continue
        a = dict(a)
        ev = str(a.get("event", "")).upper()
        if ev in ("UPDATE", "DELETE") or (ev == "NONE" and a.get("id") is not None):
            a["event"] = {"UPDATE": "ADD", "DELETE": "NONE", "NONE": "NONE"}[ev]
            a.pop("id", None)          # an ADD has no target, and a NONE must not touch any stored memory
            a.pop("old_memory", None)
        out.append(a)
    payload = dict(payload, memory=out)
    return json.dumps(payload) if serialize else payload


class Mem0System(System):
    turn_level = False

    def __init__(self, writer: str, add_only: bool = False) -> None:
        self.writer = writer
        self.add_only = add_only
        self.name = "mem0_addonly" if add_only else "mem0"
        self.mems: dict = {}

    def _open(self, d: Path):
        from mem0 import Memory
        _patch_mem0_prompt_date()
        cfg = {"llm": {"provider": "ollama", "config": {"model": self.writer, "temperature": 0.1, "max_tokens": 2000,
                                                        "ollama_base_url": HOST}},
               "embedder": {"provider": "ollama", "config": {"model": EMBED_MODEL, "ollama_base_url": HOST, "embedding_dims": 768}},
               "vector_store": {"provider": "qdrant", "config": {"collection_name": "memories", "path": str(d / "qdrant"),
                                                                 "on_disk": True, "embedding_model_dims": 768}},
               "history_db_path": str(d / "history.db"), "version": "v1.1"}
        m = Memory.from_config(cfg)
        chat = m.llm.client.chat

        def local_chat(**kw):
            kw["options"] = {**(kw.get("options") or {}), "num_ctx": NUM_CTX, "seed": 7}
            if kw.get("format") == "json":
                text = " ".join(str(x.get("content", "")) for x in kw.get("messages", []))
                kw["format"] = MEM0_ACTIONS if "smart memory manager" in text else MEM0_FACTS
            if thinks(self.writer):
                kw["think"] = False
            return chat(**kw)

        m.llm.client.chat = local_chat
        embed = m.embedding_model.embed

        def prefixed(text, memory_action=None):   # the same nomic task prefixes Raw-RAG uses
            return embed(("search_query: " if memory_action == "search" else "search_document: ") + text, memory_action)

        m.embedding_model.embed = prefixed
        if self.add_only:
            gen = m.llm.generate_response
            m.llm.generate_response = lambda *a, **k: rewrite_add_only(gen(*a, **k))
        return m

    def prepare(self, client: Client, work: Path) -> None:
        d = work / self.name / client.cid
        if (d / "DONE").exists():
            self.mems[client.cid] = self._open(d)
            return
        if d.exists():
            shutil.rmtree(d)   # a partial store is rebuilt from scratch: Mem0 adds are not idempotent
        d.mkdir(parents=True)
        m = self._open(d)
        events: dict[str, int] = {}
        with open(d / "add_log.jsonl", "w", encoding="utf-8") as log:
            for session in client.sessions:
                tok = _SESSION_DATE.set(session[0].date)
                try:
                    for i in range(0, len(session), MEM0_BATCH):
                        batch = session[i:i + MEM0_BATCH]
                        msgs = [{"role": "user" if t.speaker == "client" else "assistant", "content": f"{t.name}: {t.text}"}
                                for t in batch]
                        meta = {"created_at": f"{batch[0].date}T12:00:00+00:00"}
                        results, failed = self._add(m, client.cid, msgs, meta)
                        for r in results:
                            events[r.get("event", "?")] = events.get(r.get("event", "?"), 0) + 1
                        events["failed_messages"] = events.get("failed_messages", 0) + failed
                        log.write(json.dumps({"turns": [t.tid for t in batch], "results": results, "failed": failed},
                                             ensure_ascii=False) + "\n")
                finally:
                    _SESSION_DATE.reset(tok)
        (d / "DONE").write_text(json.dumps({"events": events}), encoding="utf-8")
        self.mems[client.cid] = m

    @staticmethod
    def _add(m, uid: str, msgs: list[dict], meta: dict) -> tuple[list, int]:
        """Add a batch; if the model call fails twice, add its messages one by one, as the paper's adapter
        splits a batch that times out. Messages that still fail are skipped and counted."""
        from ollama import ResponseError
        for _ in range(2):
            try:
                res = m.add(msgs, user_id=uid, metadata=meta)
                return (res.get("results", []) if isinstance(res, dict) else res), 0
            except (ResponseError, ValueError, KeyError):
                continue
        if len(msgs) == 1:
            return [], 1
        out, failed = [], 0
        for one in msgs:
            r, f = Mem0System._add(m, uid, [one], meta)
            out += r
            failed += f
        return out, failed

    def context(self, client: Client, q: dict) -> dict:
        res = self.mems[client.cid].search(q["text"], user_id=client.cid, limit=MEM0_TOP_K)
        items = res.get("results", []) if isinstance(res, dict) else res
        lines = [f"[{(it.get('created_at') or '')[:10]}] {it.get('memory', '')}" for it in items]
        return {"text": "\n".join(lines) or "(no memories)", "turn_ids": None}


# ---------------------------------------------------------------- structured KYC profile

STATE_FIELDS = ["risk_tolerance", "annual_income", "employer", "city", "monthly_savings", "contact_preference",
                "children_count", "spouse_name", "home_price_target", "retirement_age_target", "excluded_sector"]
EVENT_FIELDS = ["tfsa_contribution", "rrsp_contribution", "account_opened", "child_born"]
KYC_SCHEMA = {"type": "object", "properties": {"updates": {"type": "array", "items": {
    "type": "object", "properties": {"field": {"type": "string", "enum": STATE_FIELDS + EVENT_FIELDS},
                                     "value": {"type": "string", "maxLength": 80},
                                     "date": {"type": "string", "maxLength": 10}},
    "required": ["field", "value", "date"]}}}, "required": ["updates"]}
KYC_SYSTEM = (f"You maintain the know-your-client (KYC) profile that a financial advisor at {FIRM} keeps for a client. "
              "You read one meeting transcript at a time and list the profile updates it contains.")
KYC_USER = """Meeting date: {date}

Transcript:
{transcript}

List every profile update from this meeting as JSON. Fields: {fields}.
Rules:
- Record a value only when the client states it as a fact or confirms it as a decision. Do not record the advisor's suggestions, hypotheticals, or facts about other people (except spouse_name).
- State fields (risk_tolerance, annual_income, ...) hold the client's value as of this meeting. Use plain values: amounts as numbers in dollars, risk tolerance as one of conservative, moderately conservative, balanced, growth, aggressive.
- Event fields (tfsa_contribution, rrsp_contribution, account_opened, child_born) record one event each: the amount, account name or child's name as the value, and the date it happened.
- Turn relative dates ("last Friday", "yesterday") into YYYY-MM-DD using the meeting date. If a date is not known, use the meeting date.
- If nothing in the meeting changes the profile, return an empty list."""


_PLACEHOLDER = re.compile(r"^(?:unknown|null|none|n/?a|nil|no|tbd|not\s+\w+(?:\s+\w+)?|-+|\?+)$", re.I)
_MONEY_FIELDS = {"annual_income", "monthly_savings", "home_price_target", "tfsa_contribution", "rrsp_contribution"}
_INT_FIELDS = {"children_count": (0, 12), "retirement_age_target": (40, 90)}


def _valid(field: str, raw) -> str | None:
    """A profile keeps typed values only: a fixed category, a dollar amount, a small integer, or short text.
    Placeholders ("unknown", "null"), markup and quoted dialogue are dropped."""
    from .data import CONTACT, RISK
    from .scoring import labels_in, parse_int, parse_money
    v = re.sub(r"<[^>]*>", "", str(raw if raw is not None else "")).strip(" \t\"'`<>*.")
    if not v or _PLACEHOLDER.match(v) or re.search(r"\b(?:unknown|null|none)\b", v, re.I):
        return None
    if field in ("risk_tolerance", "contact_preference"):
        found = labels_in(v, RISK if field == "risk_tolerance" else CONTACT)
        return found.pop() if len(found) == 1 else None
    if field in _MONEY_FIELDS:
        m = parse_money(v)
        return money(int(round(m))) if m and m > 0 else None
    if field in _INT_FIELDS:
        n = parse_int(v)
        lo, hi = _INT_FIELDS[field]
        return str(n) if n is not None and lo <= n <= hi else None
    if ":" in v or len(v) > 60 or "advisor" in v.lower():
        return None
    return v


def build_profile(updates: list[dict]) -> dict:
    prof = {"state": {}, "events": [], "updates": updates}
    for u in updates:
        fld, meeting = u.get("field"), u.get("meeting")
        val = _valid(fld, u.get("value")) if fld in STATE_FIELDS + EVENT_FIELDS else None
        if val is None:
            continue
        if fld in EVENT_FIELDS:
            when = str(u.get("date") or "")[:10]
            when = when if re.fullmatch(r"\d{4}-\d{2}-\d{2}", when) else meeting
            prof["events"].append({"field": fld, "value": val, "date": when})
        else:
            hist = prof["state"].setdefault(fld, [])
            if not hist or hist[-1]["value"].lower() != val.lower():
                hist.append({"value": val, "since": meeting})   # a state holds as of the meeting that stated it
    return prof


class KYCProfile(System):
    turn_level = False
    name = "kyc"

    def __init__(self, writer: str) -> None:
        self.writer = writer
        self.chat = None
        self.profiles: dict[str, dict] = {}

    def prepare(self, client: Client, work: Path) -> None:
        f = work / "kyc" / f"{client.cid}.json"
        if f.exists():
            # the raw extractions are kept, so the profile is rebuilt with the current validation rules
            self.profiles[client.cid] = build_profile(json.loads(f.read_text(encoding="utf-8"))["updates"])
            return
        self.chat = self.chat or Chat(self.writer)
        updates = []
        for session in client.sessions:
            date = session[0].date
            out, _ = self.chat.json(KYC_SYSTEM, KYC_USER.format(date=date, transcript="\n".join(t.line() for t in session),
                                                                fields=", ".join(STATE_FIELDS + EVENT_FIELDS)),
                                    KYC_SCHEMA, max_tokens=800)
            updates += [{"meeting": date, **u} for u in (out.get("updates", []) if isinstance(out, dict) else [])
                        if isinstance(u, dict)]
        prof = build_profile(updates)
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(prof, ensure_ascii=False, indent=1), encoding="utf-8")
        self.profiles[client.cid] = prof

    def context(self, client: Client, q: dict) -> dict:
        p = self.profiles[client.cid]
        lines = [f"KYC profile of {client.name}"]
        for fld in STATE_FIELDS:
            hist = p["state"].get(fld)
            if hist:
                cur = hist[-1]
                prev = "; ".join(f"{h['value']} (from {h['since']})" for h in hist[:-1])
                lines.append(f"- {fld}: {cur['value']} (since {cur['since']})" + (f". Earlier: {prev}" if prev else ""))
        if p["events"]:
            lines.append("Events:")
            lines += [f"- [{e['date']}] {e['field']}: {e['value']}" for e in sorted(p["events"], key=lambda e: e["date"])]
        return {"text": "\n".join(lines), "turn_ids": None}


def answer_turns(client: Client, q: dict) -> list[str]:
    """The evidence turns that state the gold value (all evidence turns when no single turn does)."""
    ev = set(q["evidence"])
    if q["kind"] == "money":
        needle = money(int(q["gold"]))
    elif q["kind"] in ("label", "choice", "name"):
        needle = str(q["gold"])
    else:
        needle = None
    hits = [t.tid for t in client.turns if t.tid in ev and needle and needle.lower() in t.text.lower()]
    return hits or sorted(ev)


def build_systems(names: list[str], writer: str) -> dict[str, System]:
    out: dict[str, System] = {}
    index = None
    reranker = None
    for n in names:
        if n in ("raw_rag", "raw_rag_norerank"):
            index = index or RawIndex(Embedder())
            if n == "raw_rag":
                reranker = reranker or Reranker()
            out[n] = RawRAG(index, reranker if n == "raw_rag" else None)
        elif n == "recent":
            out[n] = Recent()
        elif n == "oracle":
            out[n] = Oracle()
        elif n in ("mem0", "mem0_addonly"):
            out[n] = Mem0System(writer, add_only=n == "mem0_addonly")
        elif n == "kyc":
            out[n] = KYCProfile(writer)
        else:
            raise ValueError(f"unknown system {n}")
    return out


SYSTEMS = ["raw_rag", "raw_rag_norerank", "recent", "mem0", "mem0_addonly", "kyc", "oracle"]
