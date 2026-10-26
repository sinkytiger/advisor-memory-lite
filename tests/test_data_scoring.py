import json

import pytest

from advisor_memory.data import CONTACT, RISK, build_dataset, digest, money
from advisor_memory.scoring import is_abstain, labels_in, matches, parse_date, parse_money, score
from advisor_memory.systems import (BUDGET, OVERLAP, WINDOW, Oracle, RawIndex, Recent, answer_turns, n_tokens,
                                    rewrite_add_only)

SHA = "acef72faee713145"


@pytest.fixture(scope="module")
def clients():
    return build_dataset()


def test_dataset_is_deterministic(clients):
    assert digest(clients) == SHA == digest(build_dataset())
    assert [c.split for c in clients].count("dev") == 3 and [c.split for c in clients].count("test") == 9
    assert sum(len(c.questions) for c in clients) == 291


def test_every_answerable_question_has_evidence_and_distinct_traps(clients):
    for c in clients:
        tids = {t.tid for t in c.turns}
        for q in c.questions:
            if q["kind"] == "abstain":
                assert not q["evidence"]
                continue
            assert q["evidence"] and set(q["evidence"]) <= tids, q["qid"]
            assert all(not matches(q["kind"], str(v), q["gold"], q["choices"]) for v in q["traps"].values()), q["qid"]


def test_single_statement_gold_is_printed_in_its_evidence(clients):
    for c in clients:
        text = {t.tid: t.text for t in c.turns}
        for q in c.questions:
            if q["kind"] in ("label", "choice", "name") or (q["kind"] == "money" and q["cat"] != "aggregation"):
                needle = money(int(q["gold"])) if q["kind"] == "money" else q["gold"]
                assert any(needle.lower() in text[t].lower() for t in answer_turns(c, q)), q["qid"]


def test_year_boundary_trap_is_planted(clients):
    for c in clients:
        q = next(q for q in c.questions if "calendar year 2025" in q["text"])
        assert q["traps"]["by_session_year"] < q["gold"]  # the December top-up was reported in a January meeting


@pytest.mark.parametrize("text,value", [("$4,000", 4000), ("4000 dollars", 4000), ("about $4.5k", 4500), ("CAD 12,500.00", 12500)])
def test_parse_money(text, value):
    assert parse_money(text) == value


@pytest.mark.parametrize("text,value", [("2026-03-06", "2026-03-06"), ("March 6, 2026", "2026-03-06"), ("6 March 2026", "2026-03-06"),
                                        ("Mar. 6, 2026", "2026-03-06"), ("March 6th, 2026", "2026-03-06")])
def test_parse_date(text, value):
    assert parse_date(text) == value


def test_labels_match_longest_first():
    assert labels_in("moderately conservative", RISK) == {"moderately conservative"}
    assert labels_in("Conservative.", RISK) == {"conservative"}
    assert labels_in("balanced, maybe growth", RISK) == {"balanced", "growth"}
    assert labels_in("by text", CONTACT) == {"text message"}
    assert labels_in("A phone call works best", CONTACT) == {"phone call"}


def test_scoring_statuses_and_traps(clients):
    q = next(q for c in clients for q in c.questions if q["cat"] == "knowledge_update" and "stale" in q["traps"] and q["kind"] == "money")
    assert score(q, money(int(q["gold"])))["correct"]
    assert score(q, str(q["traps"]["stale"]))["trap"] == "stale"
    assert score(q, "unknown")["status"] == "abstained"
    a = next(q for c in clients for q in c.questions if q["kind"] == "abstain")
    assert score(a, "Not mentioned in the memory.")["correct"]
    assert score(a, "750")["trap"] == "answered_unanswerable"
    assert not is_abstain("$4,000") and is_abstain("I cannot determine that")


def test_raw_windows_follow_the_paper_profile(clients, tmp_path):
    class FakeEmbedder:
        def documents(self, texts):
            import numpy as np
            return np.ones((len(texts), 4), dtype="float32") / 2

    idx = RawIndex(FakeEmbedder())
    turns, lines, windows, emb = idx.build(clients[0], tmp_path)
    assert len(emb) == len(windows) and (tmp_path / "raw_index" / "c01.npy").exists()
    assert all(n_tokens(w[1]) <= WINDOW + 2 for w in windows)
    assert len(windows) * (WINDOW - OVERLAP) >= n_tokens("\n".join(lines))


def test_recent_and_oracle_contexts(clients):
    c = clients[0]
    q = next(q for q in c.questions if q["kind"] == "money")
    r = Recent().context(c, q)
    assert n_tokens(r["text"]) <= BUDGET and r["turn_ids"][-1] == c.turns[-1].tid
    o = Oracle().context(c, q)
    assert set(o["turn_ids"]) == set(q["evidence"])


def test_add_only_rewrite():
    resp = json.dumps({"memory": [{"id": "1", "text": "Risk is growth", "event": "UPDATE", "old_memory": "Risk is balanced"},
                                  {"id": "2", "text": "Lives in Burnaby", "event": "DELETE"},
                                  {"id": "3", "text": "x", "event": "NONE"}, {"text": "new", "event": "ADD"}]})
    out = json.loads(rewrite_add_only(resp))["memory"]
    assert [a["event"] for a in out] == ["ADD", "NONE", "NONE", "ADD"]
    assert all("id" not in a and "old_memory" not in a for a in out)
    assert rewrite_add_only("not json") == "not json"


def test_kyc_profile_keeps_typed_values_only():
    from advisor_memory.systems import build_profile
    ups = [{"meeting": "2025-01-07", "field": "risk_tolerance", "value": "Moderately Conservative", "date": "2025-01-07"},
           {"meeting": "2025-02-02", "field": "risk_tolerance", "value": "unknown", "date": "2025-02-02"},
           {"meeting": "2025-03-01", "field": "risk_tolerance", "value": "MAYA CHEN: Okay, good to know", "date": "x"},
           {"meeting": "2025-03-01", "field": "annual_income", "value": "></strong>109000</strong>", "date": "2025-03-01"},
           {"meeting": "2025-03-01", "field": "employer", "value": "null", "date": "2025-03-01"},
           {"meeting": "2025-04-01", "field": "tfsa_contribution", "value": "$3,000", "date": "2025-03-28"},
           {"meeting": "2025-04-01", "field": "tfsa_contribution", "value": "None", "date": "2025-03-28"},
           {"meeting": "2025-05-01", "field": "children_count", "value": "two", "date": "bad"}]
    p = build_profile(ups)
    assert p["state"]["risk_tolerance"] == [{"value": "moderately conservative", "since": "2025-01-07"}]
    assert p["state"]["annual_income"][0]["value"] == "$109,000" and "employer" not in p["state"]
    assert p["state"]["children_count"][0]["value"] == "2"
    assert p["events"] == [{"field": "tfsa_contribution", "value": "$3,000", "date": "2025-03-28"}]
