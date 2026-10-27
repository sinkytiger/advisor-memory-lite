# advisor-memory-lite

**Does naive RAG still beat agentic memory when a client's facts keep changing?** A small, fully local check
of Raw-RAG ([Bao et al., arXiv:2610.30755](https://arxiv.org/abs/2610.30755)) on synthetic client-advisor
meetings, read by 3-4B models on a 6 GB laptop GPU.

**Short answer: yes.** Raw-RAG beat Mem0 by 37 points with llama3.2:3b and by 51 points with qwen3:4b. Mem0
wrote the right facts down, but its search did not bring them back.

> All people, firms and employers in the data are fictional. Nothing here is financial advice.

## Why this setting

Raw-RAG reports that dense retrieval plus reranking over raw conversation history ranks first among compared
memory systems in six of seven benchmark-model settings, and that disabling Mem0's UPDATE/DELETE helped. Its
readers are GPT-4o-mini class or larger, and the paper notes that capable readers can resolve changing states
from raw evidence.

Wealth-advice conversations make that last point hard. Risk tolerance, income, address and savings plans
change over months, advisors make suggestions the client turns down, clients mention a spouse's salary or a
raise they might get, and a December contribution gets mentioned in January. A memory that answers with a
superseded value is a suitability problem, not just a wrong answer.

## What is compared

| System | What the reader gets |
|---|---|
| `raw_rag` | 15 reranked 256-token windows of raw history (the paper's profile: 32-token overlap, 128 candidates) |
| `raw_rag_norerank` | the top 15 windows by embedding similarity |
| `recent` | the latest turns within the same 3,840-token budget |
| `mem0` | Mem0 1.0.11 search results (the version the paper's code pins) |
| `mem0_addonly` | Mem0 with UPDATE→ADD and DELETE→NONE, the paper's ablation |
| `kyc` | a structured know-your-client profile: an LLM extracts field updates, code keeps the history |
| `oracle` | the gold evidence turns (an upper bound) |

Readers: qwen3:4b and llama3.2:3b (Ollama, temperature 0). Memory writer for Mem0 and KYC: llama3.2:3b.
Embeddings: nomic-embed-text. Reranker: BAAI/bge-reranker-base on CPU. Total API cost: $0.

## Data

12 fictional clients, 25 meetings each (about 15-16k tokens of history per client), 291 questions in six
categories: single-session facts, knowledge updates, temporal questions, multi-meeting totals, client decisions
versus advisor proposals, and questions that cannot be answered from the meetings. Every wrong value a question
can tempt (a stale value, the spouse's income, the advisor's proposal, a hypothetical, the wrong calendar year)
is stored, so each wrong answer can be traced to its cause. Details: [docs/design.md](docs/design.md).

The hypotheses and decision rules were written down before the test split was run:
[docs/hypotheses.md](docs/hypotheses.md).

## Run it

Needs Python 3.12, [Ollama](https://ollama.com) with `qwen3:4b`, `llama3.2:3b` and `nomic-embed-text`, and
about 1 GB of disk for the reranker.

```bash
pip install -r requirements.txt
```

```bash
pytest -q
```

```bash
python -m advisor_memory build
```

```bash
bash scripts/run_all.sh
```

The full run takes many hours on a laptop GPU; every step resumes from what is already on disk. It ends by
writing `reports/results.md`. The exploratory context check runs after it:

```bash
python scripts/explore_context.py
```

## Results

Test split: 9 clients, 218 questions, one greedy run per configuration. Accuracy in %, with 95% intervals from
a hierarchical bootstrap (clients, then questions within clients). 45 of the 218 questions cannot be answered
from the meetings, so a memory that always says "unknown" would score 20.6%. Full tables, trap counts and
per-category numbers: [reports/results.md](reports/results.md).

| System | llama3.2:3b | qwen3:4b | Context tokens |
|---|---|---|---|
| `raw_rag` | **66.1** (58.6–73.4) | **82.1** (76.2–87.6) | 4,694 |
| `raw_rag_norerank` | 57.8 (49.8–66.0) | 69.7 (62.2–76.8) | 4,808 |
| `kyc` | 38.5 (29.3–48.2) | 50.5 (41.7–58.9) | 893 |
| `mem0` | 28.9 (22.5–35.7) | 30.7 (24.3–37.5) | 349 |
| `mem0_addonly` | 28.9 (22.8–35.4) | 27.5 (21.3–34.2) | 355 |
| `recent` | 27.1 (20.6–34.0) | 29.4 (23.1–36.2) | 3,710 |
| `oracle` (upper bound) | 79.4 (72.3–85.8) | 83.5 (77.3–89.1) | 96 |

### Pre-registered hypotheses

Paired differences in points, with 95% intervals. A difference counts only when the interval excludes zero.

| Hypothesis | llama3.2:3b | qwen3:4b | Verdict |
|---|---|---|---|
| **H1** Raw-RAG is at least as accurate as Mem0 and Mem0 add-only | +37.2 (+27.7 to +46.7) and +37.2 (+28.1 to +46.3) | +51.4 (+43.2 to +59.2) and +54.6 (+46.8 to +62.2) | Holds for both readers |
| **H2** On knowledge updates, Raw-RAG gives a superseded value more often than Mem0 | +13.4 (+0.0 to +27.3) | +10.4 (+3.0 to +19.1) | Holds for qwen3:4b only, and for the wrong reason (below) |
| **H3** Mem0 add-only is at least as accurate as default Mem0 | +0.0 (−3.2 to +3.2) | −3.2 (−7.1 to +0.0) | No detectable difference |
| **H4** Most Raw-RAG misses had the gold evidence in context | 55 of 66 (83%) | 26 of 38 (68%) | Holds for both readers |

**H2 holds on paper only.** Mem0 hardly answers knowledge-update questions: it says "unknown" to 49 (llama)
and 56 (qwen) of the 67, and an abstention is never a stale answer. Raw-RAG answers 45 and 56 of them
correctly; Mem0 answers 5 and 4. Among the answers each system actually gave, the stale share is about the same
(Raw-RAG 20% and 12%, Mem0 22% and 9%). So the reading registered with H2, that small readers cannot sort out
changing facts from raw history, is not supported: qwen3:4b with Raw-RAG answered 56 of the 67
knowledge-update questions correctly. The stale-answer rate was the wrong measure, because it rewards a memory
that answers nothing.

### What else the run shows (exploratory, not pre-registered)

* **Mem0's bottleneck is search, not writing.** For the 112 questions whose answer is a value stated in one
  meeting, that value appears somewhere in the Mem0 store for 93% of questions (add-only: 96%), but in the top
  20 search results for only 21% (add-only: 19%). Raw-RAG's 15 windows contain it for 98%. After 25 meetings a
  default store holds 337 memories per client on average (276 to 417), and an add-only store 547 (453 to 629).
  These are upper bounds, because a value can also appear in an unrelated memory:
  [reports/context_check.md](reports/context_check.md).
* **Reranking helps.** Raw-RAG minus the same pipeline without the reranker: +12.4 points with qwen3:4b (+5.1
  to +19.9) and +8.3 with llama3.2:3b (+0.0 to +16.7, no detectable difference).
* **A structured profile beats Mem0 but carries its writer's mistakes.** The KYC profile shows the gold value for
  95% of those 112 questions and beats Mem0 with qwen3:4b (+19.7, +10.8 to +28.8). On knowledge updates,
  though, it gives a wrong value that is neither the stale one nor empty for 29 (llama) and 26 (qwen) of the 67
  questions, and almost all of those values (28 and 24) are written in the profile the reader saw. The errors
  come in through the profile's entries, not from the reader making values up.
* **Totals across meetings are the weak spot.** On aggregation questions Raw-RAG scores 11% (llama) and 44%
  (qwen), against the oracle's 72% and 56%.
* **qwen3:4b sometimes leaves the answer empty.** In 101 of its 1,526 answers the answer field was empty or
  held only "}", 99 of them dollar amounts, usually right after the reasoning stated the amount. The dev pilot had
  none (0 of 230). The pre-registered scoring counts these as wrong. Leaving those questions out changes no
  verdict: Raw-RAG minus Mem0 becomes +54.0 points, add-only minus default Mem0 −2.4 (−6.5 to +1.0), and the
  oracle rises to 95.3% on the remaining questions, 9.0 points above Raw-RAG (+2.7 to +15.3).

### Limits

* The meetings are synthetic and template-built, from one generator and seed; 9 test clients. Real advisor
  transcripts are messier.
* The readers are 3-4B local models, and the embedder and reranker are local stand-ins for the paper's. Every
  LLM-written memory used llama3.2:3b as its writer, because qwen3:4b could not follow Mem0's update protocol.
* Mem0 ran with top-20 search as configured; a larger search budget or another embedder may change its numbers.
* One greedy run per configuration, so the intervals reflect the question and client sample only.
* Ollama updated itself mid-run, and the PC lost power twice; both times the run resumed from saved answers.
  Details: [docs/hypotheses.md](docs/hypotheses.md#changes-after-registration).

Reader time: 4.5 hours (llama3.2:3b) and 3.5 hours (qwen3:4b) for 1,526 answers each; building the Mem0 and
KYC memories took most of a night. API cost: $0.

## Credits

Raw-RAG pipeline settings and the Mem0 add-only rewrite follow the paper's code,
[Litmeb/RAW-RAG](https://github.com/Litmeb/RAW-RAG) (MIT). Mem0 is [mem0ai/mem0](https://github.com/mem0ai/mem0)
(Apache-2.0).

## License

MIT
