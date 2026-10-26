# Design

## Question

Raw-RAG (Bao et al., [arXiv:2610.30755](https://arxiv.org/abs/2610.30755), code: [Litmeb/RAW-RAG](https://github.com/Litmeb/RAW-RAG), MIT)
reports that dense retrieval plus reranking over raw conversation history matches or beats complex agentic
memory on long-conversation benchmarks. The readers there are GPT-4o-mini class or larger. This study tests
the claim where it should be hardest: client facts that keep changing, small local models, and wrong answers
that are tempting.

## Data: synthetic client-advisor meetings

Twelve fictional clients of a fictional wealth firm, 25 meetings each over about 18 months (2025-01 to
2026-09), about 15-16k tokens of history per client. Three clients form the dev split (used to fix prompts and
settings), nine the test split (run once per configuration).

Each client has a fact timeline, and every fact-bearing turn is recorded, so every question knows its gold
answer, its evidence turns, and the exact wrong values its traps produce.

| Category | Example | Traps |
|---|---|---|
| single_session | Which sector did the client ask to exclude? | — |
| knowledge_update | What is the client's current risk tolerance on file? | stale value, initial value, a hypothetical ("I might go aggressive, but not now") |
| | What is the client's current annual income? | stale, the spouse's income, a hypothetical raise, a self-corrected figure ("$85,000, sorry, $88,000") |
| temporal | What risk tolerance was on file at the end of May 2025? On what date did the client open their FHSA ("last Friday")? | the current or initial value; the meeting date instead of the resolved date |
| aggregation | How much did the client contribute to their TFSA in calendar year 2025? | a December top-up reported in a January meeting (counted under the wrong year), the total for all years, the account balance |
| decision | How much did the client decide to contribute to their RRSP? | the advisor's proposal |
| abstention | What is the client's credit score? (never said) | answering anything |

Meetings are padded with generic advice (fees, rebalancing, market moves with look-alike percentages) and
small talk, so most of the history is not about the facts asked. Account names (TFSA, RRSP, FHSA) appear only
as names: no gold answer depends on real contribution limits or rules.

## Memory systems

| System | What the reader sees | LLM writes memory? |
|---|---|---|
| raw_rag | 15 reranked 256-token windows of raw history, shown as whole turns | no |
| raw_rag_norerank | the top 15 windows by embedding similarity | no |
| recent | the most recent turns within the same 3,840-token budget | no |
| mem0 | Mem0 1.0.11 search results (top 20) with their meeting dates | yes |
| mem0_addonly | the same, with UPDATE turned into ADD and DELETE into NONE (the paper's ablation) | yes |
| kyc | a structured KYC profile: an LLM extracts field updates per meeting, code keeps the history | yes |
| oracle | the gold evidence turns | no |

The KYC profile keeps typed values only: a fixed risk or contact category, a dollar amount, a small integer,
or short text. Placeholders ("unknown", "null"), markup and quoted dialogue are dropped, so a writer's
formatting slips cannot overwrite a real value; its factual mistakes are kept.

The KYC profile is not a neutral competitor. Its fields and its instruction to skip advisor suggestions and
hypotheticals were written with the question types in mind. It shows what domain structure can buy when the
designer knows what will be asked.

## Models (all local, $0)

* Readers: qwen3:4b (thinking off) and llama3.2:3b through Ollama, temperature 0, 8,192-token context, the
  same prompt and JSON answer format for every system.
* Memory writer for Mem0 and KYC: llama3.2:3b (see the hypotheses file for why not qwen3:4b).
* Embeddings: nomic-embed-text with its search_document / search_query prefixes, for Raw-RAG and Mem0 alike.
* Reranker: BAAI/bge-reranker-base (MIT) on CPU through fastembed.
* The paper used text-embedding-3-small with Cohere Rerank 3.5, or Qwen3-Embedding-4B with Qwen3-Reranker-4B.
  Neither fits the hardware budget, so absolute numbers are not comparable with the paper's.

## Scoring

No LLM judge. Answers are parsed by type (dollar amount, integer, date, closed-set label, name) and compared
exactly with the gold value. A wrong answer that equals a planted trap value is attributed to that trap. An
"unknown" on an answerable question counts as wrong (status `abstained`); any value on an unanswerable question
counts as wrong (`answered_unanswerable`).

## Deviations from the paper's setup

* Local embedding, reranker and readers instead of API models.
* Mem0's JSON calls are schema-constrained, and failed batches are retried message by message.
* Mem0 sees each meeting's date as "today" in its extraction prompt (the stock prompt bakes in the wall-clock
  date), and stores each memory with the meeting date as `created_at`, as the paper's adapter does.
* Exact-match scoring replaces the paper's GPT-4o-mini judge, which is possible because every gold answer is
  generated.
