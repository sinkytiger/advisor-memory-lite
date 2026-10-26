# advisor-memory-lite

**Does naive RAG still beat agentic memory when a client's facts keep changing?** A small, fully local check
of Raw-RAG ([Bao et al., arXiv:2610.30755](https://arxiv.org/abs/2610.30755)) on synthetic client-advisor
meetings, read by 3-4B models on a 6 GB laptop GPU.

> Status: experiment in progress. Results will be added after the test-split run.
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

The full run takes many hours on a laptop GPU; every step resumes from what is already on disk.

## Results

Pending.

## Credits

Raw-RAG pipeline settings and the Mem0 add-only rewrite follow the paper's code,
[Litmeb/RAW-RAG](https://github.com/Litmeb/RAW-RAG) (MIT). Mem0 is [mem0ai/mem0](https://github.com/mem0ai/mem0)
(Apache-2.0).

## License

MIT
