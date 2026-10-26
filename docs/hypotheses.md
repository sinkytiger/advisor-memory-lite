# Pre-registered hypotheses

Written and committed before any model was run on the test split. The data generator, the question set,
the scoring rules and this file are frozen at that commit; later changes are listed at the end with a reason.

## Background

Raw-RAG (Bao et al., arXiv:2610.30755) reports that dense retrieval plus reranking over raw conversation
history ranks first among compared memory systems in six of seven benchmark-model settings. Its readers are
GPT-4o-mini, GPT-4.1-mini and DeepSeek-V4-Flash. The paper notes that capable readers can resolve dynamic
states from sufficient raw evidence, and that disabling Mem0's UPDATE/DELETE improved accuracy in its tested
configuration.

This study asks whether those findings hold in a setting that stresses dynamic state: client-advisor
conversations in which a client's facts (risk tolerance, income, address, savings plan) change over time,
read by small local models (qwen3:4b, llama3.2:3b).

## Hypotheses

**H1 (replication).** On the test split, Raw-RAG's overall accuracy is at least that of each agentic memory
system (Mem0, Mem0 add-only), for each reader.

**H2 (boundary).** On knowledge-update questions, Raw-RAG answers with a superseded value more often than
Mem0 with UPDATE/DELETE enabled. If H2 holds, the paper's statement about capable readers does not carry
over to small readers in this setting.

**H3 (paper ablation).** Mem0 add-only scores at least as high as default Mem0 overall, as in the paper.

**H4 (evidence).** Raw-RAG's misses are explained more by evidence use than by evidence coverage: among
Raw-RAG's wrong answers, the gold evidence turns were retrieved for most of them.

## Decision rules

* Comparisons are paired by question. A difference counts only if the 95% paired bootstrap interval
  (10,000 resamples of clients, then questions within clients) excludes zero. Otherwise the result is
  reported as "no detectable difference", not as support for either side.
* Every hypothesis is reported, whichever way it comes out.
* The dev split (3 clients) may be used to fix prompts, parsing and settings. The test split (9 clients) is
  run once per configuration; reruns happen only for crashes, never because of results.
* Readers decode greedily (temperature 0), so a configuration is one run; uncertainty comes from the
  question and client sample, not from sampling.

## Fixed settings

* Raw-RAG as in the paper's profiles: 256-token windows (cl100k tokenizer) with 32-token overlap, 128
  dense candidates, cross-encoder rerank to 15, windows expanded to whole turns for display.
* Local substitutes for the paper's API models: nomic-embed-text (Ollama) for embeddings and
  BAAI/bge-reranker-base (ONNX, CPU) for reranking. Mem0 1.0.11, as pinned by the paper's code, with
  telemetry off, its default temperature (0.1), search top-20, batches of 5 messages, and each session's own
  date as "today" in its extraction prompt.
* One memory writer for every LLM-written memory (Mem0, Mem0 add-only, KYC profile): llama3.2:3b. On the dev
  split, qwen3:4b could not follow Mem0's update protocol: once the store held one memory it returned only the
  old memories and dropped every new fact (two sessions of client c01 left one stored memory), with and
  without a JSON schema and with thinking on. llama3.2:3b stored 21 memories from the first session.
* Mem0's two JSON calls are constrained to the shapes its own prompts request (with length caps), because
  qwen3:4b looped until Ollama aborted the request under free-form JSON. A batch that still fails is retried
  message by message, and messages that fail alone are skipped and counted.
* The KYC profile keeps typed values only (a fixed risk or contact category, a dollar amount, a small integer, or
  short text); placeholders such as "unknown" or "null", markup and quoted dialogue from the writer are
  dropped. This rule was added after the dev pilot showed llama3.2:3b writing such values over real ones.
* The same reader prompt and output format for every memory system: JSON with a brief reasoning field (capped
  at 2,000 characters) followed by the answer. Stray braces or tags around an answer are stripped before
  scoring; the raw answer is kept. On the dev pilot a 600-character cap cut off 153 of 161 qwen3:4b reasonings,
  and every malformed qwen answer followed a cut-off, so the cap was raised before registration.

## Changes after registration

(none yet)
