# Results (test split, 218 questions, 9 clients)

## Reader: llama3.2:3b

| System | n | Accuracy % (95% CI) | single_session | knowledge_update | temporal | aggregation | decision | abstention | Context tokens |
|---|---|---|---|---|---|---|---|---|---|
| raw_rag | 218 | 66.1 (58.6–73.4) | 85 | 67 | 44 | 11 | 100 | 82 | 4,694 |
| raw_rag_norerank | 218 | 57.8 (49.8–66.0) | 63 | 52 | 42 | 22 | 67 | 89 | 4,808 |
| recent | 218 | 27.1 (20.6–34.0) | 4 | 19 | 2 | 0 | 6 | 96 | 3,710 |
| mem0 | 218 | 28.9 (22.5–35.7) | 22 | 7 | 2 | 11 | 28 | 98 | 349 |
| mem0_addonly | 218 | 28.9 (22.8–35.4) | 19 | 9 | 7 | 0 | 28 | 98 | 355 |
| kyc | 218 | 38.5 (29.3–48.2) | 37 | 39 | 33 | 0 | 28 | 64 | 893 |
| oracle | 218 | 79.4 (72.3–85.8) | 70 | 81 | 56 | 72 | 100 | 100 | 96 |

**Wrong answers by trap** (count of wrong answers that equal a planted wrong value)

| System | stale | initial | hypothetical | advisor_proposal | spouse | misspoke | by_session_year | other trap | abstained | answered_unanswerable | no trap |
|---|---|---|---|---|---|---|---|---|---|---|---|
| raw_rag | 13 | 3 | 2 | 0 | 1 | 0 | 0 | 16 | 6 | 8 | 25 |
| raw_rag_norerank | 22 | 4 | 1 | 2 | 0 | 0 | 1 | 18 | 12 | 5 | 27 |
| recent | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 8 | 120 | 2 | 28 |
| mem0 | 4 | 0 | 1 | 2 | 0 | 1 | 1 | 8 | 110 | 1 | 27 |
| mem0_addonly | 4 | 0 | 2 | 2 | 0 | 1 | 1 | 4 | 107 | 1 | 33 |
| kyc | 11 | 3 | 3 | 0 | 1 | 0 | 3 | 10 | 12 | 16 | 75 |
| oracle | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 10 | 11 | 0 | 22 |

### Pre-registered hypotheses

**H1** (Raw-RAG at least as accurate as agentic memory)
- raw_rag − mem0: +37.2 pts (95% CI +27.7 to +46.7, n=218) → higher
- raw_rag − mem0_addonly: +37.2 pts (95% CI +28.1 to +46.3, n=218) → higher

**H2** (on knowledge-update questions, Raw-RAG gives a superseded value more often than Mem0)
- stale-answer rate, raw_rag − mem0: +13.4 pts (95% CI +0.0 to +27.3, n=67) → no detectable difference

**H3** (Mem0 add-only at least as accurate as default Mem0)
- mem0_addonly − mem0: +0.0 pts (95% CI -3.2 to +3.2, n=218) → no detectable difference

**H4** (Raw-RAG misses are mostly use, not coverage): the gold evidence was in the context for 55 of 66 wrong answers (83%). (Aggregation questions count as covered only when every contribution turn was shown.)

### Knowledge-update questions in full (exploratory, not pre-registered)

| System | n | correct | stale value | other wrong value | abstained | stale share of answers given |
|---|---|---|---|---|---|---|
| raw_rag | 67 | 45 | 13 | 6 | 3 | 20% |
| raw_rag_norerank | 67 | 35 | 22 | 6 | 4 | 35% |
| recent | 67 | 13 | 0 | 9 | 45 | 0% |
| mem0 | 67 | 5 | 4 | 9 | 49 | 22% |
| mem0_addonly | 67 | 6 | 4 | 13 | 44 | 17% |
| kyc | 67 | 26 | 11 | 29 | 1 | 17% |
| oracle | 67 | 54 | 2 | 11 | 0 | 3% |

### Empty answer fields (exploratory, not pre-registered)

| System | empty answers | of which dollar amounts | accuracy on the rest % |
|---|---|---|---|
| raw_rag | 1 | 1 | 66.4 |
| raw_rag_norerank | 2 | 2 | 58.3 |
| recent | 0 | 0 | 27.1 |
| mem0 | 0 | 0 | 28.9 |
| mem0_addonly | 2 | 2 | 29.2 |
| kyc | 0 | 0 | 38.5 |
| oracle | 0 | 0 | 79.4 |

Paired comparisons without those questions:

- raw_rag − mem0: +37.3 pts (95% CI +28.0 to +46.9, n=217) → higher
- raw_rag − mem0_addonly: +37.2 pts (95% CI +28.1 to +46.9, n=215) → higher
- mem0_addonly − mem0: +0.0 pts (95% CI -3.2 to +3.2, n=216) → no detectable difference
- oracle − raw_rag: +12.9 pts (95% CI +5.4 to +20.6, n=217) → higher

## Reader: qwen3:4b

| System | n | Accuracy % (95% CI) | single_session | knowledge_update | temporal | aggregation | decision | abstention | Context tokens |
|---|---|---|---|---|---|---|---|---|---|
| raw_rag | 218 | 82.1 (76.2–87.6) | 100 | 84 | 65 | 44 | 89 | 98 | 4,694 |
| raw_rag_norerank | 218 | 69.7 (62.2–76.8) | 85 | 61 | 47 | 44 | 83 | 100 | 4,808 |
| recent | 218 | 29.4 (23.1–36.2) | 7 | 24 | 0 | 0 | 6 | 100 | 3,710 |
| mem0 | 218 | 30.7 (24.3–37.5) | 19 | 6 | 14 | 17 | 22 | 100 | 349 |
| mem0_addonly | 218 | 27.5 (21.3–34.2) | 22 | 6 | 7 | 0 | 11 | 100 | 355 |
| kyc | 218 | 50.5 (41.7–58.9) | 70 | 37 | 42 | 0 | 39 | 91 | 893 |
| oracle | 218 | 83.5 (77.3–89.1) | 100 | 78 | 81 | 56 | 72 | 100 | 96 |

**Wrong answers by trap** (count of wrong answers that equal a planted wrong value)

| System | stale | initial | hypothetical | advisor_proposal | spouse | misspoke | by_session_year | other trap | abstained | answered_unanswerable | no trap |
|---|---|---|---|---|---|---|---|---|---|---|---|
| raw_rag | 8 | 2 | 0 | 0 | 0 | 0 | 5 | 3 | 5 | 1 | 15 |
| raw_rag_norerank | 15 | 4 | 0 | 0 | 0 | 0 | 1 | 2 | 18 | 0 | 26 |
| recent | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 147 | 0 | 7 |
| mem0 | 1 | 0 | 1 | 3 | 0 | 0 | 1 | 4 | 125 | 0 | 16 |
| mem0_addonly | 3 | 0 | 1 | 3 | 0 | 0 | 1 | 4 | 129 | 0 | 17 |
| kyc | 11 | 4 | 2 | 1 | 0 | 0 | 1 | 7 | 11 | 4 | 67 |
| oracle | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 4 | 0 | 31 |

### Pre-registered hypotheses

**H1** (Raw-RAG at least as accurate as agentic memory)
- raw_rag − mem0: +51.4 pts (95% CI +43.2 to +59.2, n=218) → higher
- raw_rag − mem0_addonly: +54.6 pts (95% CI +46.8 to +62.2, n=218) → higher

**H2** (on knowledge-update questions, Raw-RAG gives a superseded value more often than Mem0)
- stale-answer rate, raw_rag − mem0: +10.4 pts (95% CI +3.0 to +19.1, n=67) → higher

**H3** (Mem0 add-only at least as accurate as default Mem0)
- mem0_addonly − mem0: -3.2 pts (95% CI -7.1 to +0.0, n=218) → no detectable difference

**H4** (Raw-RAG misses are mostly use, not coverage): the gold evidence was in the context for 26 of 38 wrong answers (68%). (Aggregation questions count as covered only when every contribution turn was shown.)

### Knowledge-update questions in full (exploratory, not pre-registered)

| System | n | correct | stale value | other wrong value | abstained | stale share of answers given |
|---|---|---|---|---|---|---|
| raw_rag | 67 | 56 | 8 | 3 | 0 | 12% |
| raw_rag_norerank | 67 | 41 | 15 | 7 | 4 | 24% |
| recent | 67 | 16 | 0 | 0 | 51 | 0% |
| mem0 | 67 | 4 | 1 | 6 | 56 | 9% |
| mem0_addonly | 67 | 4 | 3 | 5 | 55 | 25% |
| kyc | 67 | 25 | 11 | 31 | 0 | 16% |
| oracle | 67 | 52 | 0 | 14 | 1 | 0% |

### Empty answer fields (exploratory, not pre-registered)

| System | empty answers | of which dollar amounts | accuracy on the rest % |
|---|---|---|---|
| raw_rag | 12 | 12 | 86.9 |
| raw_rag_norerank | 19 | 18 | 76.4 |
| recent | 2 | 2 | 29.6 |
| mem0 | 9 | 9 | 32.1 |
| mem0_addonly | 10 | 10 | 28.8 |
| kyc | 22 | 21 | 56.1 |
| oracle | 27 | 27 | 95.3 |

Paired comparisons without those questions:

- raw_rag − mem0: +54.0 pts (95% CI +45.6 to +62.1, n=202) → higher
- raw_rag − mem0_addonly: +57.7 pts (95% CI +49.8 to +65.5, n=201) → higher
- mem0_addonly − mem0: -2.4 pts (95% CI -6.5 to +1.0, n=205) → no detectable difference
- oracle − raw_rag: +9.0 pts (95% CI +2.7 to +15.3, n=189) → higher

## Mem0 write decisions

| Store | clients | ADD | UPDATE | DELETE | messages skipped |
|---|---|---|---|---|---|
| mem0 | 9 | 3468 | 354 | 438 | 0 |
| mem0_addonly | 9 | 4926 | 0 | 0 | 0 |
