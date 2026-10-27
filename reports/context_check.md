# Does the gold value reach the reader? (exploratory)

Test split, 112 questions whose answer is a value stated in a meeting (single_session, knowledge_update, decision). "In context" means the gold value appears in what the reader was shown; for Mem0, "in store" means it appears in any stored memory. A value can appear for an unrelated reason, so these are upper bounds.

| System | n | gold value in context | in Mem0 store | llama3.2:3b correct when in context | qwen3:4b correct when in context |
|---|---|---|---|---|---|
| raw_rag | 112 | 110 (98%) | - | 86/110 | 99/110 |
| mem0 | 112 | 23 (21%) | 104 (93%) | 16/23 | 13/23 |
| mem0_addonly | 112 | 21 (19%) | 108 (96%) | 16/21 | 12/21 |
| kyc | 112 | 106 (95%) | - | 41/106 | 51/106 |

By category (gold value in context):

| System | single_session | knowledge_update | decision |
|---|---|---|---|
| raw_rag | 27/27 | 65/67 | 18/18 |
| mem0 | 6/27 | 8/67 | 9/18 |
| mem0_addonly | 6/27 | 7/67 | 8/18 |
| kyc | 27/27 | 63/67 | 16/18 |
