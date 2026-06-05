# Prompt library

System and user prompts for each agent stage. These are lifted (with minor
redactions of proprietary domain language) from the production codebase.

| File | Stage | Purpose |
| --- | --- | --- |
| `proposer_system.txt`         | 1 | Shared system prompt across all five Proposer slots |
| `proposer_slot_instructions.json` | 1 | The five mandate strings (textbook / cross_domain / contrarian / interaction / orthogonal) plus the two backfill slots |
| `critic_system.txt`           | 2 | Adversarial reviewer system prompt |
| `critic_user_template.md`     | 2 | The per-batch user prompt — embeds JSON proposals + research challenge |
| `revisor_system.txt`          | 3 | Revisor system prompt |
| `revisor_user_template.md`    | 3 | Batched revision prompt |
| `ranker_system.txt`           | 4 | Ranker system prompt |
| `ranker_user_template.md`     | 4 | Diversity / relevance / simplicity selection criteria |

The Python `agents.py` reads these prompts at call time. Editing them lets
you tune the agents without touching the Python.
