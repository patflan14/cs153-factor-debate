# Recorded production debates

Three recorded debates from the production system, redacted for IP. Each
shows the same architecture (5 Proposer slots → batched Critic → Revisor →
optional Backfill → optional Ranker) exercising a different code path:

| File | What it demonstrates |
| --- | --- |
| `debate_001_revision_heavy.json` | Most-common path. ~10 approve, ~10 revise, ~0 reject. Revisor recovers ~4 more. No Ranker needed. Total: 14 in slate. |
| `debate_002_heavy_reject.json`   | Critic acts as a real filter — 7 rejected outright. Triggers two backfill slots to reach the target. Demonstrates the **Critic isn't a rubber-stamp** — when proposals double-down on the failed direction, they get rejected. |
| `debate_003_clean_pass.json`     | Almost everything passes (15/16 approved). Triggers the **Ranker** because we exceed `max_approved`. Shows the Ranker's diversity-first selection in action. |

What is redacted:
- Real `input_name` strings → `proposal_NNN_<short_description>`
- Real factor formulas in `derivation` → English descriptions of the same shape
- Specific internal terms (the "oracle" comparison engine, internal performance dashboards)

What is preserved:
- The actual numerical counts (raw / approve / revise / reject / final slate / LLM calls / elapsed seconds)
- The Critic's reasoning patterns, verbatim where IP-safe
- The slot mandates and their relative productivity

`run_demo.py --dry-run` replays `debate_001_revision_heavy.json`.
