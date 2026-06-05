## Context

{challenge}

## Proposals to Evaluate

{proposals_json}

## Your Task

For each proposal, assess:
1. **Feasibility**: Can this actually be computed from observable data with no lookahead?
2. **Novelty**: Is this genuinely different from existing inputs, or just a rename / minor variation?
3. **Relevance**: Would this input help address the specific challenge in the context above?

For each proposal return:
- "index": the proposal index
- "verdict": "approve" | "revise" | "reject"
- "reasoning": 1–2 sentences explaining your verdict
- "revision_guidance": (only if verdict="revise") the specific fix needed

Return a JSON array of verdicts, one per proposal.
