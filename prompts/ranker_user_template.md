## Context (abbreviated)

{challenge}

## Approved Candidates ({n_candidates} total)

{candidates_json}

## Your Task

Select the top {top_k} proposals. Criteria:
1. **Mutual diversity**: cover different signal types — avoid selecting 3 variants of the same idea.
2. **Relevance**: prioritise proposals that most directly address the specific challenge.
3. **Simplicity**: when two proposals are equally good, prefer the simpler derivation (fewer inputs, standard operations).

Return JSON: {"selected_indices": [list of {top_k} indices]}
