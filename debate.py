"""End-to-end debate orchestrator.

Wires the four agent classes (Proposer / Critic / Revisor / Ranker) into one
debate cycle:

    challenge ──> 5 Proposer slots in parallel ──> 25 raw proposals
                                               │
                                               ▼
                             Batched Critic ──> {approve, revise, reject}
                                               │
                                               ▼
                                Revisor on every "revise"
                                               │
                                               ▼
                       if approved_count < target: backfill slots
                                               │
                                               ▼
                       if approved_count > max:  Ranker selects top-k
                                               │
                                               ▼
                                       DebateResult
"""

from __future__ import annotations

import logging
import time
from typing import List, Optional

from agents import (
    BACKFILL_SLOTS,
    PROPOSER_SLOTS,
    Critic,
    CriticVerdict,
    DebateResult,
    Proposer,
    Ranker,
    Revisor,
    SignalProposal,
    _dedup_by_name,
)


logger = logging.getLogger(__name__)


class Debate:
    """Production-architecture multi-agent debate.

    Parameters mirror the production defaults so a reproduced run lands in
    the same operating regime: target 12 approved, allow up to 15 before
    ranking, run one revision pass and up to one backfill round.
    """

    def __init__(
        self,
        target_approved: int = 12,
        max_approved: int = 15,
        do_revision: bool = True,
        max_backfill_rounds: int = 1,
        ranker_top_k: int = 12,
        critic_batch_size: int = 8,
        parallel_propose: bool = True,
        parallel_critique: bool = True,
    ) -> None:
        self.target_approved = target_approved
        self.max_approved = max_approved
        self.do_revision = do_revision
        self.max_backfill_rounds = max_backfill_rounds
        self.ranker_top_k = ranker_top_k
        self.critic_batch_size = critic_batch_size
        self.parallel_propose = parallel_propose
        self.parallel_critique = parallel_critique

        self.proposer = Proposer()
        self.critic = Critic()
        self.revisor = Revisor()
        self.ranker = Ranker()

    # ── helpers ────────────────────────────────────────────────────────

    def _classify(
        self,
        verdicts: list[CriticVerdict],
        proposals: list[SignalProposal],
    ) -> tuple[
        list[SignalProposal],
        list[tuple[SignalProposal, CriticVerdict]],
        list[str],
    ]:
        by_name = {p.input_name: p for p in proposals}
        approved: list[SignalProposal] = []
        to_revise: list[tuple[SignalProposal, CriticVerdict]] = []
        rejected: list[str] = []
        for v in verdicts:
            p = by_name.get(v.input_name)
            if p is None:
                continue
            if v.verdict == "approve":
                approved.append(p)
            elif v.verdict == "revise":
                to_revise.append((p, v))
            else:
                rejected.append(v.reasoning)
        return approved, to_revise, rejected

    # ── main entry point ──────────────────────────────────────────────

    def run(self, challenge: str) -> DebateResult:
        t0 = time.perf_counter()
        result = DebateResult(challenge=challenge)

        # Stage 1: Propose (5 parallel slots)
        raw = self.proposer.propose_all(
            challenge,
            slots=PROPOSER_SLOTS,
            parallel=self.parallel_propose,
        )
        result.raw_proposals = raw
        result.llm_calls += len(PROPOSER_SLOTS)
        logger.info("Proposer: %d raw candidates", len(raw))

        # Stage 2: Critique (batched)
        verdicts = self.critic.critique_all(
            raw, challenge,
            batch_size=self.critic_batch_size,
            parallel=self.parallel_critique,
        )
        approved, to_revise, rejected = self._classify(verdicts, raw)
        result.llm_calls += max(1, (len(raw) + self.critic_batch_size - 1)
                                // self.critic_batch_size)
        result.rejected_reasons = rejected
        logger.info(
            "Critic round 1: %d approve, %d revise, %d reject",
            len(approved), len(to_revise), len(rejected),
        )

        # Stage 3: Revise (single batched call)
        if self.do_revision and to_revise:
            revised = self.revisor.revise(to_revise, challenge)
            result.revised = revised
            result.llm_calls += 1
            if revised:
                # Re-critique revised batch.
                revised_verdicts = self.critic.critique_all(
                    revised, challenge,
                    batch_size=self.critic_batch_size,
                    parallel=self.parallel_critique,
                )
                rev_approved, _, rev_rejected = self._classify(
                    revised_verdicts, revised,
                )
                result.llm_calls += max(1, (len(revised) + self.critic_batch_size - 1)
                                        // self.critic_batch_size)
                approved.extend(rev_approved)
                result.rejected_reasons.extend(rev_rejected)
                logger.info("Revision: %d/%d revised proposals approved",
                            len(rev_approved), len(revised))

        approved = _dedup_by_name(approved)

        # Backfill if under target
        backfill_rounds = 0
        while (
            len(approved) < self.target_approved
            and backfill_rounds < self.max_backfill_rounds
        ):
            backfill_rounds += 1
            logger.info(
                "Backfill round %d: have %d/%d approved",
                backfill_rounds, len(approved), self.target_approved,
            )
            new = self.proposer.propose_all(
                challenge,
                slots=BACKFILL_SLOTS,
                parallel=self.parallel_propose,
            )
            result.llm_calls += len(BACKFILL_SLOTS)
            new_verdicts = self.critic.critique_all(
                new, challenge,
                batch_size=self.critic_batch_size,
                parallel=self.parallel_critique,
            )
            new_approved, _, new_rejected = self._classify(new_verdicts, new)
            result.llm_calls += max(1, (len(new) + self.critic_batch_size - 1)
                                    // self.critic_batch_size)
            approved.extend(new_approved)
            approved = _dedup_by_name(approved)
            result.rejected_reasons.extend(new_rejected)

        result.approved = approved

        # Stage 4: Rank
        if len(approved) > self.max_approved:
            final = self.ranker.select(approved, challenge, top_k=self.ranker_top_k)
            result.llm_calls += 1
        else:
            final = approved[:self.max_approved]
        result.final_slate = final

        result.elapsed_s = time.perf_counter() - t0
        logger.info(
            "Debate complete: %d raw → %d approved → %d in final slate "
            "(%d LLM calls, %.1fs)",
            len(raw), len(approved), len(final),
            result.llm_calls, result.elapsed_s,
        )
        return result
