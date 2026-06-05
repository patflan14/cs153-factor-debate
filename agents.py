"""Multi-agent debate roles for LLM-driven research-signal discovery.

This module mirrors the four-stage architecture that runs in production:

    Stage 1 — Proposer   : 5 parallel slots × 5 proposals each = 25 raw
                           candidates. Each slot has a distinct *mandate*
                           (textbook / cross_domain / contrarian / interaction
                           / orthogonal) so the candidate set explores the
                           signal space instead of collapsing onto one
                           dominant idea.
    Stage 2 — Critic     : Batches of ~8 candidates, evaluated in parallel.
                           Each candidate gets a verdict: approve / revise /
                           reject, with concrete reasoning.
    Stage 3 — Revisor    : Single LLM call that revises every "revise"
                           verdict in one batch using the critic's
                           per-candidate guidance.
    Stage 4 — Ranker     : When more candidates pass than the target slate
                           size, the Ranker selects a diverse, relevant,
                           parsimonious top-k.

Two optional Backfill slots (diverse, defensive) run when the Critic culls
the slate below the target count.

LLM access goes through OpenRouter (OpenAI-compatible API). The model is
configurable via the LLM_MODEL environment variable. With no API key, all
classes still import cleanly; only their `.run(...)` methods will fail.
"""

from __future__ import annotations

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from typing import Any, List, Optional

from openai import OpenAI


logger = logging.getLogger(__name__)

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    # Allow imports to succeed without a key so `--dry-run` works on a
    # fresh checkout. Any actual LLM call will surface the 401 from
    # OpenRouter, which is the right place for that error to fire.
    api_key=os.environ.get("OPENROUTER_API_KEY") or "not-set",
)
MODEL = os.environ.get("LLM_MODEL", "meta-llama/llama-3.3-70b-instruct:free")

# Soft rate limit so demos against free-tier endpoints don't burn through
# the per-minute quota.
_MIN_CALL_GAP_S = 0.3
_last_call_time = 0.0


# ── Shared dataclasses ──────────────────────────────────────────────────────


@dataclass
class SignalProposal:
    """One candidate research signal proposed by the Proposer.

    Fields are intentionally generic so the scaffold can illustrate the role
    of the dataclass without exposing the production schema.
    """

    input_name: str
    data_type: str            # one of: price, fundamental, technical, macro
    rationale: str
    derivation: str           # plain-English description of the computation
    suggested_abbrev: str = ""
    slot: str = "unknown"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SignalProposal":
        return cls(
            input_name=str(d.get("input_name", "unnamed")),
            data_type=str(d.get("data_type", "price")),
            rationale=str(d.get("rationale", "")),
            derivation=str(d.get("derivation", "")),
            suggested_abbrev=str(d.get("suggested_abbrev", ""))[:7],
            slot=str(d.get("slot", "unknown")),
        )


@dataclass
class CriticVerdict:
    """The Critic's assessment of one proposal."""

    input_name: str
    verdict: str              # approve | revise | reject
    reasoning: str
    revision_guidance: str = ""


@dataclass
class DebateResult:
    """End-to-end output of one debate cycle."""

    challenge: str
    raw_proposals: List[SignalProposal] = field(default_factory=list)
    approved: List[SignalProposal] = field(default_factory=list)
    revised: List[SignalProposal] = field(default_factory=list)
    rejected_reasons: List[str] = field(default_factory=list)
    final_slate: List[SignalProposal] = field(default_factory=list)
    llm_calls: int = 0
    elapsed_s: float = 0.0


# ── Rate-limited LLM call ───────────────────────────────────────────────────


def _rate_limit() -> None:
    global _last_call_time
    now = time.monotonic()
    gap = now - _last_call_time
    if gap < _MIN_CALL_GAP_S:
        time.sleep(_MIN_CALL_GAP_S - gap)
    _last_call_time = time.monotonic()


def _call_llm(
    system_prompt: str,
    user_prompt: str,
    model: str = MODEL,
    max_tokens: int = 1500,
    temperature: float = 0.8,
    timeout: int = 90,
) -> str:
    _rate_limit()
    resp = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=timeout,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return resp.choices[0].message.content or ""


# ── JSON parsing helpers (LLMs frequently wrap output in code fences) ───────


def _extract_json_array(raw: str) -> list[dict]:
    text = raw.strip()
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0]
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0]
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1:
        return []
    try:
        data = json.loads(text[start:end + 1])
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        return []


def _extract_json_object(raw: str) -> dict:
    text = raw.strip()
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0]
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0]
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        return {}
    try:
        data = json.loads(text[start:end + 1])
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


# ── Stage 1: Proposer slots ─────────────────────────────────────────────────

# These five mandates are lifted verbatim from the production codebase. Each
# slot is a different lens on the candidate space; running them in parallel
# forces diversity that a single Proposer with a single prompt does not
# produce.
PROPOSER_SLOTS: list[dict[str, Any]] = [
    {
        "slot": "textbook",
        "n": 5,
        "instruction": (
            "Generate 5 input proposals using well-established quantitative "
            "signals. Academic factor literature — Fama-French extensions, "
            "quality metrics, momentum variants, volatility risk premia. "
            "Adapt known signals to this specific failure."
        ),
    },
    {
        "slot": "cross_domain",
        "n": 5,
        "instruction": (
            "Generate 5 input proposals that combine data from at least 2 "
            "DIFFERENT domains. Examples: fundamental × technical (earnings "
            "yield relative to momentum), macro × price (vol regime adjusted "
            "returns), volume × fundamental (turnover-scaled quality). "
            "The alpha must come from the COMBINATION, not either input alone."
        ),
    },
    {
        "slot": "contrarian",
        "n": 5,
        "instruction": (
            "Generate 5 input proposals based on contrarian or mean-reversion "
            "logic. Things most quants would NOT build. Signals that go "
            "AGAINST the current factor drivers. If the existing signal "
            "over-indexes on earnings growth, propose inputs that capture "
            "when earnings growth is misleading."
        ),
    },
    {
        "slot": "interaction",
        "n": 5,
        "instruction": (
            "Generate 5 input proposals that capture INTERACTIONS between "
            "variables — ratios of two inputs, conditional signals, "
            "regime-dependent transformations. The signal should emerge from "
            "the relationship between inputs, not from either alone."
        ),
    },
    {
        "slot": "orthogonal",
        "n": 5,
        "instruction": (
            "Generate 5 input proposals designed to be UNCORRELATED with "
            "the dominant drivers (momentum, earnings growth). Think: "
            "balance-sheet stability, liquidity dynamics, supply/demand "
            "microstructure, defensive quality. Inputs that would rank "
            "stocks DIFFERENTLY than the current signal."
        ),
    },
]


BACKFILL_SLOTS: list[dict[str, Any]] = [
    {
        "slot": "backfill_diverse",
        "n": 5,
        "instruction": (
            "Generate 5 input proposals that are DIFFERENT from common "
            "approaches. Avoid momentum, earnings growth, and standard "
            "value metrics. Think: liquidity dynamics, balance-sheet "
            "stability, supply-chain signals, institutional ownership "
            "changes, or cross-asset correlations."
        ),
    },
    {
        "slot": "backfill_defensive",
        "n": 5,
        "instruction": (
            "Generate 5 input proposals focused on defensive and risk "
            "characteristics. Inputs that identify stocks that HOLD UP "
            "during drawdowns — low leverage, stable cash flows, low "
            "beta residuals, quality-at-reasonable-price signals."
        ),
    },
]


PROPOSER_SYSTEM = (
    "You are a quantitative researcher proposing new data series for a "
    "systematic trading system. Return ONLY valid JSON arrays. Be creative "
    "but precise — every proposed input must be computable from observable "
    "market or fundamental data, with no lookahead."
)


def _build_proposer_prompt(challenge: str, slot: dict[str, Any]) -> str:
    return f"""## Research Challenge

{challenge}

## YOUR SPECIFIC MANDATE: {slot["slot"].upper()}

{slot["instruction"]}

You are one of 5 parallel researchers. The other 4 are covering different
approaches. Do NOT hedge by mixing approaches — commit fully to your mandate.

## Output

Return a JSON array of {slot["n"]} proposals. Each item must have:
- "input_name": short snake_case identifier (≤ 30 chars)
- "data_type": one of "price", "fundamental", "technical", "macro"
- "rationale": 1–2 sentences explaining the economic intuition
- "derivation": plain-English description of how to compute the series
- "suggested_abbrev": ≤ 7 lowercase alphanumeric chars

Return only the JSON array, no surrounding prose.
"""


class Proposer:
    """Stage 1: parallel multi-slot proposal generation."""

    def propose_slot(self, challenge: str, slot: dict[str, Any]) -> list[SignalProposal]:
        prompt = _build_proposer_prompt(challenge, slot)
        raw = _call_llm(PROPOSER_SYSTEM, prompt, max_tokens=2000)
        items = _extract_json_array(raw)
        proposals: list[SignalProposal] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            item["slot"] = slot["slot"]
            try:
                proposals.append(SignalProposal.from_dict(item))
            except (KeyError, TypeError) as e:
                logger.debug("Skipping malformed proposal: %s", e)
        logger.info("Slot '%s': %d proposals parsed", slot["slot"], len(proposals))
        return proposals

    def propose_all(
        self,
        challenge: str,
        slots: list[dict[str, Any]] = PROPOSER_SLOTS,
        parallel: bool = True,
    ) -> list[SignalProposal]:
        if not parallel:
            out: list[SignalProposal] = []
            for slot in slots:
                out.extend(self.propose_slot(challenge, slot))
            return out

        results: list[SignalProposal] = []
        with ThreadPoolExecutor(max_workers=len(slots)) as ex:
            futures = {ex.submit(self.propose_slot, challenge, s): s for s in slots}
            for fut in as_completed(futures):
                try:
                    results.extend(fut.result())
                except Exception as e:
                    logger.error("Slot %s failed: %s", futures[fut]["slot"], e)
        return _dedup_by_name(results)


def _dedup_by_name(proposals: list[SignalProposal]) -> list[SignalProposal]:
    seen: set[str] = set()
    unique: list[SignalProposal] = []
    for p in proposals:
        if p.input_name not in seen:
            seen.add(p.input_name)
            unique.append(p)
    return unique


# ── Stage 2: Critic ────────────────────────────────────────────────────────


CRITIC_SYSTEM = (
    "You are a senior quantitative researcher reviewing proposed data series "
    "for a systematic trading system. You evaluate proposals for feasibility, "
    "novelty, and relevance. Return ONLY valid JSON arrays."
)


def _build_critic_prompt(proposals: list[SignalProposal], challenge: str) -> str:
    specs = [
        {
            "index": i,
            "input_name": p.input_name,
            "data_type": p.data_type,
            "rationale": p.rationale,
            "derivation": p.derivation,
        }
        for i, p in enumerate(proposals)
    ]
    return f"""## Context

{challenge}

## Proposals to Evaluate

{json.dumps(specs, indent=2)}

## Your Task

For each proposal, assess:
1. **Feasibility**: Can this actually be computed from observable data with
   no lookahead?
2. **Novelty**: Is this genuinely different from existing inputs, or just a
   rename / minor variation?
3. **Relevance**: Would this input help address the specific challenge in
   the context above?

For each proposal return:
- "index": the proposal index
- "verdict": "approve" | "revise" | "reject"
- "reasoning": 1–2 sentences explaining your verdict
- "revision_guidance": (only if verdict="revise") the specific fix needed

Return a JSON array of verdicts, one per proposal.
"""


class Critic:
    """Stage 2: batched adversarial review."""

    def critique_batch(
        self,
        proposals: list[SignalProposal],
        challenge: str,
    ) -> list[CriticVerdict]:
        if not proposals:
            return []
        prompt = _build_critic_prompt(proposals, challenge)
        raw = _call_llm(CRITIC_SYSTEM, prompt, max_tokens=2000)
        items = _extract_json_array(raw)

        verdicts: list[CriticVerdict] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            idx = item.get("index", -1)
            if 0 <= idx < len(proposals):
                verdicts.append(CriticVerdict(
                    input_name=proposals[idx].input_name,
                    verdict=str(item.get("verdict", "reject")).lower(),
                    reasoning=str(item.get("reasoning", "")),
                    revision_guidance=str(item.get("revision_guidance", "")),
                ))

        # Anything the critic skipped defaults to approve (production behaviour:
        # don't punish a parse error by rejecting good proposals).
        seen = {v.input_name for v in verdicts}
        for p in proposals:
            if p.input_name not in seen:
                verdicts.append(CriticVerdict(
                    input_name=p.input_name,
                    verdict="approve",
                    reasoning="Default approve — not in critic output",
                ))
        return verdicts

    def critique_all(
        self,
        proposals: list[SignalProposal],
        challenge: str,
        batch_size: int = 8,
        parallel: bool = True,
    ) -> list[CriticVerdict]:
        batches = [
            proposals[i:i + batch_size]
            for i in range(0, len(proposals), batch_size)
        ]
        verdicts: list[CriticVerdict] = []
        if not parallel or len(batches) <= 1:
            for b in batches:
                verdicts.extend(self.critique_batch(b, challenge))
            return verdicts

        with ThreadPoolExecutor(max_workers=len(batches)) as ex:
            futures = [ex.submit(self.critique_batch, b, challenge) for b in batches]
            for fut in as_completed(futures):
                try:
                    verdicts.extend(fut.result())
                except Exception as e:
                    logger.error("Critic batch failed: %s", e)
        return verdicts


# ── Stage 3: Revisor ────────────────────────────────────────────────────────


REVISOR_SYSTEM = (
    "You are revising proposed data series based on reviewer feedback. "
    "Each item below has the original proposal and specific guidance from "
    "the critic. Revise each one to address the feedback. "
    "Return ONLY a JSON array, one revised proposal per input item, in "
    "the same order."
)


def _build_revisor_prompt(
    items: list[tuple[SignalProposal, CriticVerdict]],
    challenge: str,
) -> str:
    payload = [
        {
            "input_name": p.input_name,
            "original": p.to_dict(),
            "feedback": v.revision_guidance or v.reasoning,
        }
        for p, v in items
    ]
    return f"""## Research Challenge (abbreviated)

{challenge[:1500]}

## Proposals Needing Revision

{json.dumps(payload, indent=2)}

## Your Task

For each item, revise the proposal to address the critic's feedback. Keep
what works, fix what was flagged. Return a JSON array with one revised
proposal per input item, in the same schema as the original. Same order.
"""


class Revisor:
    """Stage 3: single batched revision call."""

    def revise(
        self,
        to_revise: list[tuple[SignalProposal, CriticVerdict]],
        challenge: str,
    ) -> list[SignalProposal]:
        if not to_revise:
            return []
        prompt = _build_revisor_prompt(to_revise, challenge)
        raw = _call_llm(REVISOR_SYSTEM, prompt, max_tokens=2500)
        items = _extract_json_array(raw)

        revised: list[SignalProposal] = []
        for orig_pair, item in zip(to_revise, items):
            if not isinstance(item, dict):
                continue
            try:
                item["slot"] = orig_pair[0].slot + ":revised"
                revised.append(SignalProposal.from_dict(item))
            except (KeyError, TypeError):
                continue
        return revised


# ── Stage 4: Ranker ────────────────────────────────────────────────────────


RANKER_SYSTEM = (
    "You select the most diverse and promising data series proposals from a "
    "pre-approved set. Return ONLY valid JSON."
)


def _build_ranker_prompt(
    approved: list[SignalProposal],
    challenge: str,
    top_k: int,
) -> str:
    specs = [
        {"index": i, "input_name": p.input_name, "rationale": p.rationale,
         "derivation": p.derivation}
        for i, p in enumerate(approved)
    ]
    return f"""## Context (abbreviated)

{challenge[:1500]}

## Approved Candidates ({len(approved)} total)

{json.dumps(specs, indent=2)}

## Your Task

Select the top {top_k} proposals. Criteria:
1. **Mutual diversity**: cover different signal types — avoid selecting
   3 variants of the same idea.
2. **Relevance**: prioritise proposals that most directly address the
   specific challenge.
3. **Simplicity**: when two proposals are equally good, prefer the simpler
   derivation (fewer inputs, standard operations).

Return JSON: {{"selected_indices": [list of {top_k} indices]}}
"""


class Ranker:
    """Stage 4: pick a diverse, relevant, parsimonious top-k."""

    def select(
        self,
        approved: list[SignalProposal],
        challenge: str,
        top_k: int = 12,
    ) -> list[SignalProposal]:
        if len(approved) <= top_k:
            return approved
        prompt = _build_ranker_prompt(approved, challenge, top_k)
        raw = _call_llm(RANKER_SYSTEM, prompt, max_tokens=800, temperature=0.4)
        data = _extract_json_object(raw)
        indices = data.get("selected_indices", [])
        selected = [approved[i] for i in indices
                    if isinstance(i, int) and 0 <= i < len(approved)]
        return selected[:top_k] if selected else approved[:top_k]
