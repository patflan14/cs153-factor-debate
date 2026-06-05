"""End-to-end debate demo.

Two modes:

  --dry-run  Loads three pre-recorded production debates from `transcripts/`
             and prints a representative one. No API key needed.

  (default)  Runs one live debate against the OpenRouter API on a sample
             research challenge. Requires OPENROUTER_API_KEY in env.

Both modes exercise the same code path that produces the final report's
operational metrics.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from textwrap import indent

from agents import SignalProposal, CriticVerdict
from debate import Debate


SAMPLE_CHALLENGE = """\
The current ridge-regression equity signal underperformed on 2025-10-15
(daily alpha = -2.9%). Inspection shows the signal over-rotated into a
crowded earnings-growth × momentum trade just before a sharp
volatility-regime shift. Propose new candidate research signals that, if
added to the candidate factor library, would have given the model more
diverse ranking information on this kind of day. Signals should be
computable from standard market and fundamentals data with no lookahead.
"""


def _print_proposal(p: SignalProposal, prefix: str = "  ") -> None:
    print(f"{prefix}[{p.slot}] {p.input_name}  ({p.data_type})")
    print(f"{prefix}  rationale:  {p.rationale}")
    print(f"{prefix}  derivation: {p.derivation}")
    print()


def _live_run() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    debate = Debate()
    result = debate.run(SAMPLE_CHALLENGE)

    print("\n" + "=" * 70)
    print(f"DEBATE COMPLETE  ({result.llm_calls} LLM calls, {result.elapsed_s:.1f}s)")
    print("=" * 70)
    print(f"Raw candidates:   {len(result.raw_proposals)}")
    print(f"Approved:         {len(result.approved)}")
    print(f"Revised:          {len(result.revised)}")
    print(f"Rejected:         {len(result.rejected_reasons)}")
    print(f"Final slate:      {len(result.final_slate)}\n")

    print("── Final slate ──")
    for p in result.final_slate:
        _print_proposal(p)

    if result.rejected_reasons:
        print("── Rejection reasons (first 3) ──")
        for r in result.rejected_reasons[:3]:
            print(f"  • {r}")
        print()


def _dry_run() -> None:
    print("=" * 70)
    print("DRY RUN — no API calls. Loading a pre-recorded production debate.")
    print("=" * 70)

    transcripts_dir = Path(__file__).parent / "transcripts"
    files = sorted(transcripts_dir.glob("debate_*.json"))
    if not files:
        print("(no transcripts found — run with API key or check transcripts/)")
        sys.exit(1)

    chosen = files[0]
    data = json.loads(chosen.read_text())

    print(f"\nTranscript: {chosen.name}")
    print(f"Production date: {data.get('production_date', '?')}")
    print(f"Challenge:\n{indent(data['challenge'], '  ')}\n")

    print(f"Raw proposals from 5 slots: {data['n_raw']}")
    print(f"Critic verdicts: approve={data['n_approved']} "
          f"revise={data['n_revise']} reject={data['n_reject']}")
    print(f"Final slate after revision + ranking: {len(data['final_slate'])}")
    print(f"LLM calls: {data['llm_calls']}   elapsed: {data['elapsed_s']:.0f}s\n")

    print("── Final slate (anonymised) ──")
    for p_dict in data["final_slate"][:6]:
        print(f"  [{p_dict['slot']}] {p_dict['input_name']}  ({p_dict['data_type']})")
        print(f"      rationale:  {p_dict['rationale']}")
        print(f"      derivation: {p_dict['derivation']}\n")

    print("── Sample Critic catch ──")
    for v in data.get("sample_critic_catches", [])[:2]:
        print(f"  • verdict={v['verdict']} on '{v['input_name']}':")
        print(f"      \"{v['reasoning']}\"")
        if v.get("revision_guidance"):
            print(f"      guidance: \"{v['revision_guidance']}\"")
    print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Replay a recorded debate instead of calling the LLM.",
    )
    args = parser.parse_args()
    if args.dry_run:
        _dry_run()
    else:
        _live_run()


if __name__ == "__main__":
    main()
