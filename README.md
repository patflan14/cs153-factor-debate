# Multi-Agent Debate for Discovering Quantitative Trading Signals

**Patrick Flanagan · Stanford CS153 Final Project · June 2026**

A four-agent LLM debate system that proposes, critiques, revises, and ranks
new candidate factors for a live quantitative-equity trading strategy.
Running in production for ~2 months. **1,214** LLM proposals → **448
(37%)** survived the Critic + Revisor + validation. The survivors feed the
live trading strategy, which has posted **Sharpe 3.33 / CAGR 134% / 58×
total return** over 4.8 years out-of-sample at the deployed K=25, 1.5×
leverage configuration.

---

## For peer reviewers — start here

Three ways to engage with the project, in increasing order of effort.

### Option 1 · Read the notebook (zero install, ~5 minutes)

The walkthrough notebook **[`demo.ipynb`](demo.ipynb)** renders directly in
GitHub with every chart and printed output pre-computed. **Open it,
scroll top to bottom**, and you'll see the project goal, real production
prompts, three real (redacted) debate transcripts including the Critic
catching a real lookahead bug, the proposal funnel chart, the strategy's
4.8-year equity curve, and the monthly returns breakdown. No clone, no
install, nothing to run.

### Option 2 · Run it locally to verify it works (5 minutes)

```bash
git clone https://github.com/patflan14/cs153-factor-debate.git
cd cs153-factor-debate
python3 -m venv .venv
source .venv/bin/activate          # on Windows: .venv\Scripts\activate
pip install -r requirements.txt

# This should print a recorded production debate end-to-end:
python run_demo.py --dry-run
```

What you should see: a printed transcript of one real production debate,
with the five Proposer slots' raw counts, the Critic's verdicts (10
approve / 10 revise / 0 reject), the final slate of 7 proposals, and a
sample of the Critic's actual reasoning catching a 60-day-rolling-alpha
lookahead bug. No API key needed — the dry-run replays a recorded
transcript.

### Option 3 · Open the notebook interactively (5 more minutes)

After running Option 2 once:

```bash
jupyter notebook demo.ipynb
```

Then **Cell → Run All**. Every cell should execute cleanly and produce
the same outputs you saw in the GitHub-rendered version. The final cell
prints `Sharpe ratio: 3.33` — that's the headline number computed from
the actual daily returns of the deployed strategy.

### What "it works" means for the peer review

- ✅ `python run_demo.py --dry-run` finishes with exit code 0 and prints
  a coherent debate transcript.
- ✅ `jupyter nbconvert --execute --inplace demo.ipynb` finishes with
  exit code 0 (every cell runs without raising).
- ✅ The notebook contains three rendered matplotlib charts (funnel,
  equity curve, monthly returns) and the printed equity-curve cell ends
  with `Sharpe ratio: 3.33`.

Verified end-to-end on Python 3.13 in a fresh venv before this README
was committed. If anything breaks for you, please [file an
issue](https://github.com/patflan14/cs153-factor-debate/issues) — that's
genuinely useful feedback.

---

## What this project actually is

I run a **systematic quantitative-equity strategy**: every trading day, a
computer system I built ranks ~3,000 US large-cap stocks, buys the top
(long), shorts the bottom, holds for a few days, and re-ranks. The
ranking comes from combining **factors** — formulas that produce a
number per stock per day. My strategy uses several thousand of them. The
strategy's quality is bottlenecked on the quality of the factor library.

This project is an **LLM agent system that proposes new candidate
factors** to add to that library, with the failure modes of single-LLM
ideation (everything sounds like momentum, the LLM approves its own
proposals at near 100%) explicitly designed out.

### The architecture in one diagram

```
   research challenge
        │
        ▼
   Proposer × 5 parallel slots ───> ~25 raw candidates
   (textbook / cross_domain / contrarian / interaction / orthogonal)
        │
        ▼
   batched Critic → {approve, revise, reject}
        │
        ▼
   Revisor (one batched call across every "revise")
        │
        ▼
   Backfill slots if approved < target
        │
        ▼
   Ranker picks top-k if approved > max
        │
        ▼
   final slate → candidate factor library
                → downstream stages of the live trading pipeline
                → portfolio
```

Four agent roles, each with its own system prompt in [`prompts/`](prompts/):

- **Proposer** — five copies run in parallel, each with a distinct
  one-sentence mandate. The *"you are one of 5 — do not hedge"*
  instruction is what stops everyone collapsing onto momentum.
- **Critic** — sees only the proposal's name, rationale, and computation
  description. *Never* the Proposer's chain-of-thought. That structural
  separation is what drops the approval rate from ~100% (single-agent
  self-review) to ~37%.
- **Revisor** — one batched LLM call that addresses every "revise"
  verdict at once. Recovers ~20% of the slate per debate.
- **Ranker** — fires only when the Critic approved more than the slate
  cap. Diversity first, relevance second, derivation simplicity as
  tie-breaker.

---

## What's in this repo

| Where | What |
| --- | --- |
| [`demo.ipynb`](demo.ipynb) | **The main walkthrough.** Background primer, real agent code, three real debate transcripts, funnel chart, K × leverage table, equity curve, monthly returns chart. |
| [`results/REPORT.pdf`](results/REPORT.pdf) | The typeset writeup. Problem statement, architecture, transcript walkthrough, operational funnel, portfolio results, lessons, future work. |
| [`results/REPORT.md`](results/REPORT.md) | Same writeup as Markdown. |
| [`agents.py`](agents.py) | The four agent classes with their system prompts (~600 lines). |
| [`debate.py`](debate.py) | The orchestrator. Parallel proposing, batched critique, single-call revision, optional backfill and rank. |
| [`run_demo.py`](run_demo.py) | Entry point. `--dry-run` replays a recorded production debate without an API key. |
| [`prompts/`](prompts/) | Every system + user prompt as a standalone text file. Edit any of them without touching Python. |
| [`transcripts/`](transcripts/) | Three real production debates (redacted): the common case, a Critic-as-filter heavy-reject case, and a Ranker-in-action clean-pass case. |
| [`results/operational_metrics.json`](results/operational_metrics.json) | Machine-readable production metrics behind the funnel. |
| [`results/strategy_returns.json`](results/strategy_returns.json) | The deployed strategy's daily returns over the 4.8-year OOS window — what the equity-curve and monthly-returns charts plot. |

---

## Running one fresh debate end-to-end (optional, requires API key)

If you want to see the system actually call a live LLM (not the
replayed dry-run):

```bash
export OPENROUTER_API_KEY=...    # OpenRouter free key works but rate-limits hard
python run_demo.py               # one fresh debate, ~5 minutes
```

The default model is OpenRouter's free-tier
`meta-llama/llama-3.3-70b-instruct:free`. The free tier rate-limits
aggressively, so expect 429 retries — a paid model (set
`LLM_MODEL=<paid-model>`) is the reliable path.

---

## What's *not* in this repo (IP boundary)

The downstream stages of the production trading pipeline that consume
the LLM-proposed factors, the backtest engine, and the actual factor
formulas the live strategy trades — those stay in the production
codebase. The portfolio numbers above are evidence that the LLM debate
system feeds something that works in the real world; they are not
reproducible from this scaffold alone. The agent architecture, system
prompts, transcripts, and operational evidence (proposal funnel,
materialisation rate) are what's public here and what the report's claims
rest on.

---

## Quick links

- **Notebook walkthrough:** [demo.ipynb](demo.ipynb)
- **Full writeup:** [results/REPORT.pdf](results/REPORT.pdf)
- **Architecture in code:** [agents.py](agents.py) · [debate.py](debate.py)
- **Real debate (lookahead catch):** [transcripts/debate_001_revision_heavy.json](transcripts/debate_001_revision_heavy.json)
- **The five Proposer mandates verbatim:** [prompts/proposer_slot_instructions.json](prompts/proposer_slot_instructions.json)

---

## AI tool usage disclosure

(Per the CS153 AI policy.)

Claude (Anthropic) was used as a coding and writing assistant in preparing this submission — helping draft and edit prose, generate visualisation code, and refactor the agent classes from my production codebase into a standalone scaffold suitable for the class.

The underlying work — the agent architecture, the system prompts, the recorded debate transcripts, the strategy returns, and the portfolio metrics — comes from a system I designed, built, and have operated in production.
