# Multi-Agent Deal Discovery & Pricing — Complete Documentation

Everything about this project: all seven
agents, the orchestration, the memory and deduplication, the concepts, the
reasoning behind each choice, and the interview questions it invites.

## Read in this order

| Document | What it covers |
|---|---|
| [01-architecture.md](01-architecture.md) | The seven agents, how a cycle runs, and the full lifecycle of one deal |
| [02-repository-map.md](02-repository-map.md) | Every folder and file, with its role and dependencies |
| [03-code-walkthrough-agents.md](03-code-walkthrough-agents.md) | `base.py`, `deals.py`, and all seven agents — line by line |
| [04-code-walkthrough-orchestration.md](04-code-walkthrough-orchestration.md) | `config.py`, `logs.py`, `memory.py`, `framework.py` |
| [05-code-walkthrough-interfaces.md](05-code-walkthrough-interfaces.md) | `build_index.py`, `run.py`, `app.py`, the Modal service |
| [06-concepts.md](06-concepts.md) | Multi-agent design, structured outputs, RAG for pricing, ensembling, deduplication, t-SNE, serverless GPU |
| [07-design-decisions.md](07-design-decisions.md) | Every non-obvious choice, rejected alternatives, and the bugs found by running it |
| [08-interview-questions.md](08-interview-questions.md) | 30 questions with full answers |

## The project in one paragraph

Seven agents that watch retail deal feeds, work out what each item is actually
worth, and send a push notification when something is priced well below that. The
Scanner reads RSS feeds and uses structured outputs to shortlist deals with an
unambiguous price. Three independent estimators — an LLM grounded by retrieval
over a priced product catalogue, a fine-tuned model on a Modal GPU, and a local
residual network — each price the item, and an Ensemble combines them into one
number plus a confidence signal. A Planner ranks the candidates, alerts on the
single best one if it clears a discount threshold, and writes everything to a
memory that ensures the same product is never alerted twice.

## The seven agents

```
       ┌── Scanner ──── reads RSS feeds, shortlists deals with a clear price
       │
Planner├── Ensemble ─┬─ Frontier   LLM + RAG over a priced product catalogue
       │             ├─ Specialist fine-tuned model on a Modal GPU
       │             └─ Neural     local residual net over hashed text
       │
       └── Messenger ─ writes and sends the alert

           Memory ──── dedup across runs, so nothing is alerted twice
```

## What was actually verified

The full cycle was run end to end on this machine against live feeds: scraping,
structured-output shortlisting, RAG pricing against a 4,000-product Chroma index,
the local neural estimator, a two-estimator ensemble with a real spread, the
threshold logic, alert composition, memory persistence, and deduplication across
two consecutive cycles.

Two real bugs surfaced during those runs — a device mismatch in the neural
estimator's target scaler, and a failing estimator being retried on every single
deal. Both are documented in
[07-design-decisions.md](07-design-decisions.md) and both are fixed.
