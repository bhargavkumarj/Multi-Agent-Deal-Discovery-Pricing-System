# Repository Map

Every folder and every file in this repository.

```
multi-agent-deal-discovery/
├── README.md                     project-facing readme (setup, usage, design notes)
├── requirements.txt              16 dependencies
├── .env.example                  every environment variable, documented
├── .gitignore                    excludes .env, products_vectorstore/, artifacts/, memory.json
│
├── build_index.py                ENTRY POINT — build the priced product catalogue
├── run.py                        ENTRY POINT — one cycle, a loop, or inspect memory
├── app.py                        ENTRY POINT — Gradio monitoring dashboard
│
├── agents/
│   ├── __init__.py               empty, deliberately (see "Dependency direction")
│   ├── base.py                   the Agent base class: coloured, attributed logging
│   ├── deals.py                  ScrapedDeal, Deal, DealSelection, Estimate, Opportunity
│   ├── scanner.py                AGENT 1 — feeds -> structured shortlist
│   ├── frontier.py               AGENT 2 — LLM + RAG pricing
│   ├── specialist.py             AGENT 3 — fine-tuned model on Modal
│   ├── neural.py                 AGENT 4 — local residual network
│   ├── ensemble.py               AGENT 5 — combines the three estimators
│   ├── messenger.py              AGENT 6 — writes and sends alerts
│   └── planner.py                AGENT 7 — orchestrates a cycle
│
├── core/
│   ├── __init__.py               empty, deliberately
│   ├── config.py                 all settings, feeds, weights, the OpenAI client factory
│   ├── logs.py                   ANSI colours, logging setup, terminal -> HTML conversion
│   ├── memory.py                 persistence and the two-key deduplication
│   └── framework.py              DealDiscovery: owns the store, memory and planner
│
├── modal_app/
│   ├── __init__.py
│   └── pricer_service.py         the fine-tuned pricer, deployed to a Modal GPU
│
└── (generated, git-ignored)
    ├── products_vectorstore/     Chroma index of priced products
    ├── artifacts/residual_net.pt weights from the price prediction project
    └── memory.json               everything seen, and everything alerted
```

## What each file is responsible for

### Entry points

**`build_index.py`** — Builds the Chroma catalogue the Frontier agent retrieves
from. Loads a curated product dataset, embeds each product's summary with
`all-MiniLM-L6-v2`, and stores it with `{category, price}` metadata. Takes
`--limit` and `--reset`. This must run before anything else.

**`run.py`** — The operational CLI. One cycle by default, `--loop --every N` to
run continuously, `--memory` to inspect what has been found. Flags to disable the
Specialist, the Neural estimator, or notification delivery.

**`app.py`** — The Gradio dashboard. Streams the agent log live while a cycle
runs, shows every opportunity found with its price/estimate/discount, and plots
the product catalogue as a 3D t-SNE projection coloured by category.

### The agents

**`agents/base.py`** — Eleven lines. A `name`, a `colour`, and a `log` method
that prefixes every message with the agent's name and wraps it in that agent's
colour. Makes the interleaved transcript readable.

**`agents/deals.py`** — Every data type that moves between agents.
`ScrapedDeal` (raw, from a feed), `Deal` (structured, from the Scanner),
`DealSelection` (the LLM's response schema), `Estimate` (one estimator's
opinion), `Opportunity` (a valued deal). The pydantic `Field` descriptions on
`Deal` are load-bearing — they are sent to the model as the output schema.

**`agents/scanner.py`** — Fetches listings, filters ones already seen, and makes
one structured-output call to shortlist the best.

**`agents/frontier.py`** — The strongest estimator. Embeds the description,
retrieves 5 comparable products with real prices from Chroma, and asks an LLM to
estimate with those in context.

**`agents/specialist.py`** — Twenty-five lines. Connects to a deployed Modal
class and calls it. Deliberately thin: all the heavy machinery lives in the Modal
container.

**`agents/neural.py`** — `ResidualBlock`, `PriceNet` and `NeuralAgent`. Loads the
weights trained in the price prediction project and runs a local forward pass.

**`agents/ensemble.py`** — Constructs whichever estimators are available,
renormalises weights over them, and combines their outputs. Drops an estimator
that throws.

**`agents/messenger.py`** — Composes a two-sentence alert with an LLM and
delivers it via Pushover. Degrades to logging when Pushover is not configured.

**`agents/planner.py`** — The policy. Scan, deduplicate, value, rank, decide,
persist, alert.

### Core

**`core/config.py`** — Feeds, models, weights, thresholds, paths, and
`openai_client()` which honours `OPENAI_BASE_URL` so any OpenAI-compatible
endpoint works.

**`core/logs.py`** — ANSI colour constants, `configure()` for handler setup, and
`to_html()` which converts terminal colour codes into `<span>` tags so the
dashboard can render the same log lines the terminal shows.

**`core/memory.py`** — `url_key`, `content_key`, `keys`, and the `Memory` class.
The deduplication logic lives here and is the most subtle part of the project.

**`core/framework.py`** — `DealDiscovery` owns the Chroma collection, the
`Memory`, and lazily constructs the `PlanningAgent`. Also provides `plot_data`
for the dashboard's t-SNE view.

### Modal

**`modal_app/pricer_service.py`** — Defines the container image, the GPU, the
volume for the model cache, and a `Pricer` class that loads a 4-bit quantised
Llama-3.2-3B with a LoRA adapter and exposes a `price` method. Deployed with
`modal deploy`; called by `agents/specialist.py`.

## Dependency direction

```
        core/config.py  ◄── everything
              ▲
         core/logs.py
              ▲
      agents/base.py ── agents/deals.py
              ▲               ▲
    ┌─────────┼───────────────┼──────────┐
 scanner  frontier  specialist  neural    │
                    │                     │
                 ensemble ◄───────────────┘
                    ▲
                 planner ──► core/memory.py ──► agents/deals.py
                    ▲
            core/framework.py
                    ▲
            ┌───────┴───────┐
         run.py           app.py
```

### Why `agents/__init__.py` and `core/__init__.py` are empty

This was a bug fix, not an oversight.

The first version had `core/__init__.py` re-export `Memory` and
`agents/__init__.py` re-export every agent. That created a cycle:

```
build_index.py
  → core.config
    → core/__init__.py          (runs on first import of the package)
      → core.memory
        → agents.deals
          → agents/__init__.py  (runs on first import of that package)
            → agents.planner
              → core.memory     ← already being imported, not finished
```

```
ImportError: cannot import name 'Memory' from partially initialized module
core.memory (most likely due to a circular import)
```

Emptying both `__init__.py` files removed the cycle entirely. Modules import
each other directly by path, which is explicit and has no ordering hazard.

`core/memory.py` importing from `agents/deals.py` is the one "upward" edge, and
it is fine because `deals.py` has no agent dependencies — it is pure data types.

## External dependencies and why each is there

| Package | Why | Required? |
|---|---|---|
| `openai` | Structured outputs for the Scanner; chat for Frontier and Messenger | Yes |
| `pydantic` | Every data type, and the response schemas | Yes |
| `chromadb` | The priced product catalogue | Yes |
| `sentence-transformers` | Embedding descriptions for retrieval | Yes |
| `feedparser` | Parsing the RSS feeds | Yes |
| `beautifulsoup4` | Extracting text from deal pages | Yes |
| `requests` | Fetching deal pages; Pushover delivery | Yes |
| `datasets` | Loading the product catalogue in `build_index.py` | Build only |
| `torch` | The local neural estimator | No — agent degrades |
| `modal` | The Specialist agent | No — agent degrades |
| `scikit-learn` | `HashingVectorizer`, and t-SNE for the dashboard | Yes |
| `numpy` | Array handling | Yes |
| `plotly` | The dashboard's 3D catalogue plot | Dashboard only |
| `gradio` | The dashboard | Dashboard only |
| `tqdm` | Progress bar in `build_index.py` | Build only |

`torch` and `modal` are imported *inside* `EnsembleAgent.__init__`'s try blocks,
so neither is needed to run the system with just the Frontier estimator.
