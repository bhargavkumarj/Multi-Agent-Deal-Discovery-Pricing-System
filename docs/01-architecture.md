# Architecture

## The problem

Retail deal feeds publish hundreds of listings a day. Most are not bargains —
they are ordinary prices dressed up as discounts, or genuine discounts on things
nobody wants. The interesting question is not "what is discounted?" but "what is
selling for materially less than it is worth?"

Answering it requires three things the feed does not provide:

1. **A clean reading of the listing** — what the product actually is, and what
   the buyer actually pays. Feeds are full of "$200 off" phrasing where the
   saving is not the price.
2. **An independent estimate of value** — what this item is genuinely worth,
   arrived at without reference to the advertised price.
3. **Memory** — because a system that runs continuously will see the same product
   again and again, and alerting twice destroys its usefulness immediately.

Each of those is a different kind of problem, which is why this is a multi-agent
system rather than one function.

## Why seven agents

The decomposition is not decorative. Each agent is a genuinely different job with
a different failure mode, and the boundaries are drawn where the *type of work*
changes.

| Agent | Job | Kind of work |
|---|---|---|
| **Scanner** | Read feeds, scrape pages, shortlist deals with an unambiguous price | Extraction from noisy text |
| **Frontier** | Price an item using retrieved comparable products | Retrieval + reasoning |
| **Specialist** | Price an item with a fine-tuned model | Remote GPU inference |
| **Neural** | Price an item with a local network | Local numerical inference |
| **Ensemble** | Combine three estimates into one, with a confidence signal | Aggregation |
| **Messenger** | Write and deliver the alert | Generation + I/O |
| **Planner** | Sequence everything, rank, decide whether to alert | Orchestration and policy |

Two properties fall out of this split, and they are the practical argument for
it:

- **Independent failure.** The Specialist needs Modal. The Neural agent needs
  trained weights. Neither is required — the Ensemble renormalises its weights
  over whichever estimators actually loaded, and the system degrades rather than
  breaks.
- **Independent substitution.** Swapping the Scanner's model, or adding a fourth
  estimator, touches one file.

## One cycle, step by step

```
DealDiscovery.cycle()
        │
        ▼
PlanningAgent.run()
        │
        ├─► ScannerAgent.scan(seen_urls)
        │       │
        │       ├─ ScrapedDeal.fetch()          3 feeds × 10 entries
        │       │     └─ per entry: GET the deal page, extract details/features
        │       ├─ filter out URLs already in memory
        │       └─ one LLM call with structured output → DealSelection
        │             └─ drop any deal with price <= 0
        │
        ├─► dedup: drop anything in memory, and anything listed twice in this batch
        │
        ├─► for each surviving deal (up to shortlist_size):
        │       EnsembleAgent.price(description)
        │           ├─ FrontierAgent   → embed, query Chroma for 5 comparables,
        │           │                    LLM call with those prices in context
        │           ├─ SpecialistAgent → remote call to the Modal GPU
        │           └─ NeuralAgent     → local forward pass
        │           └─ weighted mean over whichever returned a value > 0
        │
        ├─► sort by discount (estimate − price), take the best
        ├─► write every opportunity to Memory
        │
        └─► if best.discount >= threshold:
                MessagingAgent.alert(best)     LLM writes it, Pushover sends it
                Memory.mark_alerted(best)
```

## Lifecycle of one deal, in full detail

This is a real trace from a measured run.

**1. Scraping.** `ScrapedDeal.fetch()` parses three RSS feeds and takes the first
10 entries each. For each entry it issues a `GET` to the deal page, finds
`div.content-section`, and splits the text on the word "Features" into `details`
and `features`. Fields are truncated — title to 100 characters, details and
features to 500 each — because they go into a prompt with 29 other listings. A
page that fails to load returns `None` and is skipped rather than failing the
scan.

**2. Pre-filtering.** The Scanner drops any listing whose URL is already in
memory. In the measured second cycle this took 9 scraped listings down to 5, so
the LLM call only ever sees new material.

**3. Shortlisting.** The remaining listings are concatenated into one prompt and
sent with `response_format=DealSelection`. The model returns structured JSON that
pydantic validates into typed `Deal` objects — description, price, url. Deals
with `price <= 0` are dropped, because a price the model could not determine is
worse than no deal.

**4. Deduplication.** The Planner drops anything whose URL key or content key is
already in memory, *and* anything whose keys collide with another deal in the
same batch. The batch check exists because the Scanner genuinely does shortlist
the same product twice — in a measured run it listed the same portable monitor
twice from two different feeds.

**5. Valuation.** Each surviving deal goes to the Ensemble:

```
[Frontier] Retrieved 5 similar products from the vector store
[Frontier] Estimates $189.00
[Neural]   Estimates $212.97
[Ensemble] frontier $189, neural $213 -> $193.79
```

The Frontier agent embeds the description with `all-MiniLM-L6-v2`, queries Chroma
for the 5 nearest products, and puts their real prices in the prompt. The Neural
agent hashes the text into 5,000 features and runs a forward pass. The Ensemble
takes a weighted mean over whichever estimators returned a positive value,
renormalising the weights over those present.

**6. Ranking and the decision.**

```
[Planner] Best candidate is $104.76 under estimate (estimator spread $64)
```

Discount is `estimate − price`. The candidates are sorted, and only the single
best is considered for an alert. The spread — the range between the highest and
lowest estimator — is recorded as a confidence signal.

**7. Memory.** *Every* valued opportunity is written to memory, not just the
alerted one. That is what makes the next cycle's deduplication work, and it is
why `run.py --memory` can show negative-discount deals the system correctly
declined to alert on.

**8. Alert.**

```
[Messenger] Composed alert: Get the Amazon Smart Thermostat for $57, saving...
```

An LLM writes a two-sentence notification; Pushover delivers it. Without Pushover
credentials the message is composed and logged, which keeps the whole path
exercisable without an account.

## The three estimators, and why three

Each fails differently, which is the entire justification for ensembling.

**Frontier (LLM + RAG)** — the strongest. It does not ask the model to recall
prices; it shows it five comparable products with their real prices and asks it
to interpolate. That is a question a language model can actually answer. It
drifts on genuinely unusual items where the retrieved comparables are poor
matches.

**Specialist (fine-tuned, on Modal)** — a Llama-3.2-3B fine-tuned on hundreds of
thousands of product/price pairs. Steady on familiar categories, and it encodes
price knowledge in its weights rather than needing retrieval.

**Neural (local residual net)** — the weakest alone. Hashed bag-of-words features
through a 13.5M-parameter residual MLP. But it costs nothing per call, adds no
latency, and — critically — it is wrong in a completely different way to the LLM,
because it has no semantic understanding at all.

The weighted mean (0.6 / 0.25 / 0.15) is more stable than any single one, and the
spread between them is a usable confidence signal: a $500 estimate all three
agree on is a different thing to a $500 estimate they are $400 apart on.

## Where the extension points are

| You want to | Change |
|---|---|
| Watch different feeds | `FEEDS` in `core/config.py` |
| Add a fourth estimator | One class with a `price()` method, plus three lines in `EnsembleAgent.__init__` |
| Change what counts as a bargain | `DISCOUNT_THRESHOLD` |
| Run entirely locally | `OPENAI_BASE_URL=http://localhost:11434/v1` |
| Use a different notification channel | `MessagingAgent.push` |
| Re-weight the ensemble | `WEIGHT_FRONTIER` / `WEIGHT_SPECIALIST` / `WEIGHT_NEURAL` |

## The degradation model

This is the design property worth understanding, because it is what makes the
system runnable by someone who has none of the optional infrastructure.

```
Modal not configured        → Specialist skipped, weights renormalise over the rest
No trained weights          → Neural skipped, same
Pushover not configured     → alerts composed and logged, not sent
An estimator throws mid-run → dropped for the remainder of that run
A deal page fails to load   → that listing skipped, scan continues
Structured output fails     → exception surfaces (this one should not be silent)
No vector store             → FileNotFoundError naming the command to run
```

The deliberate line: **missing optional capability degrades; missing required
state raises.** A missing vector store is not something to work around — there is
nothing sensible to do without it — so `DealDiscovery._collection` raises with a
message naming the fix.
