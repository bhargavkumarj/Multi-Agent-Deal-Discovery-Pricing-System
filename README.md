# Multi-Agent Deal Discovery & Pricing

Seven agents that watch retail deal feeds, work out what each item is actually
worth, and send a push notification when something is priced well below that.
It is built to run continuously, so most of the engineering is about not being
annoying: estimates that do not swing wildly, and never alerting twice on the
same product.

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

## The agents

| Agent | Job |
|---|---|
| **Scanner** | Pulls listings from the feeds, scrapes each deal page, and uses structured outputs to shortlist the ones with a real, unambiguous price. Listings phrased as "$200 off" are the main thing it has to get right. |
| **Frontier** | Prices an item by first retrieving five similar products with known prices from a Chroma catalogue, then asking an LLM. The retrieved comparables are what turn a guess into an estimate. |
| **Specialist** | Calls a fine-tuned Llama-3.2-3B pricer served on a Modal GPU. |
| **Neural** | A residual network over hashed text features, running locally. Weakest alone, but cheap and wrong in different ways to the LLM. |
| **Ensemble** | Weighted mean of whichever estimators are available, plus the spread between them as a confidence signal. |
| **Messenger** | Writes the notification and sends it through Pushover. |
| **Planner** | Runs the cycle: scan, drop duplicates, value each candidate, rank by discount, alert on the best one if it clears the threshold. |

## Documentation

A full documentation set lives in [`docs/`](docs/): architecture, a file-by-file
code walkthrough, the concepts behind the implementation, the reasoning for every
non-obvious decision, and 30 interview questions with worked answers. Start with
[`docs/README.md`](docs/README.md) for the reading order.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # OPENAI_API_KEY, optionally PUSHOVER_*

python build_index.py --limit 20000   # build the priced product catalogue
```

The Specialist and Neural agents are optional. Without Modal configured or
trained weights in `artifacts/`, the ensemble logs that they are unavailable and
carries on with what it has.

## Usage

```bash
python run.py                              # one cycle
python run.py --loop --every 600           # keep running
python run.py --no-specialist --no-notify  # no Modal, compose alerts but do not send
python run.py --memory                     # show what has been found so far
python app.py                              # monitoring dashboard
```

The dashboard streams the agent log while a cycle runs, lists everything found so
far with its price, estimate and discount, and plots the product catalogue as a
3D t-SNE projection coloured by category.

To serve the fine-tuned model:

```bash
modal deploy modal_app/pricer_service.py
```

## Design notes

**Ensembling is about disagreement, not accuracy.** The three estimators fail in
different places — the LLM drifts on unusual items, the fine-tuned model is
steady on familiar categories, the neural net is noisy but unbiased. A weighted
mean is more stable than any of them, and the spread between them is recorded on
every opportunity as a confidence signal. A $500 estimate that all three agree on
is a different thing to a $500 estimate they are $400 apart on.

**Weights renormalise over whatever loaded.** Estimators are optional, so the
weights are applied over the ones that actually initialised rather than assuming
all three are present. An estimator that throws mid-run is dropped for the rest
of that run instead of failing on every remaining deal.

**Dedup uses two keys.** The URL with tracking parameters stripped, and a content
key built from the significant words of the description plus the price rounded to
the nearest ten. The URL alone is not enough: the same product appears on several
feeds with different URLs, and in testing the scanner shortlisted one item twice
in a single pass. Both keys are checked against memory *and* within the batch.

**One alert per cycle.** Everything priced is written to memory, but only the
single best opportunity above the threshold triggers a notification. The value of
this system is a small number of alerts worth opening.

**Retrieval before generation.** The Frontier agent is the strongest estimator
precisely because it does not ask the model to recall prices. It shows the model
five comparable products and what they sold for, which is a question a language
model can actually answer.

## Verified

Run end to end on this machine: feed scraping, the scanner's structured-output
shortlisting, RAG pricing against a 4,000-product Chroma index, the neural
estimator, the two-estimator ensemble, threshold logic, alert composition, memory
persistence and dedup across runs. A second cycle correctly skipped the listings
already in memory.

Exercised with local models through an OpenAI-compatible endpoint
(`OPENAI_BASE_URL=http://localhost:11434/v1`), which is also a supported way to
run it. The Specialist agent needs a Modal account and a deployed fine-tuned
model; without those it is skipped, and that path was confirmed to degrade
cleanly rather than break the run. Pushover delivery needs credentials; without
them alerts are composed and logged.

Testing surfaced two bugs worth naming: the neural estimator's target scaler was
loading onto a different device to the tensor it was applied to, and a failing
estimator was being retried on every deal. Both are fixed.
