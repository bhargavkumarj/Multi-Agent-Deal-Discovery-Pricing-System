# Code Walkthrough — Configuration, Logging, Memory, Framework

Covers `core/config.py`, `core/logs.py`, `core/memory.py` and
`core/framework.py`.

---

## `core/config.py`

### The feeds

```python
FEEDS = [
    "https://www.dealnews.com/c142/Electronics/?rss=1",
    "https://www.dealnews.com/c39/Computers/?rss=1",
    "https://www.dealnews.com/f1912/Smart-Home/?rss=1",
]
```

Three categories, chosen to match what the product catalogue contains. Retrieval
only helps if the catalogue holds comparable items — pointing the scanner at a
Garden feed while the index holds Electronics would produce poor comparables and
bad estimates. The feeds and the index are a matched pair.

```python
feeds: list[str] = field(default_factory=lambda: list(FEEDS))
```

`default_factory` returning a **copy**. A mutable default in a dataclass is
shared across instances; without the copy, mutating `settings.feeds` would edit
the module-level constant.

### `_float` and `_flag`

```python
def _float(name: str, default: float) -> float:
    return float(os.getenv(name, default))

def _flag(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}
```

`_flag` exists because `bool("false")` is `True` in Python. A naive conversion
would silently invert `NOTIFY=false` — the system would send notifications while
the config said not to.

`_float` relies on `float()` accepting both a string from the environment and the
numeric default.

### The OpenAI-compatible client

```python
api_base: str | None = os.getenv("OPENAI_BASE_URL") or None

def openai_client():
    from openai import OpenAI

    if settings.api_base:
        return OpenAI(base_url=settings.api_base, api_key=os.getenv("OPENAI_API_KEY", "local"))
    return OpenAI()
```

This small function is what made the project testable.

Setting `OPENAI_BASE_URL=http://localhost:11434/v1` points every agent at Ollama,
which speaks the OpenAI protocol. The entire system then runs locally: Scanner,
Frontier and Messenger all work, including `chat.completions.parse` with
structured outputs, which Ollama supports.

That is how the measured end-to-end runs were done, and it is a genuine
deployment option — local models, vLLM, or a corporate gateway all work the same
way.

`os.getenv("OPENAI_API_KEY", "local")` supplies a placeholder because the client
requires *some* key, and local endpoints ignore it.

`or None` converts an empty string to `None`, so `OPENAI_BASE_URL=` in a `.env`
does not produce a client pointed at the empty string.

### The ensemble weights

```python
weight_frontier: float = _float("WEIGHT_FRONTIER", 0.6)
weight_specialist: float = _float("WEIGHT_SPECIALIST", 0.25)
weight_neural: float = _float("WEIGHT_NEURAL", 0.15)
```

They sum to 1.0, but nothing enforces that — `EnsembleAgent` divides by the sum
of the weights actually used, so any positive numbers work as relative weights.

The ordering reflects measured quality. Frontier is strongest because retrieval
grounds it. The Specialist is a fine-tuned model with real price knowledge but no
retrieval. The Neural agent is weakest alone — around $71 average error in the
price prediction project — but cheap and independently wrong, which is what makes
it worth 15%.

These are **hand-set priors, not learned weights**, and that is a fair
criticism. With ground-truth prices you could fit them by regression. The honest
position is that the defaults are informed by measured per-model performance, and
the mechanism for tuning them exists.

### Thresholds

```python
discount_threshold: float = _float("DISCOUNT_THRESHOLD", 50)
similar_products: int = int(os.getenv("SIMILAR_PRODUCTS", "5"))
deals_per_feed: int = int(os.getenv("DEALS_PER_FEED", "10"))
shortlist_size: int = int(os.getenv("SHORTLIST_SIZE", "5"))
```

`discount_threshold` is the product decision: how good does a deal have to be
before it is worth interrupting someone? An absolute dollar amount rather than a
percentage, because $50 off a $60 item and $50 off a $600 item are both worth
knowing, while 30% off a $10 item is not.

`shortlist_size` bounds cost. Each valuation is one to three model calls, so five
is the ceiling on spend and latency per cycle.

---

## `core/logs.py`

### `configure()`

```python
def configure(level: int = logging.INFO) -> None:
    root = logging.getLogger()
    if any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        return
    handler = logging.StreamHandler(sys.stdout)
    ...
```

**The idempotence guard is the point.** `DealDiscovery.__init__` calls
`configure()`, and so does `run.py --memory`. Without the check, constructing the
framework twice — which the Gradio app does on reload — would attach two handlers
and print every line twice.

`sys.stdout` rather than the default `stderr`, so the output pipes and greps
normally.

### `to_html()`

```python
HTML_COLOURS = {
    BG_BLACK + RED: "#dd0000",
    ...
    BG_BLUE + WHITE: "#ff7800",
}

def to_html(message: str) -> str:
    for code, colour in HTML_COLOURS.items():
        message = message.replace(code, f'<span style="color: {colour}">')
    return message.replace(RESET, "</span>")
```

The agents log with ANSI escape codes. The dashboard needs HTML. Rather than
having agents log twice or carry a format flag, the terminal codes are translated
at the display boundary.

The keys are **two-code sequences** (`BG_BLACK + RED`), which is why agents emit
background and foreground together. Matching on the pair rather than on the
foreground alone avoids ambiguity, and `BG_BLUE + WHITE` gives the framework its
own distinct colour that no agent uses.

Each opening code becomes a `<span>` and every `RESET` becomes `</span>`, so the
tags balance as long as agents always emit the pair. They do, because `Agent.log`
is the only thing that emits them.

---

## `core/memory.py`

The most subtle file in the project. Deduplication is what makes a continuously
running system usable.

### The problem

The same product appears:

- **Across cycles** — the feed still lists it an hour later.
- **Across feeds** — Electronics and Computers both carry the same monitor, with
  different URLs.
- **Within one batch** — the Scanner shortlists it twice, with slightly different
  wording. Observed in a measured run.
- **With different URLs** — tracking parameters, `?iref=rss-c142` versus
  `?iref=rss-c39`.

A URL-only check catches the first case and misses the rest.

### `url_key()`

```python
def url_key(deal: Deal) -> str:
    return deal.url.split("?")[0].rstrip("/").lower()
```

Strip the query string, strip a trailing slash, lowercase. Three normalisations
that make `.../product/12345?iref=rss-c142` and
`.../Product/12345/` the same key.

The query string is where the tracking parameters live, and they differ purely by
which feed served the link — the same product page, two different URLs.

### `content_key()`

```python
words = re.findall(r"[a-z0-9]+", deal.product_description.lower())
significant = sorted({w for w in words if w not in STOPWORDS and len(w) > 2})[:10]
return " ".join(significant) + f"|{round(deal.price / 10) * 10}"
```

A fuzzy fingerprint of the product itself. Five decisions:

**`re.findall(r"[a-z0-9]+")` on lowercased text** — tokenise to alphanumerics,
dropping punctuation and case.

**Stopword and length filtering** — remove "the", "with", "new", "refurbished",
and anything three characters or shorter. What survives is the distinctive
vocabulary.

**`sorted(set(...))`** — a **set** so repeated words count once, and **sorted**
so two descriptions with the same vocabulary in different order produce the same
key. This is what makes the key robust to rewording: *"A portable monitor with a
15.6-inch 1080p display"* and *"A portable monitor that doubles your screen real
estate"* share enough sorted vocabulary to collide.

**`[:10]`** — the first ten alphabetically. Bounds the key length and, because
it is deterministic, two descriptions sharing their first ten sorted significant
words produce identical keys.

**`round(price / 10) * 10`** — the price bucketed to the nearest $10. Prices
drift slightly between listings of the same item, and exact matching would let a
$64.99 versus $65.00 difference defeat the whole check. Bucketing also
disambiguates: two genuinely different products with similar descriptions but
very different prices get different keys.

The `|` separator prevents the price digits merging with the last word.

### `keys()` and the set-intersection test

```python
def keys(deal: Deal) -> set[str]:
    return {key for key in (url_key(deal), content_key(deal)) if key}

def has_seen(self, deal: Deal) -> bool:
    return bool(keys(deal) & self.seen)
```

Each deal gets **both** keys, and `has_seen` is a set intersection — a match on
*either* is enough.

That is the right logic. The same URL is definitely the same deal. The same
content key is very probably the same product. Requiring both would miss
cross-feed duplicates entirely, which is the case that motivated the second key.

The `if key` filter drops empty strings, so a deal with no URL contributes only
its content key rather than an empty string that would match every other
URL-less deal.

### Two records: seen and alerted

```python
@property
def seen(self) -> set[str]:
    return {key for o in self.opportunities for key in keys(o.deal)}

def mark_alerted(self, opportunity: Opportunity) -> None:
    self.alerted |= keys(opportunity.deal)
```

`seen` is derived from the stored opportunities; `alerted` is stored separately.

They answer different questions. *Seen* means "do not price this again" — it
covers everything valued, including deals rejected for being overpriced.
*Alerted* means "do not notify about this again" — a much smaller set.

`seen` being a computed property means it cannot drift out of sync with the
opportunity list. It is O(n) per call, which is fine at this scale and would need
caching at a hundred thousand records.

### Backwards-compatible loading

```python
data = json.loads(self.path.read_text())
rows = data.get("opportunities", []) if isinstance(data, dict) else data
self.opportunities = [Opportunity(**row) for row in rows]
self.alerted = set(data.get("alerted", [])) if isinstance(data, dict) else set()
```

Handles both the current format (`{"opportunities": [...], "alerted": [...]}`)
and a bare list from an earlier version. Cheap insurance against a schema change
making an existing memory file unreadable — which, for a system whose value
accumulates in that file, would be a genuine loss.

### Saving on every write

```python
def add(self, opportunity: Opportunity) -> bool:
    if self.has_seen(opportunity.deal):
        return False
    self.opportunities.append(opportunity)
    self.save()
    return True
```

Every `add` and every `mark_alerted` writes the whole file. Inefficient — but
this is a process that runs for hours or days and can be killed at any moment,
and losing the memory means re-alerting on everything. Durability is worth far
more than the write cost at this volume.

`add` returns a bool so a caller can tell whether it was new. The Planner
pre-filters, so this is a second line of defence.

---

## `core/framework.py`

### What `DealDiscovery` owns

```python
def __init__(self, **ensemble_options):
    logs.configure()
    self.memory = Memory()
    self.collection = self._collection()
    self.ensemble_options = ensemble_options
    self.planner: PlanningAgent | None = None
```

Three long-lived resources — logging, memory, the Chroma collection — created
once and shared. `ensemble_options` is stashed rather than applied, because the
planner is built lazily.

### Failing loudly on missing required state

```python
def _collection(self):
    if not settings.vector_store.exists():
        raise FileNotFoundError(
            f"No product vector store at {settings.vector_store}. "
            "Run `python build_index.py` first."
        )
```

This is the deliberate counterpoint to all the graceful degradation elsewhere.

A missing Specialist is workable — the ensemble carries on. A missing vector
store is not: the Frontier agent is the core estimator and without comparables it
is a zero-shot guesser, which the measured results show is worse than useless.

`get_or_create_collection` would silently produce an empty collection and the
system would appear to work while every estimate was a hallucination. Raising
with the exact command to run is the correct behaviour.

**The principle: missing optional capability degrades, missing required state
raises.**

### Lazy agent construction

```python
def start(self) -> PlanningAgent:
    if self.planner is None:
        self.log("Starting the agents")
        self.planner = PlanningAgent(self.collection, self.memory, **self.ensemble_options)
    return self.planner
```

Constructing the planner constructs all seven agents, which loads a
sentence-transformer, possibly a 13.5M-parameter network, and possibly a Modal
connection. Several seconds.

Doing that lazily means `run.py --memory` constructs a `Memory`, prints it and
exits without loading a single model. The dashboard also benefits — it can render
the catalogue plot before the agents are warm.

Once constructed the planner is cached, so a `--loop` run pays the cost once
rather than per cycle.

### `cycle()`

```python
def cycle(self) -> Opportunity | None:
    planner = self.start()
    self.log(f"Cycle starting, {len(self.memory)} deals in memory")
    result = planner.run()
    self.log("Cycle complete" + (f": alerted on ${result.discount:,.2f} off" if result else ""))
    return result
```

Bookended logging with the memory size, so a `--loop` transcript shows memory
growing — which is how you confirm deduplication is doing its job. Measured
across two cycles: `0 deals in memory` then `5 deals in memory`, with the second
cycle's scanner reporting `9 listings, 5 not seen before`.

The conditional suffix means the completion line states the outcome, so grepping
for `Cycle complete:` finds exactly the cycles that alerted.

### `plot_data()`

```python
import numpy as np
from sklearn.manifold import TSNE

result = self.collection.get(include=["embeddings", "documents", "metadatas"], limit=limit)
vectors = np.array(result["embeddings"])
categories = [metadata["category"] for metadata in result["metadatas"]]
reduced = TSNE(n_components=3, random_state=42, init="pca").fit_transform(vectors)
```

Projects 384-dimensional product embeddings down to 3 for display.

**Imports inside the function** — sklearn's t-SNE and numpy are only needed by
the dashboard, and `run.py` should not pay for them.

**`limit=1500`** — t-SNE is roughly O(n²) and would take minutes on the full
catalogue while blocking the UI.

**`random_state=42`** — t-SNE is stochastic and the layout would otherwise change
on every page load, which makes it useless as a reference view.

**`init="pca"`** initialises from a PCA projection rather than randomly. More
stable and it preserves global structure better; the sklearn default changed to
this for good reason.

t-SNE is for *visual* inspection only. Distances in the projection are not
meaningful and the axes mean nothing — but clusters by category are, and seeing
Electronics separate from Musical Instruments is a quick confirmation that the
embeddings carry real signal.
