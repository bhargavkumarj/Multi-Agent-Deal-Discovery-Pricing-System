# Concepts Explained

The ideas behind the system, tied back to the code and the measured runs.

---

## 1. What makes something a multi-agent system

### The honest definition

An agent here is a component with its own model or tool, its own prompt or
parameters, and its own failure mode. It is *not* just a function with a nice
name.

The test: could you replace this component with a different implementation
without touching anything else? For all seven, yes. Swap the Scanner's model,
replace the Neural estimator with a different network, change Pushover for Slack
— each is one file.

### Why seven and not one function

The decomposition follows the *kind of work*, because different kinds of work
fail differently:

| Agent | Work | Fails by |
|---|---|---|
| Scanner | Extraction from noisy text | Misreading "$200 off" as the price |
| Frontier | Retrieval + reasoning | Poor comparables for unusual items |
| Specialist | Remote GPU inference | Network, auth, cold start |
| Neural | Local numerical inference | Missing weights, device mismatch |
| Ensemble | Aggregation | Weight mis-normalisation |
| Messenger | Generation + I/O | Delivery failure |
| Planner | Policy | Wrong threshold, bad ranking |

Two properties fall out, and they are the practical argument:

**Independent failure.** The Specialist needs Modal, the Neural agent needs
weights. Neither is required — the Ensemble renormalises over whichever loaded.
Measured: with Modal unconfigured, `Ready with frontier, neural`.

**Independent substitution.** Adding a fourth estimator is one class with a
`price()` method plus three lines in `EnsembleAgent.__init__`.

### When this would be over-engineering

If all three estimators were the same model with different prompts, this would be
ceremony. It is justified here because they are genuinely different systems — an
API call with retrieval, a remote fine-tuned model, and a local network — with
different dependencies, costs and failure modes.

### What this is *not*

It is not an autonomous agent loop where an LLM decides which tool to call next.
The Planner's sequence is hard-coded: scan, dedup, value, rank, alert.

That is a deliberate choice. The workflow is known, fixed and correct. Letting an
LLM decide the order would add latency, cost and nondeterminism to a pipeline
with no branching decisions to make. LLM-driven planning earns its place when the
sequence genuinely varies by input; here it does not.

---

## 2. Structured outputs

### The problem

Early LLM integrations asked for JSON in the prompt and parsed the reply. Every
failure mode you can imagine occurs: markdown fences around the JSON, a trailing
explanation, single quotes, a missing field, a string where a number belongs.

### The mechanism

```python
class Deal(BaseModel):
    product_description: str = Field(description="...")
    price: float = Field(description="...")
    url: str = Field(description="...")

response = client.chat.completions.parse(..., response_format=DealSelection)
selection = response.choices[0].message.parsed
```

The pydantic model is converted to a JSON schema and sent with the request. The
provider constrains generation so only tokens producing schema-valid output are
sampled. The reply is parsed and validated into typed objects.

Result: `selection.deals[0].price` is a `float`. Not a string, not `None`, not a
`KeyError`.

### `Field` descriptions are prompts

This is the part people miss. The `description` argument is included in the
schema sent to the model:

```python
price: float = Field(
    description="The price being charged for the item. If the listing says '$100 off the "
    "usual $300', the price is 200."
)
```

That worked example is an instruction attached to the field it governs, which is
more reliable than the same sentence buried in a system prompt — the model sees
it at the moment it generates that field.

It matters here because it is the highest-consequence extraction in the system. A
model returning the *saving* instead of the *price* produces a wrong discount,
which produces a confident, useless alert.

### Validation is still needed

```python
selection.deals = [deal for deal in selection.deals if deal.price > 0]
```

Structured outputs guarantee the *shape*, not the *semantics*. A price of 0.0 is
schema-valid and meaningless. Worse, it would compute a discount equal to the
entire estimate and rank first.

**Schema constrains form; code constrains meaning.**

---

## 3. RAG for pricing

### The failure this fixes

Measured in the price prediction project, a zero-shot LLM asked to price products
returned:

```
Projector with WiFi       truth $53.40   guess $4,500.00
Toner Cartridge           truth $49.98   guess $2,500.00
1000 Piece Jigsaw Puzzle  truth $79.95   guess $1,500.00
```

Average error $264 — worse than predicting the training mean. Note the round
numbers. That is the signature of *guessing*, not estimating.

The model has never seen this product's current market price. Asked to recall
something it does not know, it produces a plausible-shaped answer.

### The reframing

Retrieval changes the question:

> **Without:** "What does this product cost?" — requires knowledge the model lacks.
>
> **With:** "Here are five similar products and what they sold for. What does
> this one cost?" — requires interpolation, which the model is good at.

The second question is answerable from the context alone.

### The implementation

```python
vector = self.encoder.encode([description]).astype(float).tolist()
results = self.collection.query(query_embeddings=vector, n_results=5)
documents = results["documents"][0]
prices = [metadata["price"] for metadata in results["metadatas"][0]]
```

Embed the deal description, find the 5 nearest products in a catalogue of 4,000
priced items, and put their descriptions and prices in the prompt.

**The price in the metadata is the whole point.** Retrieving five similar
*descriptions* would be useless. Retrieving five similar descriptions *and what
they sold for* provides the anchor.

### The hard requirement

The deal description is embedded with the same model that embedded the catalogue
(`all-MiniLM-L6-v2`). Vectors from different models are not comparable —
dimension 142 of one model and dimension 142 of another are unrelated numbers.
Chroma would return nearest neighbours in a meaningless sense, with no error,
because the arithmetic is valid.

### The remaining weakness

Retrieval only helps when the catalogue contains genuinely comparable items. For
a product unlike anything indexed, the five "nearest" neighbours are just the
least-distant of a bad set, and the anchor is misleading.

This is why the feeds and the catalogue are a matched pair — both Electronics,
Computers and Smart Home — and why the ensemble includes estimators that do not
depend on retrieval at all.

---

## 4. Ensembling

### Why it works

Combining models helps when their errors are **uncorrelated**. If two models are
wrong in the same direction on the same items, averaging them changes nothing. If
they are wrong in different directions, the errors partially cancel.

### The three estimators, and why they are different

| Estimator | Mechanism | Strong on | Weak on |
|---|---|---|---|
| Frontier | LLM + retrieved comparables | Anything with good comparables | Novel items where retrieval fails |
| Specialist | Fine-tuned Llama-3.2-3B | Categories seen in fine-tuning | Anything outside them |
| Neural | Hashed bag-of-words → residual MLP | Broad price range, no semantics needed | Anything needing understanding |

The important property is that these are *not variations of one approach*. One
retrieves, one has knowledge in its weights, one does pure statistical pattern
matching on word presence. They fail in genuinely different places, which is what
makes averaging them worth doing.

Measured, on the same deal:

```
[Frontier] Estimates $189.00
[Neural]   Estimates $212.97
[Ensemble] frontier $189, neural $213 -> $193.79
```

### Weighted mean, and renormalisation

```python
total = sum(self.weights[e.source] for e in estimates)
combined = sum(e.value * self.weights[e.source] for e in estimates) / total
```

Weights 0.6 / 0.25 / 0.15, reflecting measured quality — Frontier strongest
because retrieval grounds it, Neural weakest alone at around $71 average error.

**Renormalising over the estimators that actually returned a value** is the
subtle part. Without it, losing the Specialist would shrink every estimate by
25%: the numerator drops while the denominator stays at 1.0. Every estimate would
be biased low, every discount would look smaller, and the system would quietly
stop alerting with no error anywhere.

Renormalisation happens **per item**, because an estimator can fail on one
description and succeed on the next.

### Spread as a confidence signal

```python
@property
def spread(self) -> float:
    values = [e.value for e in self.estimates]
    return max(values) - min(values) if len(values) > 1 else 0.0
```

The range between the highest and lowest estimator. A $500 estimate all three
agree on is a different claim to a $500 estimate they span $400 on.

It is recorded on every opportunity and persisted. The current threshold logic
does not use it — deliberately. The signal is cheap to collect and the right way
to act on it (a wider threshold when spread is high? a confidence band in the
alert?) should be decided from data, not guessed.

### The honest limitation

The weights are **hand-set priors, not learned**. With ground-truth prices for
scraped deals you could fit them by regression and get a genuinely optimal
combination. The defaults are informed by measured per-model performance, and the
mechanism to tune them exists, but they are not optimal and it would be wrong to
claim otherwise.

---

## 5. Deduplication

The unglamorous part that determines whether the system is usable.

### Why it is hard

The same product appears:

- **Across cycles** — the feed still lists it an hour later.
- **Across feeds** — Electronics and Computers both carry the same monitor.
- **With different URLs** — tracking parameters differ by source feed:
  `?iref=rss-c142` versus `?iref=rss-c39`.
- **Within one batch** — the Scanner shortlists it twice with different wording.
  Observed in a measured run.

A URL check catches the first and misses the rest.

### Two keys

```python
def url_key(deal):
    return deal.url.split("?")[0].rstrip("/").lower()

def content_key(deal):
    words = re.findall(r"[a-z0-9]+", deal.product_description.lower())
    significant = sorted({w for w in words if w not in STOPWORDS and len(w) > 2})[:10]
    return " ".join(significant) + f"|{round(deal.price / 10) * 10}"
```

**`url_key`** is exact. Stripping the query string is what makes the same product
page from two feeds match.

**`content_key`** is fuzzy. Five choices make it robust:

1. **Lowercase, alphanumeric tokens** — punctuation and case do not matter.
2. **Stopwords and short words removed** — what remains is distinctive vocabulary.
3. **`set`** — repeated words count once.
4. **`sorted`** — word *order* does not matter, so two rewordings of the same
   product produce the same key.
5. **Price bucketed to $10** — `round(price / 10) * 10`. Prices drift slightly
   between listings, and exact matching would let $64.99 versus $65.00 defeat the
   whole check. The bucket also disambiguates two different products with similar
   descriptions but very different prices.

```python
def has_seen(self, deal):
    return bool(keys(deal) & self.seen)
```

A match on **either** key is enough. Same URL is definitely the same deal; same
content key is very probably the same product. Requiring both would miss
cross-feed duplicates, which is the case the second key exists for.

### Two records, not one

*Seen* means "do not price this again" and covers everything valued, including
deals rejected for being overpriced. *Alerted* means "do not notify about this
again" and is a much smaller set.

Conflating them would mean either re-pricing rejected deals forever, or never
reconsidering something whose price changed.

### Within-batch deduplication

```python
candidates, batch_keys = [], set()
for deal in selection.deals:
    deal_keys = keys(deal)
    if self.memory.has_seen(deal) or deal_keys & batch_keys:
        continue
    batch_keys |= deal_keys
    candidates.append(deal)
```

Memory covers across-run duplicates. `batch_keys` covers within-run duplicates,
which memory structurally cannot catch because neither has been seen before.

This check was added after observing the Scanner shortlist the same portable
monitor twice in one pass. Without it, the system would price it twice and could
alert on it twice in the same cycle.

---

## 6. Serverless GPU inference

### The economics

A fine-tuned Llama-3.2-3B needs a GPU. This system makes a handful of calls every
ten minutes.

A dedicated GPU instance costs the same whether it serves one request an hour or
ten thousand. For this workload, utilisation would be a fraction of a percent.

```python
min_containers=0
```

Zero idle cost. The container spins up on the first call and shuts down after a
quiet period. You pay for seconds of GPU time, not hours of availability.

The trade is **cold-start latency** — the first call after a quiet period waits
for a container, and for the model to load. For a system that runs on a
ten-minute timer, that is invisible. For an interactive product it would not be,
and you would set `min_containers=1` and pay for it.

### Making cold starts survivable

```python
cache = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)

@app.cls(image=image.env({"HF_HUB_CACHE": CACHE}), volumes={CACHE: cache}, ...)
```

Without the volume, every cold start downloads a 3B model from HuggingFace —
gigabytes, and minutes. The persistent volume mounted at the HF cache path means
the download happens once ever.

```python
@modal.enter()
def load(self):
    ...
```

Runs **once per container**, not per request. Model loading is the expensive
part; inference is cheap. Loading in `price()` would reload the model on every
call and make the whole design pointless.

### 4-bit quantisation

```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
)
```

A 3B model in fp16 is about 6GB. In 4-bit it is under 2GB, which fits comfortably
on a T4 — the cheapest useful GPU available.

- **`nf4`** (NormalFloat4) is the information-theoretically optimal 4-bit data
  type for normally-distributed weights, which neural network weights
  approximately are.
- **Double quantisation** compresses the quantisation constants themselves,
  saving a further ~0.4 bits per parameter.
- **`compute_dtype=float16`** means weights are *stored* in 4-bit but matmuls
  still run in fp16, so throughput stays reasonable.

Quality loss from 4-bit NF4 is small for inference. For a price estimate feeding
an ensemble, it is well within tolerance.

### LoRA

```python
self.model = PeftModel.from_pretrained(base, FINETUNED_MODEL, revision=REVISION)
```

Low-Rank Adaptation freezes the base model and trains small rank-decomposition
matrices injected into the attention layers. The fine-tuned artefact is megabytes
rather than gigabytes — so the base model is cached once and the adapter is the
only thing that changes between model versions.

**`revision=REVISION`** pins a commit hash. A mutable branch reference would let
the model change under a running service, silently, with no deployment.

---

## 7. Graceful degradation, and where to draw the line

### The pattern

```python
try:
    from agents.specialist import SpecialistAgent
    self.estimators["specialist"] = SpecialistAgent()
    self.weights["specialist"] = settings.weight_specialist
except Exception as error:
    self.log(f"Specialist unavailable, skipping it ({error})")
```

Missing package, missing credentials, missing weights — all treated the same:
that estimator is unavailable, the system runs with the rest.

### Where it stops

```python
if not settings.vector_store.exists():
    raise FileNotFoundError(
        f"No product vector store at {settings.vector_store}. "
        "Run `python build_index.py` first."
    )
```

A missing vector store **raises**. There is no sensible behaviour without it —
`get_or_create_collection` would produce an empty collection and the Frontier
agent would degrade to exactly the zero-shot guessing the measured results show
is worse than useless. The system would appear to work while every estimate was a
hallucination.

**The principle: missing optional capability degrades, missing required state
raises.** The test is whether there is a correct behaviour without the thing.

### Failing estimators are dropped, not retried

```python
except Exception as error:
    self.log(f"{source} failed, dropping it for this run: {error}")
    del self.estimators[source]
```

The first version logged and continued, which meant retrying a permanently broken
estimator on every deal — five identical failures per cycle, each with the
latency of a failed connection.

Dropping for the run, not permanently: a new `EnsembleAgent` is built per
framework start, so a transient failure gets another chance next process while a
persistent one stops costing time immediately.

---

## 8. t-SNE

Used in the dashboard to project 384-dimensional product embeddings to 3
dimensions.

### What it does

t-distributed Stochastic Neighbour Embedding converts pairwise distances into
probabilities and finds a low-dimensional layout whose probability distribution
matches the high-dimensional one, minimising KL divergence.

Practically: **points close in the original space stay close in the projection.**

### What it does not do

- **Distances are not meaningful.** Twice as far apart does not mean twice as
  dissimilar.
- **Axes mean nothing.** They are the reason `app.py` blanks the axis titles —
  labelling them would imply a meaning they do not have.
- **Cluster sizes are not meaningful.** t-SNE expands dense regions and contracts
  sparse ones.

Only the **cluster structure** is interpretable.

### Why it is in the dashboard

Seeing Electronics separate from Musical Instruments is a quick confirmation that
the embeddings carry real category signal, and therefore that retrieval is
finding semantically related products rather than noise. It is a sanity check you
can take in at a glance.

```python
TSNE(n_components=3, random_state=42, init="pca").fit_transform(vectors)
```

`random_state=42` because t-SNE is stochastic and the layout would otherwise
change on every page load, which makes it useless as a reference view.

`init="pca"` initialises from a PCA projection rather than randomly — more stable
and it preserves global structure better.

`limit=1500` because t-SNE is roughly O(n²) and would block the UI for minutes on
the full catalogue.

---

## 9. Designing for continuous operation

Most of what separates this from a script is the handling of *running forever*.

| Concern | Handling |
|---|---|
| Duplicate alerts | Two-key dedup, memory persisted on every write |
| Unbounded memory growth | `best(limit)` caps display; the file would need pruning at scale |
| Network hangs | `timeout=15` on every HTTP call — without it one dead server hangs the loop forever |
| Broken estimator | Dropped for the run after the first failure |
| Crash mid-cycle | Memory is written on every `add`, so at most the current deal is lost |
| Alert fatigue | One alert per cycle, above a threshold, ranked |
| Cost | `shortlist_size` bounds valuations per cycle; `deals_per_feed` bounds scraping |
| Observability | Every agent logs with its own name and colour; the dashboard renders the same stream |

The timeout is the one most often forgotten. `requests.get` without a timeout
waits indefinitely by default. In a `--loop` deployment, one unresponsive deal
page stops the system permanently, with no error and no output — the worst
possible failure mode because it looks like nothing is wrong.
