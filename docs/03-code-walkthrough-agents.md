# Code Walkthrough — The Agents

Covers `agents/base.py`, `deals.py`, and all seven agents.

---

## `agents/base.py` — the shared contract

```python
class Agent:
    name = "Agent"
    colour = logs.WHITE

    def log(self, message: str) -> None:
        logging.info(f"{logs.BG_BLACK}{self.colour}[{self.name}] {message}{logs.RESET}")
```

Eleven lines, and there is no abstract `run()` or `act()` method. That is
deliberate: the seven agents genuinely do different things. `ScannerAgent.scan`,
`FrontierAgent.price` and `MessagingAgent.alert` have nothing in common except
that they log.

Forcing a shared `execute()` would mean seven methods with the same name doing
unrelated things behind `**kwargs` — an abstraction that costs clarity and buys
nothing, because no code ever iterates over a heterogeneous list of agents.

The three estimators *do* share a `price(description) -> float` shape, and
`EnsembleAgent` relies on it via duck typing. That contract is real and it is
also implicit, which is a fair thing to criticise — a `Pricer` protocol would
document it.

**What the base class does buy** is a readable transcript. With seven agents
interleaving output, every line is prefixed and coloured by origin:

```
[21:03:39] [Neural] Estimates $212.97
[21:03:39] [Ensemble] frontier $189, neural $213 -> $193.79
[21:03:46] [Planner] Best candidate is $104.76 under estimate (estimator spread $64)
```

---

## `agents/deals.py` — the data that moves between agents

### `clean_html()`

```python
soup = BeautifulSoup(snippet, "html.parser")
block = soup.find("div", class_="snippet summary")
text = block.get_text(strip=True) if block else soup.get_text(" ", strip=True)
return re.sub(r"\s+", " ", text).strip()
```

Tries the specific container first, falls back to all text in the snippet. The
fallback matters because feed HTML changes without notice, and a scraper that
returns nothing when a CSS class is renamed is worse than one that returns
slightly noisier text.

`re.sub(r"\s+", " ", text)` collapses every whitespace run into one space. The
output goes into a prompt alongside 29 other listings, so whitespace is tokens
wasted.

### `ScrapedDeal`

```python
def __init__(self, title, summary, url, details="", features=""):
    self.title = title[:100]
    self.details = details[:500]
    self.features = features[:500]
```

Truncation at construction, not at use. Thirty listings × 1,100 characters ≈
33,000 characters of prompt. Uncapped, one verbose listing could consume the
whole context and crowd out the other 29.

```python
@classmethod
def from_entry(cls, entry) -> "ScrapedDeal | None":
    url = entry["links"][0]["href"]
    details, features = "", ""
    try:
        page = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        soup = BeautifulSoup(page.content, "html.parser")
        section = soup.find("div", class_="content-section")
        if section:
            content = section.get_text().replace("\nmore", "").replace("\n", " ")
            if "Features" in content:
                details, features = content.split("Features", 1)
            else:
                details = content
    except requests.RequestException:
        return None
    return cls(entry["title"], clean_html(entry["summary"]), url, details, features)
```

The RSS summary is short, so each entry's page is fetched for the real
description.

**`timeout=TIMEOUT`** (15s) is not optional. Without a timeout, `requests` waits
indefinitely, and one unresponsive server would hang the entire cycle — in a
`--loop` deployment, forever.

**A custom `User-Agent`** because some sites reject the default `python-requests`
string.

**`split("Features", 1)`** with `maxsplit=1` — a description mentioning
"Features" later would otherwise produce three parts and raise on unpacking.

**Returning `None` on a network error** is the key decision. A deal page that
will not load is not worth failing a scan of 30 listings for. `fetch` filters
them out.

Catching `requests.RequestException` specifically, not bare `Exception` — a
`KeyError` on a malformed feed entry is a genuine bug and should surface.

### `describe()`

```python
return (f"Title: {self.title}\nDetails: {self.details.strip()}\n"
        f"Features: {self.features.strip()}\nURL: {self.url}")
```

The labelled format the Scanner's prompt consumes. Labels help the model separate
30 concatenated listings, and including the URL is what lets the model return it
in the structured output — the Scanner never has to map results back to inputs.

### `Deal` — where the prompt engineering actually lives

```python
class Deal(BaseModel):
    product_description: str = Field(
        description="Three or four sentences describing the product itself. Describe what the "
        "item is and what it does, not the terms of the discount."
    )
    price: float = Field(
        description="The price being charged for the item. If the listing says '$100 off the "
        "usual $300', the price is 200."
    )
    url: str = Field(description="The URL of the deal, exactly as given in the input")
```

These `Field` descriptions are **sent to the model as part of the JSON schema**.
They are instructions, not documentation.

The `price` description carries the single most important rule in the system.
Deal feeds constantly phrase things as "$100 off" or "reduced by $200", and a
model that returns the *saving* instead of the *price* produces a wildly wrong
discount calculation downstream — which is exactly the kind of error that
generates a confident, useless alert. Putting a worked example in the schema
description is more reliable than putting it only in the prompt, because the
schema is attached to the field it governs.

The `product_description` instruction fights a different failure: models
naturally summarise the *deal* ("40% off this weekend only") rather than the
*product*. But the description is what gets embedded for retrieval and priced by
the estimators, so it must describe the item.

### `Estimate` and `Opportunity`

```python
class Estimate(BaseModel):
    source: str
    value: float

class Opportunity(BaseModel):
    deal: Deal
    estimate: float
    discount: float
    estimates: list[Estimate] = Field(default_factory=list)
    found_at: str | None = None

    @property
    def spread(self) -> float:
        values = [e.value for e in self.estimates]
        return max(values) - min(values) if len(values) > 1 else 0.0
```

**Keeping the individual estimates, not just the combined number,** is what makes
`spread` possible. Spread is a cheap, genuine confidence signal: three estimators
within $20 of each other is a different claim to three estimators spanning $400.

It is recorded on every opportunity and persisted to memory, so it is available
for later analysis even though the current threshold logic does not use it. That
is a deliberate choice — collect the signal now, decide how to act on it once
there is data.

`found_at` is an ISO-8601 UTC string rather than a `datetime` so it serialises to
JSON without a custom encoder.

---

## Agent 1 — `agents/scanner.py`

### The system prompt

```
Be careful with listings phrased as "$XXX off" or "reduced by $XXX": that is the
saving, not the price. If you cannot work out what the buyer actually pays, leave
the deal out. Never include a deal with a price of zero.
```

The same rule as the `Field` description, restated at the system level. Belt and
braces on the highest-consequence extraction in the pipeline.

**"If you cannot work out what the buyer actually pays, leave the deal out"** is
the important instruction. It gives the model explicit permission to return fewer
deals. Without it, models fill the requested count — inventing a price is a
*worse* outcome than returning four deals instead of five.

### `fresh_listings()`

```python
scraped = ScrapedDeal.fetch()
fresh = [deal for deal in scraped if deal.url not in seen_urls]
self.log(f"{len(scraped)} listings, {len(fresh)} not seen before")
```

Filtering **before** the LLM call, not after. Three reasons: it is free, it saves
tokens, and it stops the model wasting its shortlist slots on items already
alerted on.

Measured on the second consecutive cycle: `9 listings, 5 not seen before`.

This is a URL-only check — the cheap, exact filter. The fuzzy content-based check
happens later in the Planner, because it needs the structured description that
only exists after the LLM call.

### `scan()`

```python
response = self.client.chat.completions.parse(
    model=self.model,
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ],
    response_format=DealSelection,
)
selection = response.choices[0].message.parsed
selection.deals = [deal for deal in selection.deals if deal.price > 0]
```

**`.parse` with `response_format`** is OpenAI's structured outputs API. The
pydantic model becomes a JSON schema the provider constrains generation against,
and `.parsed` returns typed objects. No regex, no `json.loads` inside a
try/except, no "the model wrapped it in markdown fences" handling.

**The `price > 0` filter** is a second line of defence behind the prompt
instruction. Prompts are probabilistic; a filter is not. A zero-price deal would
compute a discount equal to the entire estimate and rank first — the single most
damaging failure mode in the ranking.

**`if not listings: return None`** short-circuits before the LLM call when
everything has been seen. The Planner treats `None` as "nothing new".

---

## Agent 2 — `agents/frontier.py`

The strongest estimator, and the one that justifies having a vector store.

### Why retrieval

An LLM asked "what does this cost?" with nothing else to go on produces a
plausible round number. Measured in the price prediction project, a zero-shot
local model guessed $4,500 for a $53 projector and $1,500 for an $80 jigsaw
puzzle — round figures, the signature of guessing.

The fix is not a better prompt. No prompt supplies information the model does not
have. The fix is to *give* it the information: five comparable products with
their real prices.

That turns "recall the market price of this product" — which the model cannot do
— into "interpolate between these five priced examples", which it can.

### `similar_products()`

```python
vector = self.encoder.encode([description]).astype(float).tolist()
results = self.collection.query(query_embeddings=vector, n_results=settings.similar_products)
documents = results["documents"][0]
prices = [metadata["price"] for metadata in results["metadatas"][0]]
```

The deal description is embedded with the **same model** that embedded the
catalogue (`all-MiniLM-L6-v2`). That is a hard requirement — vectors from
different models are not comparable, and Chroma would return confident nonsense
rather than an error.

`.astype(float).tolist()` converts numpy float32 to Python floats, because
Chroma's client expects JSON-serialisable values.

`results["documents"][0]` — the `[0]` is the first (only) query's results. The
API supports batch queries and always returns a list per query.

`metadata["price"]` is what `build_index.py` stored alongside each product. The
price is the entire point of the retrieval; the text alone would be useless.

### `context_for()`

```python
lines = ["Here are similar products and what they sell for:\n"]
for document, price in zip(documents, prices):
    lines.append(f"{document}\nPrice is ${price:.2f}\n")
```

Each comparable is presented as description followed by price, in a consistent
shape the model can pattern-match across.

```python
if not documents:
    return ""
```

An empty catalogue degrades to zero-shot rather than producing a prompt with an
empty "here are similar products" header, which would be actively confusing.

### `to_price()`

```python
NUMBER = re.compile(r"[-+]?\d*\.\d+|\d+")

@staticmethod
def to_price(reply: str) -> float:
    match = NUMBER.search(reply.replace("$", "").replace(",", ""))
    return float(match.group()) if match else 0.0
```

The decimal branch is first in the alternation. Regex alternation is ordered, so
with `\d+` first, `"249.99"` would match `"249"` and silently drop the cents.

Stripping `$` and `,` before matching handles `"$1,249.99"` — without removing
the comma the match would stop at it and return `1.0`.

Returning `0.0` on no match rather than raising: `EnsembleAgent` filters
`value > 0`, so an unparseable reply drops that estimator from this item's
ensemble instead of failing the deal.

### `price()`

```python
response = self.client.chat.completions.create(
    model=self.model,
    messages=[{"role": "user", "content": PROMPT.format(...)}],
    seed=42,
)
```

`seed=42` asks the provider for reproducible sampling. Best-effort — providers do
not guarantee it — but it reduces run-to-run variance, which matters when the
output feeds a threshold decision.

**Measured:** this path works end to end against a 4,000-product Chroma index,
producing estimates like `$189.00`, `$1,414.00` and `$260.00` from real scraped
listings.

---

## Agent 3 — `agents/specialist.py`

```python
class SpecialistAgent(Agent):
    def __init__(self):
        import modal

        self.log(f"Connecting to Modal app {settings.modal_app}")
        pricer = modal.Cls.from_name(settings.modal_app, settings.modal_class)
        self.service = pricer()

    def price(self, description: str) -> float:
        estimate = float(self.service.price.remote(description))
        return estimate
```

Twenty-five lines, deliberately. All the machinery — 4-bit quantisation, the LoRA
adapter, GPU allocation — lives in `modal_app/pricer_service.py` and runs in
Modal's container. This side is a typed remote-procedure call.

**`import modal` inside `__init__`** so the package is only needed if this agent
is constructed.

**`modal.Cls.from_name(app, class)`** looks the deployed class up by name rather
than importing it, which is what decouples the client from the server: the Modal
app can be redeployed with a new model and this code does not change.

**`.remote(description)`** is the RPC. It blocks, and on a cold container it
blocks for a while — the GPU has to spin up and load a quantised 3B model.
`min_containers=0` in the service definition means you pay nothing when idle and
pay latency on the first call. For a system that runs every ten minutes, that is
the right trade.

### The failure mode this agent taught

Construction succeeds even when Modal is not authenticated —
`modal.Cls.from_name` is lazy. The failure only appears at `.remote()`:

```
[Ensemble] specialist failed: Token missing. Could not authenticate client.
[Ensemble] specialist failed: Token missing. Could not authenticate client.
[Ensemble] specialist failed: Token missing. Could not authenticate client.
```

Once per deal, in a measured run. The `try/except` in `EnsembleAgent.__init__`
did not catch it, because nothing threw at construction. That is what led to the
drop-on-first-failure behaviour described under `ensemble.py`.

---

## Agent 4 — `agents/neural.py`

The local estimator. Architecture identical to the price prediction project's
`PriceNet` — it has to be, because it loads those exact weights.

```python
path = weights_path or settings.artifacts / WEIGHTS
if not path.exists():
    raise FileNotFoundError(
        f"No trained weights at {path}. Train the residual net in the price "
        "prediction project and copy residual_net.pt into artifacts/."
    )
```

Raising in the constructor is exactly right here — `EnsembleAgent` catches it and
skips this estimator. The error message names the fix, which is what someone
reading a log needs.

```python
blob = torch.load(path, map_location=self.device)
self.model = PriceNet().to(self.device)
self.model.load_state_dict(blob["state"])
self.model.eval()
self.mean = float(blob["mean"])
self.std = float(blob["std"])
```

**`float(blob["mean"])` is a bug fix**, and it is the most instructive bug in the
project. The checkpoint stores the target scaler alongside the weights.
`map_location=self.device` moved those scalar tensors to MPS, while the
prediction path computed on CPU:

```
RuntimeError: Expected all tensors to be on the same device,
but found at least two devices, mps:0 and cpu!
```

It failed on every single deal in a measured run. It had never appeared in the
price prediction project's *training* path, because there the scaler was created
on CPU and stayed there. It only surfaced when a different project **loaded** the
saved weights — which is the argument for the cross-project integration being a
real test rather than a formality.

Storing the scaler as plain Python floats removes device semantics from
arithmetic that never needed them.

**`self.model.eval()`** in the constructor disables dropout permanently. This
agent never trains, so there is no reason to toggle it per call — and forgetting
to call it would mean the same item gets a different price every time.

```python
self.vectorizer = HashingVectorizer(n_features=FEATURES, stop_words="english", binary=True)
```

Reconstructed from constants, not loaded. This is the hashing trick paying off
across a project boundary: because there is no vocabulary to fit, the *entire*
artefact needed to run this model is one `.pt` file. With a `CountVectorizer` the
vocabulary would have to be serialised and version-matched too.

```python
with torch.no_grad():
    x = self._vectorise... (inline)
    scaled = float(self.model(x)[0].item())
    estimate = max(math.expm1(scaled * self.std + self.mean), 0.0)
```

The inverse chain, exactly reversing training: un-standardise, then `expm1` to
undo `log1p`, then floor at zero. `torch.no_grad()` skips the autograd graph.

**Measured:** `[Neural] Estimates $212.97` — working, and contributing a genuinely
different opinion to the ensemble.

---

## Agent 5 — `agents/ensemble.py`

### Optional construction

```python
self.estimators: dict[str, object] = {"frontier": FrontierAgent(collection)}
self.weights = {"frontier": settings.weight_frontier}

if use_specialist:
    try:
        from agents.specialist import SpecialistAgent
        self.estimators["specialist"] = SpecialistAgent()
        self.weights["specialist"] = settings.weight_specialist
    except Exception as error:
        self.log(f"Specialist unavailable, skipping it ({error})")
```

Frontier is **not** optional — it is the strongest estimator and the one with no
special infrastructure requirement. The other two are added only if they
construct successfully.

`from agents.specialist import SpecialistAgent` is inside the `try`, so a missing
`modal` package is caught exactly like a missing Modal account. Same for `torch`
and the Neural agent.

Broad `except Exception` is justified here by what the block *is*: an optional
capability probe. Any failure means "cannot use this estimator", and the system
has a correct behaviour for that.

Measured output with Modal unconfigured and weights present:

```
[Neural] Loaded weights on mps
[Ensemble] Ready with frontier, neural
```

### Weight renormalisation

```python
total = sum(self.weights[e.source] for e in estimates)
combined = sum(e.value * self.weights[e.source] for e in estimates) / total
```

Weights are normalised over **the estimators that actually returned a value on
this item**, not over the configured three.

Without this, losing the Specialist would silently shrink every estimate by 25%,
because the numerator would drop while the denominator stayed at 1.0. Every
estimate would be biased low, and every discount would look smaller — the system
would quietly stop alerting and there would be no error to find.

It renormalises **per item**, not per run, because an estimator can fail on one
description and succeed on the next.

### Dropping a failed estimator

```python
for source, estimator in list(self.estimators.items()):
    try:
        value = estimator.price(description)
    except Exception as error:
        self.log(f"{source} failed, dropping it for this run: {error}")
        del self.estimators[source]
        continue
```

This is the fix for the Modal authentication failure. The first version logged
and continued, which meant retrying a permanently broken estimator on every deal
— five identical stack traces per cycle, each with the latency of a failed
connection attempt.

**`list(self.estimators.items())`** takes a snapshot before iterating, because
the loop deletes from the dict it is iterating. Without it: `RuntimeError:
dictionary changed size during iteration`.

Dropping for the run rather than permanently is the right scope: a new
`EnsembleAgent` is built per framework start, so a transient failure gets another
chance on the next process, but a persistent one stops costing time immediately.

### The value filter

```python
if value > 0:
    estimates.append(Estimate(source=source, value=value))
```

An estimator returning 0 (unparseable LLM reply, a degenerate model output) is
excluded rather than dragging the weighted mean toward zero. A zero estimate
would compute a large negative discount and, more dangerously, a *low* estimate
combined with a low price could pass a threshold check for the wrong reason.

```python
if not estimates:
    return 0.0, []
```

If every estimator failed, return zero. The Planner's discount becomes
`0 - price`, i.e. negative, which cannot clear the threshold — so a total
estimator failure produces no alert rather than a wrong one.

---

## Agent 6 — `agents/messenger.py`

### The prompt

```
Write a two sentence push notification about this deal. Say what the item is,
what it costs and roughly how much below its estimated value it is. Be direct,
no emoji, no exclamation marks.
```

"Be direct, no emoji, no exclamation marks" fights the model's default
marketing-copy register. A notification that reads like an advert is one the user
learns to dismiss without reading, which defeats the system.

Three required facts — what, how much, how far below value — so the user can
decide from the notification alone without opening anything.

Measured output: *"Get the Amazon Smart Thermostat for $57, saving 65% on its
estimated value of $161.76."*

### Graceful degradation

```python
def push(self, text: str) -> bool:
    if not settings.pushover_configured:
        return False
    response = requests.post(PUSHOVER_URL, data={...}, timeout=15)
    return response.ok

def alert(self, opportunity: Opportunity) -> str:
    text = self.compose(opportunity)
    message = f"{text}\n{opportunity.deal.url}"
    delivered = self.push(message[:900]) if settings.notify else False
    self.log(f"{'Sent' if delivered else 'Composed'} alert: {text[:90]}")
    return message
```

`push` returns a bool rather than raising, and `alert` logs `Sent` or `Composed`
accordingly. So the entire alert path — including the LLM call that writes the
message — is exercisable without a Pushover account, and the log tells you
honestly which happened.

The constructor warns once at startup rather than on every alert.

**`message[:900]`** — Pushover's limit is 1,024 characters and rejects longer
messages. Truncating before sending is better than a failed delivery.

**`settings.notify`** is the `--no-notify` flag: compose but never send. Useful
for testing the full path without spamming a real device, which is how the
measured runs were done.

---

## Agent 7 — `agents/planner.py`

The policy layer. Everything here is a decision about *what to do*, not *how to
do it*.

### Batch deduplication

```python
candidates, batch_keys = [], set()
for deal in selection.deals:
    deal_keys = keys(deal)
    if self.memory.has_seen(deal) or deal_keys & batch_keys:
        continue
    batch_keys |= deal_keys
    candidates.append(deal)
```

Two checks: against memory (across runs), and against `batch_keys` (within this
run).

**The batch check was added after watching it fail.** In a measured run the
Scanner returned:

```
$65.00   A portable monitor with a 15.6-inch 1080p display, ideal for working on the go.
$65.00   A portable monitor that doubles your screen real estate, ideal for multitasking.
$1429.00 A gaming desktop featuring Intel Core i5-14400F and GeForce RTX 5060...
$1799.00 A gaming monitor in the form of a desktop PC with NVIDIA GeForce RTX 5060...
```

The same portable monitor twice, and arguably the same desktop twice, from
different feeds with different URLs. Memory could not catch them — neither had
been seen before. Without the batch check the system would price the same product
twice and could alert on it twice in one cycle.

### Ranking and the single alert

```python
opportunities = [self.value(deal) for deal in candidates[: settings.shortlist_size]]
opportunities.sort(key=lambda o: o.discount, reverse=True)
best = opportunities[0]
```

**Only the best one is alerted.** The value of this system is a small number of
alerts worth opening. Alerting on everything above the threshold makes it a feed,
and a feed gets ignored.

`candidates[:shortlist_size]` caps the work per cycle. Each valuation costs
between one and three model calls, so this bounds both latency and spend.

### Persist everything, alert on one

```python
for opportunity in opportunities:
    self.memory.add(opportunity)

if best.discount < settings.discount_threshold:
    self.log(f"Below the ${settings.discount_threshold:,.0f} threshold, no alert")
    return None

self.messenger.alert(best)
self.memory.mark_alerted(best)
```

Two distinct records: **seen** (every valued opportunity) and **alerted** (only
the one notified).

Persisting everything is what makes the next cycle's deduplication work. Storing
only alerted deals would mean re-pricing the same rejected deals forever.

The separation also makes `run.py --memory` genuinely useful — it shows what was
considered and rejected, including negative discounts:

```
alerted  $  104.76 off   paid $   57.00   worth $  161.76   Amazon Smart Thermostat
         $   53.73 off   paid $   49.99   worth $  103.72   SAMA Z60 White ATX PC Case
         $ -213.25 off   paid $1,429.00   worth $1,215.75   MSI Gaming Codex R2 Desktop
```

The negative rows are the system working — it priced them, found them overpriced,
and said nothing.

`mark_alerted` is called **after** `alert()` returns, so a failed alert does not
mark the deal as notified and it remains eligible next cycle.
