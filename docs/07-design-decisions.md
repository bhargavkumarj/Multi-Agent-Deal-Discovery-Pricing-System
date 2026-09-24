# Design Decisions

Every non-obvious choice, the alternatives rejected, and the bugs found by
running the system against live feeds.

---

## Decision 1 — Seven agents, not one function

**Choice.** Scanner, Frontier, Specialist, Neural, Ensemble, Messenger, Planner.

**Why the boundaries are where they are.** Each agent is a different *kind* of
work with a different failure mode: extraction from noisy text, retrieval plus
reasoning, remote GPU inference, local numerical inference, aggregation,
generation plus delivery, policy.

**What it buys.** Independent failure — the Specialist needs Modal, the Neural
agent needs weights, neither is required. Independent substitution — adding a
fourth estimator is one class and three lines.

**When it would be over-engineering.** If all three estimators were the same
model with different prompts. They are not: an API call with retrieval, a remote
fine-tuned model, and a local network, with different dependencies and costs.

---

## Decision 2 — No abstract `run()` on the base class

**Choice.** `Agent` provides `name`, `colour` and `log`, and nothing else.

**Rejected.** A shared `execute()` or `act()` method.

**Why rejected.** `ScannerAgent.scan`, `FrontierAgent.price` and
`MessagingAgent.alert` have nothing in common. Forcing one name would give seven
methods doing unrelated things behind `**kwargs` — an abstraction that costs
clarity and buys nothing, because no code ever iterates over a heterogeneous list
of agents.

**The honest gap.** The three estimators *do* share
`price(description) -> float`, and `EnsembleAgent` depends on it via duck typing.
That contract is real and implicit. A `Pricer` protocol would document it, and
that is a fair criticism to accept rather than argue with.

---

## Decision 3 — The Planner's sequence is hard-coded

**Choice.** Scan, dedup, value, rank, alert — in Python, not decided by an LLM.

**Rejected.** An autonomous agent loop where an LLM picks the next tool.

**Why rejected.** The workflow is known, fixed and correct. There are no
branching decisions for a model to make. Letting an LLM choose the order would
add latency, cost and nondeterminism with no upside.

**When the other choice is right.** When the sequence genuinely varies by input —
different deal types needing different research paths, or a user asking
open-ended questions. Neither is true here.

---

## Decision 4 — `Field` descriptions carry the hardest instruction

```python
price: float = Field(
    description="The price being charged for the item. If the listing says '$100 off the "
    "usual $300', the price is 200."
)
```

**Why it is there and not only in the system prompt.** The `description` is
included in the JSON schema sent to the model, attached to the field it governs,
so the model sees it at the moment it generates that field.

**Why this particular instruction.** It is the highest-consequence extraction in
the system. Deal feeds constantly say "$200 off". A model returning the *saving*
instead of the *price* produces a wrong discount, which produces a confident,
useless alert — the exact failure that destroys trust in a notification system.

It is stated in the system prompt *as well*. Belt and braces on the one thing
that must not go wrong.

---

## Decision 5 — Validate what the schema cannot

```python
selection.deals = [deal for deal in selection.deals if deal.price > 0]
```

**Why.** Structured outputs guarantee the *shape*, not the *semantics*. A price
of 0.0 is schema-valid and meaningless — and it would compute a discount equal to
the entire estimate and rank **first**, making it the deal the system alerts on.

**The principle: schema constrains form, code constrains meaning.** Prompts are
probabilistic; a filter is not.

---

## Decision 6 — Filter by URL before the LLM call, by content after

**Choice.** `ScannerAgent.fresh_listings` drops seen URLs before prompting;
`PlanningAgent.run` drops content duplicates after.

**Why split.** The URL check is exact, free and possible before the LLM call — so
it saves tokens and stops the model wasting shortlist slots on old deals.
Measured: `9 listings, 5 not seen before` on a second cycle.

The content check *needs* the structured description, which only exists after the
LLM call. It cannot happen earlier.

---

## Decision 7 — Two dedup keys, matched on either

**Rejected.** URL only.

**Why rejected.** The same product appears across feeds with different tracking
parameters, and the Scanner sometimes lists it twice in one batch with different
wording. A URL check catches neither.

**Chosen.** A normalised URL key *and* a fuzzy content key built from sorted
significant words plus the price bucketed to $10. A match on either is enough.

**Why bucketed price.** Prices drift between listings of the same item; exact
matching would let $64.99 versus $65.00 defeat the check. The bucket also
disambiguates two genuinely different products with similar descriptions but very
different prices.

**Why sorted set of words.** Order-independence is what makes the key survive
rewording — which is exactly the case it was built for.

---

## Decision 8 — Within-batch deduplication

**The observation that caused it.** In a measured run the Scanner returned:

```
$65.00   A portable monitor with a 15.6-inch 1080p display, ideal for working on the go.
$65.00   A portable monitor that doubles your screen real estate, ideal for multitasking.
```

The same product, twice, in one shortlist, from two feeds with different URLs.
Memory could not catch it — neither had been seen before.

**The fix.** Track `batch_keys` within the loop and skip anything colliding with
a deal already accepted this pass.

**Why it matters.** Without it the system prices the same product twice — wasted
calls — and could alert on it twice in a single cycle.

---

## Decision 9 — Renormalise ensemble weights per item

```python
total = sum(self.weights[e.source] for e in estimates)
combined = sum(e.value * self.weights[e.source] for e in estimates) / total
```

**The bug this prevents.** Without renormalisation, losing the Specialist would
shrink every estimate by 25% — the numerator drops while the denominator stays at
1.0.

Every estimate biased low, every discount smaller, the system quietly stops
alerting. **No error anywhere.** That is the most dangerous class of bug: silently
wrong, plausibly shaped output.

**Per item, not per run,** because an estimator can fail on one description and
succeed on the next.

---

## Decision 10 — Drop a failing estimator instead of retrying it

**The bug.** A measured run with Modal unconfigured produced:

```
[Ensemble] specialist failed: Token missing. Could not authenticate client.
[Ensemble] specialist failed: Token missing. Could not authenticate client.
[Ensemble] specialist failed: Token missing. Could not authenticate client.
[Ensemble] specialist failed: Token missing. Could not authenticate client.
[Ensemble] specialist failed: Token missing. Could not authenticate client.
```

Once per deal. The `try/except` in `__init__` had not caught it because
`modal.Cls.from_name` is **lazy** — construction succeeded, and the failure only
appeared at `.remote()`.

**The fix.**

```python
for source, estimator in list(self.estimators.items()):
    try:
        value = estimator.price(description)
    except Exception as error:
        self.log(f"{source} failed, dropping it for this run: {error}")
        del self.estimators[source]
        continue
```

**The `list(...)` matters** — the loop deletes from the dict it iterates, and
without a snapshot that is `RuntimeError: dictionary changed size during
iteration`.

**The lesson.** A lazily-connecting client cannot be health-checked at
construction. Either probe it explicitly, or handle failure at the call site.
This takes the second route because a probe would mean spinning up a GPU just to
ask whether the GPU is reachable.

---

## Decision 11 — The neural scaler as plain floats

**The bug.** Every deal in a measured run produced:

```
[Ensemble] neural failed: Expected all tensors to be on the same device,
but found at least two devices, mps:0 and cpu!
```

**The cause.** The checkpoint stores the target scaler (`mean`, `std`) alongside
the weights. `torch.load(path, map_location=self.device)` moved those scalar
tensors to MPS, while the prediction path computed on CPU.

**Why it was invisible until integration.** In the price prediction project's
*training* path, the scaler was created on CPU and stayed there. The bug only
existed on the **load** path, which only a different project exercised.

**The fix.** `float(blob["mean"])` — store the scaler as plain Python numbers,
removing device semantics from arithmetic that never needed them.

**Two lessons.** A value that is not a model parameter should not be a tensor.
And the cross-project integration is a real test, not a formality — it exercised
a path neither project's own tests would have reached.

---

## Decision 12 — Empty `__init__.py` files

**The bug.** The first version re-exported from both package `__init__.py` files
and produced:

```
ImportError: cannot import name 'Memory' from partially initialized module
core.memory (most likely due to a circular import)
```

The cycle: `core/__init__` → `core.memory` → `agents.deals` → `agents/__init__`
→ `agents.planner` → `core.memory`, which was still initialising.

**The fix.** Empty both `__init__.py` files. Modules import each other directly
by path.

**Why not lazy re-exports or `TYPE_CHECKING` guards.** Those work but add
machinery to solve a problem created by convenience. The convenience — writing
`from agents import PlanningAgent` instead of
`from agents.planner import PlanningAgent` — was not worth it.

**The general point.** Package `__init__.py` re-exports create import-order
dependencies that are invisible until they bite, and they bite at the least
convenient time.

---

## Decision 13 — Persist every opportunity, alert on one

```python
for opportunity in opportunities:
    self.memory.add(opportunity)

if best.discount < settings.discount_threshold:
    return None

self.messenger.alert(best)
self.memory.mark_alerted(best)
```

**Why persist everything.** That is what makes the next cycle's deduplication
work. Storing only alerted deals would mean re-pricing the same rejected deals
forever — paying for the same LLM calls every ten minutes.

**Why alert on one.** The value of the system is a small number of alerts worth
opening. Alerting on everything above the threshold makes it a feed, and a feed
gets ignored. Alert fatigue is the failure mode that kills notification systems,
and it is a product decision, not a technical one.

**Why `mark_alerted` comes after `alert()`.** A failed alert does not mark the
deal as notified, so it stays eligible next cycle.

---

## Decision 14 — An absolute dollar threshold

**Choice.** `DISCOUNT_THRESHOLD = 50`, in dollars.

**Rejected.** A percentage.

**Why.** $50 off a $60 item and $50 off a $600 item are both worth knowing. 30%
off a $10 item is not worth a notification. The absolute amount tracks "is this
worth my attention?" better than the ratio does, for a personal alerting tool.

A commercial repricing system would likely want both, or a percentage floor with
an absolute minimum.

---

## Decision 15 — Record the spread but do not act on it

```python
@property
def spread(self) -> float:
    values = [e.value for e in self.estimates]
    return max(values) - min(values) if len(values) > 1 else 0.0
```

**Choice.** Compute and persist estimator disagreement; do not use it in the
threshold decision.

**Why.** It is a genuine confidence signal — three estimators within $20 is a
different claim to three spanning $400 — and it is nearly free to collect.

But *how* to act on it is a judgement call that should be made from data: widen
the threshold when spread is high? Attach a confidence band to the alert? Suppress
entirely? Guessing now would bake in an arbitrary rule. Collecting the signal
means the decision can be made properly later.

---

## Decision 16 — Timeouts on every HTTP call

```python
requests.get(url, headers=HEADERS, timeout=15)
requests.post(PUSHOVER_URL, data={...}, timeout=15)
```

**Why it is not optional.** `requests` waits **indefinitely** by default. In a
`--loop` deployment, one unresponsive deal page stops the system permanently,
with no error and no output.

That is the worst possible failure mode, because it looks exactly like nothing
being wrong. A crash gets noticed; a hang does not.

---

## Decision 17 — `openai_client()` honouring `OPENAI_BASE_URL`

```python
def openai_client():
    from openai import OpenAI
    if settings.api_base:
        return OpenAI(base_url=settings.api_base, api_key=os.getenv("OPENAI_API_KEY", "local"))
    return OpenAI()
```

**What it buys.** Pointing `OPENAI_BASE_URL` at
`http://localhost:11434/v1` runs the entire system against Ollama — including
`chat.completions.parse` with structured outputs, which Ollama supports.

This is how every measured end-to-end run was done, and it is also a genuine
deployment option: local models, vLLM, or a corporate gateway all speak the same
protocol.

**The secondary benefit** is that it made the project testable when the available
API key was returning 401. Nine lines that turned "cannot verify this works" into
a full end-to-end run.

---

## Decision 18 — Degrade for optional capability, raise for required state

```python
# Optional — degrade
except Exception as error:
    self.log(f"Specialist unavailable, skipping it ({error})")

# Required — raise
raise FileNotFoundError(
    f"No product vector store at {settings.vector_store}. "
    "Run `python build_index.py` first."
)
```

**The test.** Is there a correct behaviour without this thing?

Without the Specialist, yes — the ensemble runs on the remaining estimators.

Without the vector store, no. `get_or_create_collection` would return an empty
collection and the Frontier agent would degrade to zero-shot guessing, which the
measured results show is worse than useless. The system would *appear* to work
while every estimate was a hallucination.

The error message names the exact command to run, which is what someone reading a
log needs.

---

## Decision 19 — Lazy planner construction

```python
def start(self) -> PlanningAgent:
    if self.planner is None:
        self.planner = PlanningAgent(self.collection, self.memory, **self.ensemble_options)
    return self.planner
```

**Why.** Constructing the planner constructs all seven agents — loading a
sentence-transformer, possibly a 13.5M-parameter network, possibly a Modal
connection. Several seconds.

`run.py --memory` needs none of that. It reads one JSON file and prints, which is
what you want from the command you run to check on a system.

Caching means a `--loop` run pays the cost once, not per cycle.

---

## Decision 20 — A custom logging handler for the UI

```python
class LogStream(logging.Handler):
    def emit(self, record):
        self.queue.put(self.format(record))
```

**Why not have agents write to the UI directly.** Because then agents would know
about Gradio, and running headless would need a null UI.

Agents call `logging.info`. The dashboard attaches a handler. Adding an eighth
agent requires no UI change, and the same log stream serves the terminal and the
browser — `logs.to_html` translates the ANSI colours at the display boundary.

---

## Bugs found by running the system

| Bug | Symptom | Root cause | Fix |
|---|---|---|---|
| Circular import | `cannot import name 'Memory' from partially initialized module` | Package `__init__.py` re-exports created an import cycle | Empty both `__init__.py` files |
| Neural device mismatch | `Expected all tensors on the same device, mps:0 and cpu` on every deal | `map_location` moved the saved scaler to MPS while prediction ran on CPU | Store the scaler as plain floats |
| Estimator retry storm | Five identical Modal auth failures per cycle | Lazy client — construction succeeded, only `.remote()` failed | Drop a failing estimator for the rest of the run |
| Scanner duplicates | Same portable monitor shortlisted twice in one pass | Memory only catches across-run duplicates | Within-batch key check in the Planner |

All four were found by running the system against live feeds. None would have
been caught by reading the code.

The device mismatch is the most instructive: it did not exist in the project that
*created* the artefact, only in the project that *loaded* it. That is the
argument for treating the cross-project integration as a real test rather than a
demonstration.
