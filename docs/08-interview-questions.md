# Interview Questions — Multi-Agent Deal Discovery & Pricing

30 questions with full answers, grouped by what they test. The strongest answers
cite what was actually observed in the measured runs, so those are included.

---

## Part 1 — Multi-agent design (1–8)

### Q1. Walk me through the system.

Seven agents watch retail deal feeds and alert when something is priced well
below what it is worth.

The **Scanner** parses three RSS feeds, scrapes each deal page, filters out URLs
already in memory, and makes one structured-output call to shortlist the deals
with a clear, unambiguous price.

Three **estimators** then price each shortlisted item independently. The
**Frontier** agent embeds the description, retrieves five comparable products
with their real prices from a Chroma catalogue, and asks an LLM to estimate with
those in context. The **Specialist** calls a fine-tuned Llama-3.2-3B running on a
Modal GPU. The **Neural** agent runs a local residual network over hashed text
features.

The **Ensemble** combines them into a weighted mean, renormalising over whichever
estimators actually returned a value, and records the spread between them as a
confidence signal.

The **Planner** deduplicates, ranks by discount, writes everything to memory, and
if the best candidate clears the threshold, hands it to the **Messenger**, which
writes a two-sentence notification and delivers it via Pushover.

The memory is what makes it runnable continuously — it ensures the same product
is never alerted on twice.

### Q2. Why seven agents instead of one function?

The decomposition follows the kind of work, because different kinds of work fail
differently. Extraction from noisy text, retrieval plus reasoning, remote GPU
inference, local numerical inference, aggregation, generation and delivery, and
policy are seven genuinely different problems.

Two things fall out that I actually rely on.

**Independent failure.** The Specialist needs Modal, the Neural agent needs
trained weights. Neither is required — the Ensemble renormalises its weights over
whichever estimators loaded. In a measured run with Modal unconfigured, it logged
`Ready with frontier, neural` and carried on.

**Independent substitution.** Adding a fourth estimator is one class with a
`price()` method plus three lines in the Ensemble's constructor.

I would also say when this *would* be over-engineering: if all three estimators
were the same model with different prompts, it would be ceremony. It is justified
here because they are genuinely different systems with different dependencies,
costs and failure modes.

### Q3. Your base Agent class has no abstract `run()`. Why not?

Because the seven agents genuinely do different things.
`ScannerAgent.scan(seen_urls)`, `FrontierAgent.price(description)` and
`MessagingAgent.alert(opportunity)` have nothing in common except that they log.

Forcing a shared `execute()` would give me seven methods with the same name doing
unrelated things behind `**kwargs`. That costs clarity and buys nothing, because
no code ever iterates over a heterogeneous list of agents — the Planner calls each
by its specific method.

What the base class does buy is a readable transcript. With seven agents
interleaving output, every line is prefixed and coloured by origin.

I would volunteer the gap: the three estimators *do* share
`price(description) -> float`, and the Ensemble depends on that via duck typing.
That contract is real and currently implicit. A `Pricer` protocol would document
it properly, and that is a fair criticism.

### Q4. Is this really "agentic"? The Planner's sequence is hard-coded.

It is a multi-agent system, not an autonomous agent loop, and I would be precise
about the difference rather than blur it.

The Planner's sequence — scan, dedup, value, rank, alert — is Python, not
decided by an LLM. That is deliberate. The workflow is known, fixed and correct;
there are no branching decisions for a model to make. Letting an LLM choose the
order would add latency, cost and nondeterminism with no upside.

LLM-driven planning earns its place when the sequence genuinely varies by input —
different deal types needing different research, or a user asking open-ended
questions. Neither is true here.

What makes the agents agents is that each has its own model or tool, its own
prompt or parameters, and its own failure mode, and each can be swapped without
touching the others.

### Q5. How does the system behave when things are missing?

By a principle I would state explicitly: **missing optional capability degrades,
missing required state raises.**

Optional and degrading: no Modal means the Specialist is skipped and the ensemble
weights renormalise. No trained weights means the Neural agent is skipped. No
Pushover credentials means alerts are composed and logged rather than sent. An
estimator that throws mid-run is dropped for the rest of that run. A deal page
that will not load is skipped.

Required and raising: no vector store raises `FileNotFoundError` naming the
command to run.

The test is whether there is a correct behaviour without the thing. Without the
Specialist, yes. Without the vector store, no — `get_or_create_collection` would
return an empty collection and the Frontier agent would silently degrade to
zero-shot guessing, which my measurements show is worse than useless. The system
would *appear* to work while every estimate was a hallucination.

### Q6. How do the agents communicate?

Through typed pydantic objects, not dicts or strings.

`ScrapedDeal` is the raw listing from a feed. `Deal` is the structured output
from the Scanner — description, price, url. `Estimate` is one estimator's opinion
with its source. `Opportunity` is a valued deal, carrying the deal, the combined
estimate, the discount, the individual estimates and a timestamp.

The value of typing them shows up in two places. The `Deal` model doubles as the
Scanner's response schema, so the type that flows between agents is the same type
the LLM is constrained to produce. And `Opportunity` serialises straight to JSON
for memory with `model_dump()`, with no custom encoder.

Keeping the individual estimates on the `Opportunity` rather than just the
combined number is what makes the `spread` property possible.

### Q7. How would you add a fourth estimator?

One class with a `price(description) -> float` method, and three lines in
`EnsembleAgent.__init__`:

```python
if use_new:
    try:
        from agents.new import NewAgent
        self.estimators["new"] = NewAgent()
        self.weights["new"] = settings.weight_new
    except Exception as error:
        self.log(f"New estimator unavailable, skipping it ({error})")
```

Nothing else changes. The weights renormalise automatically over whatever is
present, the spread calculation picks it up, and it is persisted in the
`Opportunity` like the others.

The import inside the `try` is deliberate — it means a missing package is handled
identically to a missing credential or a missing weights file.

### Q8. What would you change to make this handle 100 feeds instead of 3?

Several things, in order.

**Scraping goes concurrent.** Right now `ScrapedDeal.fetch` is a sequential loop
with a `time.sleep(0.05)` between pages. At 100 feeds that is the bottleneck. It
is pure I/O wait, so a thread pool fixes it — with a per-domain rate limit, since
politeness matters more at that volume.

**The Scanner's prompt would not fit.** 100 feeds × 10 entries × ~1,100
characters is far past a sensible prompt. I would shortlist per feed or per
batch, then run a second pass over the winners.

**Memory needs a real store.** The current design rewrites the whole JSON file on
every add and computes `seen` as an O(n) property. Fine at hundreds; wrong at
hundreds of thousands. SQLite with an index on both key columns would be the
minimal change.

**Deduplication needs to scale.** Exact key matching in a set is fine, but at
volume I would want near-duplicate detection — MinHash or SimHash over the
description — rather than my sorted-first-ten-words heuristic.

**The estimators become the cost.** At 100 feeds, `shortlist_size` is the lever
that controls spend. I would likely add a cheap pre-filter — the Neural agent
costs nothing, so use it to triage and only pay for Frontier and Specialist on
promising candidates.

---

## Part 2 — Estimation and ensembling (9–16)

### Q9. Why does the Frontier agent use retrieval?

Because without it, an LLM asked to price a product guesses.

I measured this in my price prediction project: a zero-shot LLM returned $4,500
for a $53 projector and $1,500 for an $80 jigsaw puzzle. Average error $264 —
worse than predicting the training mean. Look at the numbers: $4,500, $2,500,
$1,500. Round figures. That is the signature of guessing, not estimating.

The model has never seen this product's current market price. Retrieval changes
the question from "what does this cost?", which requires knowledge it lacks, to
"here are five similar products and what they sold for — what does this one
cost?", which is interpolation from the context.

The critical implementation detail is that the retrieved metadata carries the
**real price**. Retrieving five similar descriptions would be useless. Retrieving
five similar descriptions *and their prices* is what provides the anchor.

### Q10. What has to be true for that retrieval to work?

Three things.

**The same embedding model on both sides.** The deal description is embedded with
`all-MiniLM-L6-v2`, the same model that embedded the catalogue. Vectors from
different models are not comparable — and Chroma would return nearest neighbours
in a meaningless sense with no error, because the arithmetic is valid. That is a
silent failure, which makes it the dangerous kind.

**The catalogue must contain comparable items.** This is why my feeds and my
index are a matched pair — Electronics, Computers and Smart Home on both sides.
Pointing the scanner at a Garden feed while the index holds Electronics would
retrieve the least-distant of a bad set, and the anchor would actively mislead.

**The description must describe the product, not the deal.** That is why the
Scanner's `Field` description says "describe what the item is and what it does,
not the terms of the discount" — the description is what gets embedded.

### Q11. Why ensemble three models instead of using the best one?

Because they fail in different places, which is the only condition under which
ensembling helps.

The Frontier agent is strongest but drifts on unusual items where retrieval finds
poor comparables. The Specialist has price knowledge in its weights and is steady
on categories it was fine-tuned on. The Neural agent is weakest alone — around
$71 average error — but it does pure statistical pattern matching on word
presence with no semantic understanding at all, so it is wrong in a completely
different way to the LLM.

That last point is the key one. These are not three variations of one approach.
If they were, averaging would change nothing.

Measured on a real deal:

```
[Frontier] Estimates $189.00
[Neural]   Estimates $212.97
[Ensemble] frontier $189, neural $213 -> $193.79
```

I would also be honest that my weights are hand-set priors informed by measured
per-model performance, not learned. With ground-truth prices for scraped deals I
could fit them by regression.

### Q12. Explain the weight renormalisation and why it matters.

```python
total = sum(self.weights[e.source] for e in estimates)
combined = sum(e.value * self.weights[e.source] for e in estimates) / total
```

Weights are normalised over the estimators that actually returned a value **on
this item**, not over the configured three.

Without it, losing the Specialist means the numerator drops its 0.25 contribution
while the denominator stays at 1.0. Every estimate comes out 25% low. Every
discount looks smaller. The system quietly stops alerting.

And there is no error anywhere. That is the most dangerous class of bug — silently
wrong output with a plausible shape. Nobody investigates a system that just
stopped finding deals; they assume there were no deals.

It renormalises per item rather than per run because an estimator can fail on one
description and succeed on the next.

### Q13. What is the spread and why do you record it but not use it?

Spread is the range between the highest and lowest estimator on a single item. A
$500 estimate that all three agree on is a very different claim to a $500 estimate
they span $400 on.

It is a genuine confidence signal and nearly free to compute, so I record it on
every opportunity and persist it. In a measured run: `Best candidate is $104.76
under estimate (estimator spread $64)`.

I do not act on it yet, deliberately. *How* to act on it is a judgement call that
should come from data — widen the threshold when spread is high? Attach a
confidence band to the alert? Suppress entirely? Guessing now would bake in an
arbitrary rule.

Collecting the signal first means the decision can be made properly later. That
is the same reasoning as instrumenting before optimising.

### Q14. Why is the Specialist agent only 25 lines?

Because all the machinery belongs on the other side of the boundary.

The Modal container holds the 4-bit quantisation config, the base model load, the
LoRA adapter, and the generation call. The agent does
`modal.Cls.from_name(app, class)` and `.remote(description)`.

That split means the local machine needs no torch, no transformers, no peft, and
no GPU. It also means the Modal app can be redeployed with a different model and
this code does not change — the lookup is by name, not by import.

It is a typed RPC, and keeping it thin is the point.

### Q15. Walk me through the Modal service.

It is infrastructure as code. The container image, the HuggingFace secret, the
GPU type and a persistent volume are all declared in Python, and `modal deploy`
provisions them.

The `Pricer` class has a `@modal.enter()` method that runs **once per container**,
not per request. It loads Llama-3.2-3B with 4-bit NF4 quantisation — which takes
it from about 6GB to under 2GB so it fits on a cheap T4 — and then applies a LoRA
adapter with `PeftModel.from_pretrained`, pinned to a commit hash so the model
cannot change under a running service.

The persistent volume mounted at the HF cache path is what makes cold starts
survivable. Without it, every cold start would download gigabytes.

`min_containers=0` means zero idle cost and a cold start on the first call after
a quiet period. For a system on a ten-minute timer that trade is correct. For an
interactive product it would not be, and I would pay for a warm container.

The `price` method builds a prompt ending at `"Price is $"` — mid-sentence,
matching the fine-tuning format — and generates at most 5 tokens, because the
answer is a number.

### Q16. The Neural agent loads weights from a different project. How does that work, and is it wise?

It loads `residual_net.pt`, which my price prediction project's `run.py train`
produces. The architecture classes are duplicated so both sides agree on the
shape.

It works cleanly because of the hashing trick. `HashingVectorizer` has no
vocabulary to fit, so it is reconstructed from constants — which means the
*entire* artefact needed to run the model is one `.pt` file containing the
weights and the target scaler. With a `CountVectorizer` I would have to serialise
and version-match a vocabulary too.

Is it wise? It is the right architecture with the wrong packaging. Duplicating
the model classes is real duplication, and if the architecture changed in one
place the checkpoint would load into the wrong shape. The correct fix is a shared
package both projects depend on, with the architecture version recorded in the
checkpoint.

I would also point out this integration found a real bug that neither project's
own usage would have — which is the argument for it being a genuine test rather
than a convenience.

---

## Part 3 — Continuous operation (17–23)

### Q17. How do you make sure the same deal is not alerted twice?

Two keys per deal, matched on either.

`url_key` strips the query string, trailing slash and case. That handles tracking
parameters — the same product page served from two feeds has different `?iref=`
values.

`content_key` is a fuzzy fingerprint: lowercase alphanumeric tokens, stopwords
and short words removed, then a **sorted set** of the first ten significant
words, plus the price bucketed to the nearest $10.

Every design choice there is doing something. The set means repeated words count
once. The sorting means word *order* does not matter, so two rewordings of the
same product produce the same key. The price bucket means $64.99 versus $65.00
does not defeat the check, and it disambiguates two different products with
similar descriptions but very different prices.

`has_seen` is a set intersection, so a match on either key is enough — the same
URL is definitely the same deal, and the same content key is very probably the
same product.

### Q18. Why two dedup checks in different places?

Because they catch different things and can happen at different times.

The Scanner filters by URL **before** the LLM call. That is exact, free, and it
saves tokens while stopping the model wasting shortlist slots on old deals.
Measured on a second cycle: `9 listings, 5 not seen before`.

The Planner filters by content key **after** the LLM call, because the content
key needs the structured description, which does not exist until the model has
produced it.

The Planner also checks within the batch, which memory structurally cannot do.

### Q19. Tell me about the within-batch deduplication.

That was added after watching it fail. In a measured run the Scanner returned:

```
$65.00   A portable monitor with a 15.6-inch 1080p display, ideal for working on the go.
$65.00   A portable monitor that doubles your screen real estate, ideal for multitasking.
```

The same product, twice, in one shortlist, from two feeds with different URLs.
Memory could not catch it — neither had been seen before, because it was the same
pass.

So the Planner tracks `batch_keys` as it iterates and skips anything colliding
with a deal already accepted this cycle. Without it, the system would price the
same product twice — wasted calls — and could alert on it twice in one cycle,
which is exactly the behaviour the whole memory system exists to prevent.

It is a good example of why running a system against real data matters. I would
not have predicted that the shortlisting model would duplicate within its own
output.

### Q20. Why does memory store deals you did not alert on?

Two distinct records, answering different questions.

**Seen** means "do not price this again". It covers every valued opportunity,
including ones rejected for being overpriced. Without it, the system would
re-price the same rejected deals every ten minutes forever — paying for the same
LLM calls repeatedly.

**Alerted** means "do not notify about this again", and it is a much smaller set.

Keeping them separate also makes `run.py --memory` genuinely useful. It shows
what was considered and declined:

```
alerted  $  104.76 off   paid $   57.00   worth $  161.76   Amazon Smart Thermostat
         $   53.73 off   paid $   49.99   worth $  103.72   SAMA Z60 ATX PC Case
         $ -213.25 off   paid $1,429.00   worth $1,215.75   MSI Gaming Desktop
```

The negative rows are the system working — it priced them, found them overpriced,
and stayed quiet.

### Q21. Why only one alert per cycle?

It is a product decision, not a technical one. The value of this system is a
small number of alerts worth opening. Alerting on everything above the threshold
makes it a feed, and a feed gets ignored.

Alert fatigue is the failure mode that kills notification systems. One well-chosen
alert every ten minutes at most, ranked by discount, above a threshold.

Related: the threshold is an absolute dollar amount, not a percentage. $50 off a
$60 item and $50 off a $600 item are both worth knowing; 30% off a $10 item is
not worth a notification.

### Q22. What are the failure modes of running this continuously, and how did you handle them?

I would go through them concretely.

**Duplicate alerts** — two-key dedup plus within-batch checking, with memory
persisted on every write so a crash loses at most the current deal.

**Network hangs** — every `requests` call has `timeout=15`. Without a timeout,
`requests` waits indefinitely, and one unresponsive deal page would stop the loop
permanently with no error and no output. That is the worst failure mode because it
looks exactly like nothing being wrong.

**A broken estimator** — dropped for the rest of the run after its first failure,
rather than retried on every deal.

**Cost** — `shortlist_size` bounds valuations per cycle, `deals_per_feed` bounds
scraping.

**Observability** — every agent logs under its own name and colour, and the
dashboard renders the same stream via a custom logging handler.

**Unbounded memory growth** — this one I have *not* handled. The JSON file grows
without limit and `seen` is computed as an O(n) property on every check. It is
fine at hundreds of records and wrong at hundreds of thousands. SQLite with
indexed key columns is the fix.

### Q23. How does the dashboard stream the agent log live?

A custom `logging.Handler` whose `emit` pushes formatted records onto a
`queue.Queue`.

The cycle runs on a background thread while a Gradio generator drains the queue
and yields updates. Without the thread the UI would freeze for a minute with no
output until the cycle finished.

Two details that matter. The drain loop condition is `worker.is_alive() or not
queue.empty()` — continuing after the worker finishes, because the last log lines
are still queued and checking only `is_alive()` would truncate the transcript at
the interesting moment. And `queue.get(timeout=0.2)` rather than `get_nowait()`
in a tight loop, so it blocks briefly instead of spinning the CPU.

The design point is that the agents know nothing about Gradio. They call
`logging.info` as always; the dashboard attaches a handler. Adding an eighth agent
needs no UI change. `logs.to_html` translates the ANSI colour codes into spans at
the display boundary, so the browser shows the same colour-coded transcript the
terminal does.

---

## Part 4 — Engineering and judgement (24–30)

### Q24. Tell me about a bug you found in this system.

The most instructive one came from the cross-project integration.

The Neural agent loads weights trained in my price prediction project. Every
single deal produced:

```
[Ensemble] neural failed: Expected all tensors to be on the same device,
but found at least two devices, mps:0 and cpu!
```

The checkpoint stores the target scaler — mean and standard deviation of the log
prices — alongside the weights. `torch.load(path, map_location=self.device)` moved
those scalar tensors to MPS, while the prediction path computed on CPU.

What makes it interesting is that the bug did not exist in the project that
*created* the artefact. In the training path the scaler was created on CPU and
stayed there, so it worked perfectly. The bug lived only on the load path, which
only a different project exercised.

The fix was to store the scaler as plain Python floats, removing device semantics
from arithmetic that never needed them.

Two lessons. A value that is not a model parameter should not be a tensor. And
the cross-project integration is a real test, not a formality — it reached a path
neither project's own usage would have.

### Q25. Tell me about a bug that came from how you structured the code.

A circular import, and it killed the whole thing on startup.

I had `core/__init__.py` re-export `Memory` and `agents/__init__.py` re-export
every agent. Running `build_index.py` produced:

```
ImportError: cannot import name 'Memory' from partially initialized module
core.memory (most likely due to a circular import)
```

The cycle was: `core/__init__` imports `core.memory`, which imports
`agents.deals`, which triggers `agents/__init__`, which imports `agents.planner`,
which imports `core.memory` — still initialising.

I fixed it by emptying both `__init__.py` files. Modules import each other
directly by path.

I could have used lazy re-exports or `TYPE_CHECKING` guards, but those add
machinery to solve a problem created purely by convenience — writing
`from agents import PlanningAgent` instead of
`from agents.planner import PlanningAgent`. That convenience was not worth it.

The general point: package `__init__.py` re-exports create import-order
dependencies that are invisible until they bite.

### Q26. You had an estimator failing on every deal. What happened?

Modal was not authenticated, and the log showed five identical failures per
cycle:

```
[Ensemble] specialist failed: Token missing. Could not authenticate client.
[Ensemble] specialist failed: Token missing. Could not authenticate client.
...
```

I had a `try/except` around constructing the Specialist agent that was supposed
to catch exactly this. It did not fire, because `modal.Cls.from_name` is **lazy**
— construction succeeds without contacting Modal, and the failure only appears at
`.remote()`.

So I was paying the latency of a failed connection attempt once per deal, forever.

The fix was to drop a failing estimator from the rotation for the rest of the
run:

```python
for source, estimator in list(self.estimators.items()):
    try:
        value = estimator.price(description)
    except Exception as error:
        self.log(f"{source} failed, dropping it for this run: {error}")
        del self.estimators[source]
```

The `list(...)` is necessary — the loop deletes from the dict it iterates, and
without a snapshot that is a `RuntimeError`.

Dropping for the run rather than permanently is the right scope: a new Ensemble
is built per process start, so a transient failure gets another chance while a
persistent one stops costing time immediately.

The broader lesson is that a lazily-connecting client cannot be health-checked at
construction. You either probe it explicitly or handle failure at the call site.
I chose the call site, because a probe would mean spinning up a GPU just to ask
whether the GPU is reachable.

### Q27. How did you test this without paid API access and without Modal?

Two mechanisms, and they are also genuine deployment options rather than test
scaffolding.

`openai_client()` honours `OPENAI_BASE_URL`. Pointing it at
`http://localhost:11434/v1` runs the whole system against Ollama, which speaks
the OpenAI protocol — including `chat.completions.parse` with structured outputs.
That is how the full end-to-end runs were done: live RSS scraping, structured
shortlisting, RAG pricing against a 4,000-product index, the neural estimator, a
two-estimator ensemble with a real spread, threshold logic, alert composition,
memory persistence, and dedup verified across two consecutive cycles.

The `--no-notify` flag composes alerts without delivering them, so the entire
messaging path including the LLM call is exercised without a Pushover account.

The Specialist genuinely could not be run — I have no Modal deployment. I say so
plainly in the README rather than claiming it works. What I *did* verify is that
its absence degrades cleanly, which is the part I control.

### Q28. What are the weaknesses of this system?

Several, and I would rather name them than be asked.

**The ensemble weights are guesses.** Informed by measured per-model performance,
but not learned. With ground-truth prices for scraped deals I could fit them by
regression.

**There is no ground truth loop.** I never find out whether a flagged deal was
actually a bargain. Without that I cannot compute precision or recall on the
thing the system exists to do — every estimate is unvalidated. The fix is to
record alerts and check prices later, or spot-check manually.

**The estimates are only as good as the catalogue.** For a product unlike
anything indexed, the five nearest neighbours are the least-distant of a bad set
and the anchor actively misleads. Nothing currently detects that — a minimum
similarity threshold on the retrieved comparables would.

**Memory does not scale.** Whole-file rewrite per add, O(n) `seen` property.

**Scraping is fragile.** It depends on a CSS class name on a site I do not
control. There is a fallback in `clean_html`, but `from_entry` returning `None` on
any failure means a site redesign would silently produce zero deals.

**The spread is collected but unused.** A real confidence signal I have not yet
acted on.

### Q29. How would you know if this system is working well?

This is the question I find most useful, and the honest answer is that I
currently cannot fully answer it.

What I *can* measure now: how many deals are scanned versus shortlisted, how
often estimators disagree (spread), how often the threshold is cleared, and
whether dedup is working — which I verified by watching memory grow across cycles
while the scanner reported fewer fresh listings.

What I **cannot** measure without more work is the thing that matters: were the
alerts actually good deals? That needs ground truth.

I would get it by recording every alert with its URL and estimate, then checking
the actual price some days later — either by re-scraping or manually. That gives
precision on alerts. Recall is harder, because it requires knowing about the good
deals the system missed, which likely means sampling non-alerted deals and
checking them by hand.

With that I could compute the estimator error per source, fit the ensemble
weights properly instead of hand-setting them, and calibrate the threshold
against a real false-positive rate. Right now all three are judgement calls.

### Q30. What would you do differently if you started again?

**Build the ground truth loop first.** Every other improvement — learned weights,
a calibrated threshold, choosing between estimators — depends on knowing whether
alerts were good. I built the system that produces estimates before the system
that evaluates them, and that ordering means most of my tuning decisions are
unvalidated.

**Share the model code properly.** The Neural agent duplicates architecture
classes from the price prediction project. It works, but the architecture version
should be in the checkpoint and both projects should depend on one package.

**Define an explicit `Pricer` protocol.** The three estimators share a contract
that currently only exists in my head and in the Ensemble's duck typing.

**Use SQLite for memory from the start.** JSON was the right call for getting
something running, and it is the first thing that breaks at scale.

**Add a retrieval quality check.** The Frontier agent's estimate is only as good
as its comparables, and nothing currently notices when they are poor. A minimum
similarity threshold, falling back to a wider estimate range, would make the
weakest link visible.

What I would keep: the seven-agent decomposition, the degradation model, the
two-key deduplication, recording the individual estimates rather than just the
combined number, and the OpenAI-compatible base URL — which turned "I cannot
verify this works" into a full end-to-end run.
