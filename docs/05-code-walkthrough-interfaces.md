# Code Walkthrough — Entry Points and the Modal Service

Covers `build_index.py`, `run.py`, `app.py` and `modal_app/pricer_service.py`.

---

## `build_index.py`

Builds the priced product catalogue the Frontier agent retrieves from. Nothing
else works until this has run.

### What goes in the index

```python
documents = [row["summary"] or row["title"] for row in rows]
collection.add(
    ids=[f"product_{start + offset}" for offset in range(len(documents))],
    documents=documents,
    embeddings=encoder.encode(documents).astype(float).tolist(),
    metadatas=[{"category": row["category"], "price": float(row["price"])} for row in rows],
)
```

Three things per product: the text (embedded and searchable), the embedding, and
metadata carrying **the real price**.

The price in the metadata is the entire point. Retrieving five similar product
descriptions would be useless; retrieving five similar products *and what they
sold for* is what turns an LLM guess into an estimate.

`row["summary"] or row["title"]` — the LLM-written summary if the dataset has
one, otherwise the raw title. The summaries come from the price prediction
project's curation stage, and they are normalised five-line descriptions, which
embed much more consistently than raw marketing copy.

`float(row["price"])` because the Hub stores it as a string in some revisions,
and Chroma metadata must be a JSON scalar. Without the cast, `metadata["price"]`
comes back as `"64.3"` and the f-string in `FrontierAgent.context_for` would
format a string with `:.2f` and raise.

### Batching

```python
BATCH = 1_000
for start in tqdm(range(0, len(train), BATCH)):
    rows = train.select(range(start, min(start + BATCH, len(train))))
```

Embedding 20,000 products in one call would hold every vector in memory and give
no progress feedback. 1,000 per batch keeps memory flat and the progress bar
honest. Measured: 4,000 products in about 10 seconds on CPU.

### `--reset`

```python
if args.reset and settings.collection in [c.name for c in client.list_collections()]:
    client.delete_collection(settings.collection)
collection = client.get_or_create_collection(settings.collection)
```

Opt-in, not automatic. Chroma's `add` appends, so re-running without `--reset`
would duplicate every product — and duplicate comparables in a retrieval result
would skew the price context by weighting one product five times.

Making it a flag rather than the default means an interrupted build can be
resumed by re-running, and a deliberate rebuild is an explicit choice.

### `--limit`

```python
if args.limit:
    train = train.select(range(min(args.limit, len(train))))
```

Index a subset. The measured runs used `--limit 4000` — enough for meaningful
retrieval, fast enough to iterate on. `min(..., len(train))` prevents an
out-of-range select when the limit exceeds the dataset.

---

## `run.py`

The operational CLI.

### `--memory` short-circuits everything

```python
if args.memory:
    logs.configure()
    show(Memory(), args.limit)
    return
```

Placed before `DealDiscovery` is constructed, so inspecting memory loads no
models, opens no Chroma client and makes no network calls. It reads one JSON file
and prints.

That matters because the natural way to check on a running system is to look at
what it has found, and that should be instant.

### `show()`

```python
for opportunity in memory.best(limit):
    flag = "alerted" if memory.was_alerted(opportunity) else "       "
    print(
        f"  {flag}  ${opportunity.discount:>8,.2f} off   "
        f"paid ${opportunity.deal.price:>8,.2f}   "
        f"worth ${opportunity.estimate:>8,.2f}   "
        f"{opportunity.deal.product_description[:60]}"
    )
```

Right-aligned currency with `:>8,.2f` so the columns line up and the numbers are
scannable. The `flag` is either `"alerted"` or seven spaces — same width, so
alignment holds.

Sorted by discount, best first, which puts the interesting rows at the top.

Real output:

```
  alerted  $  104.76 off   paid $   57.00   worth $  161.76   Amazon Smart Thermostat
           $   53.73 off   paid $   49.99   worth $  103.72   SAMA Z60 White ATX PC Case
           $   27.50 off   paid $   34.99   worth $   62.49   Blink Outdoor 2K+ Doorbell
           $ -213.25 off   paid $1,429.00   worth $1,215.75   MSI Gaming Codex R2 Desktop
           $ -319.21 off   paid $  513.00   worth $  193.79   Open-box Unlocked Cell Phones
```

The negative rows are the system working correctly — it priced them, found them
overpriced relative to its estimate, and stayed quiet.

### Mutating settings from flags

```python
settings.discount_threshold = args.threshold
settings.notify = not args.no_notify

framework = DealDiscovery(
    use_specialist=not args.no_specialist, use_neural=not args.no_neural
)
```

Two mechanisms, and the split is deliberate.

`discount_threshold` and `notify` are read deep inside the Planner and Messenger,
so mutating the singleton avoids threading them through four constructors.

`use_specialist` and `use_neural` are **constructor arguments**, passed through
`DealDiscovery` into `EnsembleAgent`. They affect object construction, not
runtime behaviour, so they belong in the constructor where the decision is
visible.

### The loop

```python
while True:
    framework.cycle()
    if not args.loop:
        break
    print(f"\nSleeping {args.every}s\n")
    time.sleep(args.every)
```

A `while True` with a conditional break rather than a do-while, because Python
has no do-while. One cycle always runs; `--loop` makes it repeat.

The planner is constructed once inside `framework.start()` and cached, so a long
loop pays model-loading cost once.

The default of 600 seconds is matched to how fast deal feeds actually update.
Polling faster wastes API calls on unchanged content — the scanner's
already-seen filter would drop almost everything.

---

## `app.py`

The Gradio monitoring dashboard.

### Streaming the log into the UI

```python
class LogStream(logging.Handler):
    def __init__(self):
        super().__init__()
        self.queue: queue.Queue[str] = queue.Queue()
        self.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S"))

    def emit(self, record):
        self.queue.put(self.format(record))
```

A custom logging handler that pushes formatted records onto a thread-safe queue
instead of writing them out.

This is the clean way to get agent output into a UI. The agents know nothing
about Gradio — they call `logging.info` as always — and the dashboard attaches a
handler to the root logger. Adding an eighth agent requires no UI change.

`queue.Queue` is thread-safe by design, which matters because the cycle runs on a
worker thread and the UI reads on another.

### The generator that drives the UI

```python
def cycle(self, lines):
    result: list = []
    worker = threading.Thread(target=lambda: result.append(self.framework.cycle()))
    worker.start()

    while worker.is_alive() or not self.stream.queue.empty():
        try:
            lines = lines + [logs.to_html(self.stream.queue.get(timeout=0.2))]
        except queue.Empty:
            continue
        yield lines, panel(lines), rows(self.framework.memory)

    worker.join()
    yield lines, panel(lines), rows(self.framework.memory)
```

The cycle runs on a background thread while the generator drains the log queue
and yields updates. Without the thread, `framework.cycle()` would block for a
minute or more and the UI would freeze with no output until it finished.

**`worker.is_alive() or not queue.empty()`** — keep draining after the worker
finishes, because the last few log lines are still queued. Checking only
`is_alive()` would truncate the transcript at exactly the interesting moment.

**`get(timeout=0.2)`** rather than `get_nowait()` in a tight loop. The timeout
blocks briefly instead of spinning the CPU, and `queue.Empty` is caught to
re-check the loop condition.

**`lines = lines + [...]`** builds a new list rather than mutating Gradio's state
object.

**The final `yield` after `join()`** guarantees a last update with the complete
log and the final table, even if the loop exited on an empty queue.

`logs.to_html` converts the ANSI colours into spans, so the dashboard shows the
same colour-coded transcript the terminal does.

### `panel()`

```python
body = "<br>".join(lines[-30:])
return (
    "<div style='height:380px;overflow-y:auto;border:1px solid #333;"
    f"background:#16161d;padding:10px;font-family:ui-monospace,monospace;font-size:12px'>{body}</div>"
)
```

Last 30 lines only. A long `--loop` session accumulates thousands, and
re-rendering all of them on every token would make the UI crawl.

Fixed height with `overflow-y: auto` so the panel does not resize as it fills —
a growing div would push the rest of the page around on every update.

### Three refresh paths

```python
ui.load(self.cycle, inputs=[lines], outputs=[lines, log_view, table])
scan.click(self.cycle, inputs=[lines], outputs=[lines, log_view, table])
gr.Timer(value=600).tick(self.cycle, inputs=[lines], outputs=[lines, log_view, table])
```

Run on page load, on button click, and every ten minutes — all bound to the same
generator. Opening the dashboard immediately does something useful rather than
showing an empty page.

### The catalogue plot

```python
figure.update_layout(
    height=380,
    margin=dict(l=0, r=0, t=10, b=0),
    scene=dict(xaxis_title="", yaxis_title="", zaxis_title=""),
    template="plotly_dark",
)
```

Axis titles are blanked because **t-SNE axes have no meaning**. Labelling them
"x", "y", "z" would imply they do. Only the cluster structure is meaningful, and
the per-category colouring is what makes it readable.

`plotly_dark` matches the dark log panel.

---

## `modal_app/pricer_service.py`

The fine-tuned pricer, deployed to a serverless GPU.

### Infrastructure as code

```python
app = modal.App(APP_NAME)
image = modal.Image.debian_slim().pip_install(
    "huggingface_hub", "torch", "transformers", "bitsandbytes", "accelerate", "peft"
)
secrets = [modal.Secret.from_name("huggingface-secret")]
cache = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)
```

The container image, secrets and persistent volume are declared in Python. No
Dockerfile, no separate deployment config — `modal deploy` reads this file and
provisions everything.

**The volume is the important one.** Without it, every cold start would download
a 3B model from HuggingFace — gigabytes, and minutes of latency. Mounting a
persistent volume at `HF_HUB_CACHE` means the download happens once ever.

`modal.Secret.from_name` pulls the HF token from Modal's secret store rather than
the environment, so the token never appears in this file or in the client.

### `@modal.enter()` — the cold-start boundary

```python
@app.cls(image=image.env({"HF_HUB_CACHE": CACHE}), secrets=secrets, gpu="T4",
         timeout=1800, min_containers=0, volumes={CACHE: cache})
class Pricer:
    @modal.enter()
    def load(self):
        quantisation = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        ...
        base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=quantisation, device_map="auto")
        self.model = PeftModel.from_pretrained(base, FINETUNED_MODEL, revision=REVISION)
```

`@modal.enter()` runs **once per container**, not per request. Model loading is
the expensive part; inference is cheap. Putting the load in `__init__` or in
`price()` would reload a 3B model on every call.

**4-bit NF4 quantisation** takes a 3B model from roughly 6GB in fp16 to under
2GB, which is what lets it fit comfortably on a T4 — the cheapest useful GPU
Modal offers. `nf4` is the information-theoretically optimal 4-bit type for
normally-distributed weights, and double quantisation compresses the
quantisation constants themselves. `compute_dtype=float16` means matmuls still
run in fp16 while weights are stored in 4-bit.

**`PeftModel.from_pretrained(base, FINETUNED_MODEL, revision=REVISION)`** loads a
LoRA adapter on top of the base model. Only the adapter is fine-tuned, so the
artefact is megabytes rather than gigabytes. Pinning `revision` to a commit hash
means a redeploy loads exactly the same weights — a mutable branch reference
would let the model change silently under a running service.

**`min_containers=0`** means no idle cost and a cold start on the first request
after a quiet period. For a system that runs every ten minutes, that trade is
correct: paying for an always-warm GPU to serve a handful of calls an hour would
dominate the entire project's cost.

**`timeout=1800`** gives a cold start room to download and load without being
killed.

### `price()`

```python
set_seed(42)
prompt = f"{QUESTION}\n\n{description}\n\n{PREFIX}"
inputs = self.tokenizer.encode(prompt, return_tensors="pt").to("cuda")
with torch.no_grad():
    outputs = self.model.generate(inputs, max_new_tokens=5)

reply = self.tokenizer.decode(outputs[0]).split(PREFIX)[-1].replace(",", "")
match = re.search(r"[-+]?\d*\.\d+|\d+", reply)
return float(match.group()) if match else 0.0
```

**The prompt ends mid-sentence at `"Price is $"`.** That is the fine-tuning
format: training examples ended with the answer, test prompts stop just before
it, so the model's most likely continuation is the number. It is a completion
task, not a chat task, which is why there is no chat template.

**`max_new_tokens=5`** is enough for `"249.00"` and not enough for an
explanation. It bounds latency to a handful of forward passes.

**`.split(PREFIX)[-1]`** takes everything after the *last* occurrence of
`"Price is $"`. The decoded output includes the prompt, and `[-1]` is robust to
the phrase appearing in the description too.

**`set_seed(42)`** makes generation reproducible, matching the `seed=42` on the
Frontier agent. An estimator feeding a threshold decision should not vary between
identical calls.

Imports are **inside** the method because they only exist in the Modal container
— the local machine running `agents/specialist.py` has no torch, transformers or
peft, and does not need them.

### `@app.local_entrypoint()`

```python
@app.local_entrypoint()
def main(description: str = "Quadcast HyperX condenser microphone for streaming"):
    print(f"${Pricer().price.remote(description):,.2f}")
```

`modal run modal_app/pricer_service.py` exercises the deployed service from the
command line. A one-line smoke test that isolates "is the service working?" from
"is the agent working?" — worth having before debugging the ensemble.
