"""The fine-tuned pricing model, served on a Modal GPU.

Deploy with `modal deploy modal_app/pricer_service.py`. The Specialist agent
calls the deployed class by name; the weights stay in a Modal volume so the
container does not pull them from the Hub on every cold start.
"""

import modal

APP_NAME = "pricer-service"
BASE_MODEL = "meta-llama/Llama-3.2-3B"
FINETUNED_MODEL = "ed-donner/price-2025-11-28_18.47.07"
REVISION = "b19c8bfea3b6ff62237fbb0a8da9779fc12cefbd"
CACHE = "/cache"

QUESTION = "What does this cost to the nearest dollar?"
PREFIX = "Price is $"

app = modal.App(APP_NAME)
image = modal.Image.debian_slim().pip_install(
    "huggingface_hub", "torch", "transformers", "bitsandbytes", "accelerate", "peft"
)
secrets = [modal.Secret.from_name("huggingface-secret")]
cache = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)


@app.cls(
    image=image.env({"HF_HUB_CACHE": CACHE}),
    secrets=secrets,
    gpu="T4",
    timeout=1800,
    min_containers=0,
    volumes={CACHE: cache},
)
class Pricer:
    @modal.enter()
    def load(self):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        quantisation = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, quantization_config=quantisation, device_map="auto"
        )
        self.model = PeftModel.from_pretrained(base, FINETUNED_MODEL, revision=REVISION)

    @modal.method()
    def price(self, description: str) -> float:
        import re

        import torch
        from transformers import set_seed

        set_seed(42)
        prompt = f"{QUESTION}\n\n{description}\n\n{PREFIX}"
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = self.model.generate(inputs, max_new_tokens=5)

        reply = self.tokenizer.decode(outputs[0]).split(PREFIX)[-1].replace(",", "")
        match = re.search(r"[-+]?\d*\.\d+|\d+", reply)
        return float(match.group()) if match else 0.0


@app.local_entrypoint()
def main(description: str = "Quadcast HyperX condenser microphone for streaming"):
    print(f"${Pricer().price.remote(description):,.2f}")
