import re

from openai import OpenAI

from agents.base import Agent
from core import logs
from core.config import openai_client, settings

NUMBER = re.compile(r"[-+]?\d*\.\d+|\d+")

PROMPT = """Estimate what this product sells for, to the nearest dollar.

{description}

{context}Reply with the price only, as a number. No explanation, no currency symbol."""


class FrontierAgent(Agent):
    """Prices a product by retrieving similar items and their real prices first.

    The LLM on its own has no idea what this year's model of a mid-range router
    goes for. Five priced neighbours from the catalogue anchor the estimate.
    """

    name = "Frontier"
    colour = logs.BLUE

    def __init__(self, collection, client: OpenAI | None = None):
        self.collection = collection
        self.client = client or openai_client()
        self.model = settings.frontier_model
        self.log("Loading the sentence transformer")
        from sentence_transformers import SentenceTransformer

        self.encoder = SentenceTransformer(settings.embedding_model)

    def similar_products(self, description: str) -> tuple[list[str], list[float]]:
        vector = self.encoder.encode([description]).astype(float).tolist()
        results = self.collection.query(
            query_embeddings=vector, n_results=settings.similar_products
        )
        documents = results["documents"][0]
        prices = [metadata["price"] for metadata in results["metadatas"][0]]
        return documents, prices

    @staticmethod
    def context_for(documents: list[str], prices: list[float]) -> str:
        if not documents:
            return ""
        lines = ["Here are similar products and what they sell for:\n"]
        for document, price in zip(documents, prices):
            lines.append(f"{document}\nPrice is ${price:.2f}\n")
        return "\n".join(lines) + "\n"

    @staticmethod
    def to_price(reply: str) -> float:
        match = NUMBER.search(reply.replace("$", "").replace(",", ""))
        return float(match.group()) if match else 0.0

    def price(self, description: str) -> float:
        documents, prices = self.similar_products(description)
        self.log(f"Retrieved {len(documents)} similar products from the vector store")
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": PROMPT.format(
                        description=description, context=self.context_for(documents, prices)
                    ),
                }
            ],
            seed=42,
        )
        estimate = self.to_price(response.choices[0].message.content)
        self.log(f"Estimates ${estimate:,.2f}")
        return estimate
