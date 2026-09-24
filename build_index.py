"""Build the product vector store the Frontier agent retrieves from.

Embeds curated product summaries with their real prices, so that at pricing time
the LLM can be shown what comparable items actually sell for.
"""

import argparse

import chromadb
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from core.config import settings

BATCH = 1_000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="ed-donner/items_lite")
    parser.add_argument("--limit", type=int, default=None, help="only index the first N products")
    parser.add_argument("--reset", action="store_true", help="drop an existing collection first")
    args = parser.parse_args()

    train = load_dataset(args.dataset, split="train")
    if args.limit:
        train = train.select(range(min(args.limit, len(train))))
    print(f"Indexing {len(train):,} products from {args.dataset}")

    client = chromadb.PersistentClient(path=str(settings.vector_store))
    if args.reset and settings.collection in [c.name for c in client.list_collections()]:
        client.delete_collection(settings.collection)
    collection = client.get_or_create_collection(settings.collection)

    encoder = SentenceTransformer(settings.embedding_model)
    for start in tqdm(range(0, len(train), BATCH)):
        rows = train.select(range(start, min(start + BATCH, len(train))))
        documents = [row["summary"] or row["title"] for row in rows]
        collection.add(
            ids=[f"product_{start + offset}" for offset in range(len(documents))],
            documents=documents,
            embeddings=encoder.encode(documents).astype(float).tolist(),
            metadatas=[
                {"category": row["category"], "price": float(row["price"])} for row in rows
            ],
        )

    print(f"{collection.count():,} products indexed into {settings.vector_store}")


if __name__ == "__main__":
    main()
