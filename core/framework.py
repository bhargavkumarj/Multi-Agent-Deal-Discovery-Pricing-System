import chromadb

from agents.deals import Opportunity
from agents.planner import PlanningAgent
from core import logs
from core.config import settings
from core.memory import Memory


class DealDiscovery:
    """Owns the vector store, the memory and the agent that drives a cycle."""

    def __init__(self, **ensemble_options):
        logs.configure()
        self.memory = Memory()
        self.collection = self._collection()
        self.ensemble_options = ensemble_options
        self.planner: PlanningAgent | None = None

    def _collection(self):
        if not settings.vector_store.exists():
            raise FileNotFoundError(
                f"No product vector store at {settings.vector_store}. "
                "Run `python build_index.py` first."
            )
        client = chromadb.PersistentClient(path=str(settings.vector_store))
        return client.get_or_create_collection(settings.collection)

    def log(self, message: str) -> None:
        import logging

        logging.info(f"{logs.BG_BLUE}{logs.WHITE}[Framework] {message}{logs.RESET}")

    def start(self) -> PlanningAgent:
        if self.planner is None:
            self.log("Starting the agents")
            self.planner = PlanningAgent(self.collection, self.memory, **self.ensemble_options)
        return self.planner

    def cycle(self) -> Opportunity | None:
        planner = self.start()
        self.log(f"Cycle starting, {len(self.memory)} deals in memory")
        result = planner.run()
        self.log("Cycle complete" + (f": alerted on ${result.discount:,.2f} off" if result else ""))
        return result

    def plot_data(self, limit: int = 1500):
        """3D t-SNE projection of the product catalogue for the monitoring UI."""
        import numpy as np
        from sklearn.manifold import TSNE

        result = self.collection.get(
            include=["embeddings", "documents", "metadatas"], limit=limit
        )
        vectors = np.array(result["embeddings"])
        categories = [metadata["category"] for metadata in result["metadatas"]]
        reduced = TSNE(n_components=3, random_state=42, init="pca").fit_transform(vectors)
        return result["documents"], reduced, categories
