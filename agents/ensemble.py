from agents.base import Agent
from agents.deals import Estimate
from agents.frontier import FrontierAgent
from core import logs
from core.config import settings


class EnsembleAgent(Agent):
    """Combines the three estimators into one number.

    Each one fails differently: the LLM anchors on similar products but drifts on
    anything unusual, the fine-tuned specialist is steady on familiar categories,
    the neural net is noisy but unbiased. A weighted mean of whichever are
    available is more stable than any of them alone, and the spread between them
    is a usable confidence signal.
    """

    name = "Ensemble"
    colour = logs.YELLOW

    def __init__(self, collection, use_specialist: bool = True, use_neural: bool = True):
        self.estimators: dict[str, object] = {"frontier": FrontierAgent(collection)}
        self.weights = {"frontier": settings.weight_frontier}

        if use_specialist:
            try:
                from agents.specialist import SpecialistAgent

                self.estimators["specialist"] = SpecialistAgent()
                self.weights["specialist"] = settings.weight_specialist
            except Exception as error:
                self.log(f"Specialist unavailable, skipping it ({error})")

        if use_neural:
            try:
                from agents.neural import NeuralAgent

                self.estimators["neural"] = NeuralAgent()
                self.weights["neural"] = settings.weight_neural
            except Exception as error:
                self.log(f"Neural estimator unavailable, skipping it ({error})")

        self.log(f"Ready with {', '.join(self.estimators)}")

    def estimates_for(self, description: str) -> list[Estimate]:
        estimates = []
        for source, estimator in list(self.estimators.items()):
            try:
                value = estimator.price(description)
            except Exception as error:
                # One broken estimator should not slow every remaining deal down,
                # so it is dropped for the rest of the run.
                self.log(f"{source} failed, dropping it for this run: {error}")
                del self.estimators[source]
                continue
            if value > 0:
                estimates.append(Estimate(source=source, value=value))
        return estimates

    def price(self, description: str) -> tuple[float, list[Estimate]]:
        estimates = self.estimates_for(description)
        if not estimates:
            return 0.0, []

        total = sum(self.weights[e.source] for e in estimates)
        combined = sum(e.value * self.weights[e.source] for e in estimates) / total
        detail = ", ".join(f"{e.source} ${e.value:,.0f}" for e in estimates)
        self.log(f"{detail} -> ${combined:,.2f}")
        return combined, estimates
