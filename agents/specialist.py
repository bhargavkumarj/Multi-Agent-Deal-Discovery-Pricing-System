from agents.base import Agent
from core import logs
from core.config import settings


class SpecialistAgent(Agent):
    """Calls the fine-tuned pricing model running on a Modal GPU.

    Connecting is done lazily and is allowed to fail: if Modal is not set up the
    ensemble simply carries on with the estimators it does have.
    """

    name = "Specialist"
    colour = logs.RED

    def __init__(self):
        import modal

        self.log(f"Connecting to Modal app {settings.modal_app}")
        pricer = modal.Cls.from_name(settings.modal_app, settings.modal_class)
        self.service = pricer()
        self.log("Connected")

    def price(self, description: str) -> float:
        estimate = float(self.service.price.remote(description))
        self.log(f"Estimates ${estimate:,.2f}")
        return estimate
