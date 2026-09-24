import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)

ROOT = Path(__file__).resolve().parent.parent

FEEDS = [
    "https://www.dealnews.com/c142/Electronics/?rss=1",
    "https://www.dealnews.com/c39/Computers/?rss=1",
    "https://www.dealnews.com/f1912/Smart-Home/?rss=1",
]


def _float(name: str, default: float) -> float:
    return float(os.getenv(name, default))


def _flag(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    memory_file: Path = ROOT / "memory.json"
    vector_store: Path = ROOT / "products_vectorstore"
    collection: str = "products"
    artifacts: Path = ROOT / "artifacts"

    feeds: list[str] = field(default_factory=lambda: list(FEEDS))
    deals_per_feed: int = int(os.getenv("DEALS_PER_FEED", "10"))
    shortlist_size: int = int(os.getenv("SHORTLIST_SIZE", "5"))

    # Point this at any OpenAI-compatible endpoint (Ollama, vLLM, a gateway).
    api_base: str | None = os.getenv("OPENAI_BASE_URL") or None
    scanner_model: str = os.getenv("SCANNER_MODEL", "gpt-4.1-mini")
    frontier_model: str = os.getenv("FRONTIER_MODEL", "gpt-4.1-mini")
    messenger_model: str = os.getenv("MESSENGER_MODEL", "gpt-4.1-mini")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    modal_app: str = os.getenv("MODAL_APP", "pricer-service")
    modal_class: str = os.getenv("MODAL_CLASS", "Pricer")

    # Ensemble weights are renormalised over whichever estimators actually loaded.
    weight_frontier: float = _float("WEIGHT_FRONTIER", 0.6)
    weight_specialist: float = _float("WEIGHT_SPECIALIST", 0.25)
    weight_neural: float = _float("WEIGHT_NEURAL", 0.15)

    discount_threshold: float = _float("DISCOUNT_THRESHOLD", 50)
    similar_products: int = int(os.getenv("SIMILAR_PRODUCTS", "5"))

    notify: bool = _flag("NOTIFY", True)
    pushover_user: str | None = os.getenv("PUSHOVER_USER")
    pushover_token: str | None = os.getenv("PUSHOVER_TOKEN")

    def __post_init__(self):
        self.artifacts.mkdir(parents=True, exist_ok=True)

    @property
    def pushover_configured(self) -> bool:
        return bool(self.pushover_user and self.pushover_token)


settings = Settings()


def openai_client():
    from openai import OpenAI

    if settings.api_base:
        return OpenAI(base_url=settings.api_base, api_key=os.getenv("OPENAI_API_KEY", "local"))
    return OpenAI()
