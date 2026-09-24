import json
import re
from pathlib import Path

from agents.deals import Deal, Opportunity
from core.config import settings

STOPWORDS = {
    "the", "a", "an", "and", "with", "for", "from", "that", "this", "your",
    "new", "refurb", "refurbished", "featuring", "includes", "comes",
}


def url_key(deal: Deal) -> str:
    """The listing URL without tracking parameters."""
    return deal.url.split("?")[0].rstrip("/").lower()


def content_key(deal: Deal) -> str:
    """A loose identity for the product itself.

    The same item shows up on several feeds with different URLs and slightly
    reworded copy, and the scanner occasionally shortlists it twice in one pass.
    Matching on the significant words of the description plus the rounded price
    catches those; matching on the URL alone does not.
    """
    words = re.findall(r"[a-z0-9]+", deal.product_description.lower())
    significant = sorted({w for w in words if w not in STOPWORDS and len(w) > 2})[:10]
    return " ".join(significant) + f"|{round(deal.price / 10) * 10}"


def keys(deal: Deal) -> set[str]:
    return {key for key in (url_key(deal), content_key(deal)) if key}


class Memory:
    """Everything the system has already looked at, so it never alerts twice."""

    def __init__(self, path: Path | None = None):
        self.path = path or settings.memory_file
        self.opportunities: list[Opportunity] = []
        self.alerted: set[str] = set()
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            return
        data = json.loads(self.path.read_text())
        rows = data.get("opportunities", []) if isinstance(data, dict) else data
        self.opportunities = [Opportunity(**row) for row in rows]
        self.alerted = set(data.get("alerted", [])) if isinstance(data, dict) else set()

    def save(self) -> None:
        self.path.write_text(
            json.dumps(
                {
                    "opportunities": [o.model_dump() for o in self.opportunities],
                    "alerted": sorted(self.alerted),
                },
                indent=2,
            )
        )

    @property
    def seen(self) -> set[str]:
        return {key for o in self.opportunities for key in keys(o.deal)}

    def seen_urls(self) -> set[str]:
        return {o.deal.url for o in self.opportunities}

    def has_seen(self, deal: Deal) -> bool:
        return bool(keys(deal) & self.seen)

    def add(self, opportunity: Opportunity) -> bool:
        if self.has_seen(opportunity.deal):
            return False
        self.opportunities.append(opportunity)
        self.save()
        return True

    def mark_alerted(self, opportunity: Opportunity) -> None:
        self.alerted |= keys(opportunity.deal)
        self.save()

    def was_alerted(self, opportunity: Opportunity) -> bool:
        return bool(keys(opportunity.deal) & self.alerted)

    def best(self, limit: int = 20) -> list[Opportunity]:
        return sorted(self.opportunities, key=lambda o: o.discount, reverse=True)[:limit]

    def __len__(self) -> int:
        return len(self.opportunities)
