import re
import time

import feedparser
import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field

from core.config import settings

HEADERS = {"User-Agent": "deal-discovery/1.0"}
TIMEOUT = 15


def clean_html(snippet: str) -> str:
    soup = BeautifulSoup(snippet, "html.parser")
    block = soup.find("div", class_="snippet summary")
    text = block.get_text(strip=True) if block else soup.get_text(" ", strip=True)
    return re.sub(r"\s+", " ", text).strip()


class ScrapedDeal:
    """A listing pulled off an RSS feed and enriched from its own page."""

    def __init__(self, title: str, summary: str, url: str, details: str = "", features: str = ""):
        self.title = title[:100]
        self.summary = summary
        self.url = url
        self.details = details[:500]
        self.features = features[:500]

    def __repr__(self) -> str:
        return f"<{self.title}>"

    def describe(self) -> str:
        return (
            f"Title: {self.title}\nDetails: {self.details.strip()}\n"
            f"Features: {self.features.strip()}\nURL: {self.url}"
        )

    @classmethod
    def from_entry(cls, entry) -> "ScrapedDeal | None":
        url = entry["links"][0]["href"]
        details, features = "", ""
        try:
            page = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
            soup = BeautifulSoup(page.content, "html.parser")
            section = soup.find("div", class_="content-section")
            if section:
                content = section.get_text().replace("\nmore", "").replace("\n", " ")
                if "Features" in content:
                    details, features = content.split("Features", 1)
                else:
                    details = content
        except requests.RequestException:
            # A deal page that will not load is not worth holding up the scan for.
            return None
        return cls(entry["title"], clean_html(entry["summary"]), url, details, features)

    @classmethod
    def fetch(cls, feeds: list[str] | None = None, per_feed: int | None = None):
        per_feed = per_feed or settings.deals_per_feed
        scraped = []
        for feed_url in feeds or settings.feeds:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries[:per_feed]:
                deal = cls.from_entry(entry)
                if deal:
                    scraped.append(deal)
                time.sleep(0.05)
        return scraped


class Deal(BaseModel):
    product_description: str = Field(
        description="Three or four sentences describing the product itself. Describe what the "
        "item is and what it does, not the terms of the discount."
    )
    price: float = Field(
        description="The price being charged for the item. If the listing says '$100 off the "
        "usual $300', the price is 200."
    )
    url: str = Field(description="The URL of the deal, exactly as given in the input")


class DealSelection(BaseModel):
    deals: list[Deal] = Field(
        description="The deals with the clearest price and the most detailed product description"
    )


class Estimate(BaseModel):
    """What one estimator thinks a product is worth."""

    source: str
    value: float


class Opportunity(BaseModel):
    deal: Deal
    estimate: float
    discount: float
    estimates: list[Estimate] = Field(default_factory=list)
    found_at: str | None = None

    @property
    def spread(self) -> float:
        """How far apart the estimators were — wide spread means low confidence."""
        values = [e.value for e in self.estimates]
        return max(values) - min(values) if len(values) > 1 else 0.0
