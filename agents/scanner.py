from openai import OpenAI

from agents.base import Agent
from agents.deals import DealSelection, ScrapedDeal
from core import logs
from core.config import openai_client, settings

SYSTEM_PROMPT = """You select the most promising deals from a list of scraped listings.

Pick the ones with a detailed product description and an unambiguous price. Rewrite each
description so it describes the product itself — what it is, what it does, the specs that
matter — and not the terms of the discount.

Be careful with listings phrased as "$XXX off" or "reduced by $XXX": that is the saving,
not the price. If you cannot work out what the buyer actually pays, leave the deal out.
Never include a deal with a price of zero."""

USER_PROMPT = """Select the {count} best deals from the listings below and respond in the
required format. Include a deal only if you are confident about its price.

Listings:

{listings}"""


class ScannerAgent(Agent):
    """Turns noisy RSS listings into a short list of structured deals."""

    name = "Scanner"
    colour = logs.CYAN

    def __init__(self, client: OpenAI | None = None):
        self.client = client or openai_client()
        self.model = settings.scanner_model

    def fresh_listings(self, seen_urls: set[str]) -> list[ScrapedDeal]:
        self.log(f"Reading {len(settings.feeds)} feeds")
        scraped = ScrapedDeal.fetch()
        fresh = [deal for deal in scraped if deal.url not in seen_urls]
        self.log(f"{len(scraped)} listings, {len(fresh)} not seen before")
        return fresh

    def scan(self, seen_urls: set[str] | None = None) -> DealSelection | None:
        listings = self.fresh_listings(seen_urls or set())
        if not listings:
            return None

        prompt = USER_PROMPT.format(
            count=settings.shortlist_size,
            listings="\n\n".join(listing.describe() for listing in listings),
        )
        self.log(f"Asking {self.model} to shortlist {settings.shortlist_size} deals")
        response = self.client.chat.completions.parse(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            response_format=DealSelection,
        )
        selection = response.choices[0].message.parsed
        selection.deals = [deal for deal in selection.deals if deal.price > 0]
        self.log(f"Shortlisted {len(selection.deals)} deals with a usable price")
        return selection
