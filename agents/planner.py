from datetime import datetime, timezone

from agents.base import Agent
from agents.deals import Deal, Opportunity
from agents.ensemble import EnsembleAgent
from agents.messenger import MessagingAgent
from agents.scanner import ScannerAgent
from core import logs
from core.config import settings
from core.memory import Memory, keys


class PlanningAgent(Agent):
    """Runs one cycle: scan, value, rank, alert.

    Only the single best opportunity in a cycle is alerted on, and only if it
    clears the discount threshold — the point of the system is a handful of
    alerts worth reading, not a feed of everything it looked at.
    """

    name = "Planner"
    colour = logs.GREEN

    def __init__(self, collection, memory: Memory, **ensemble_options):
        self.memory = memory
        self.scanner = ScannerAgent()
        self.ensemble = EnsembleAgent(collection, **ensemble_options)
        self.messenger = MessagingAgent()
        self.log("Ready")

    def value(self, deal: Deal) -> Opportunity:
        estimate, estimates = self.ensemble.price(deal.product_description)
        return Opportunity(
            deal=deal,
            estimate=estimate,
            discount=estimate - deal.price,
            estimates=estimates,
            found_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )

    def run(self) -> Opportunity | None:
        selection = self.scanner.scan(seen_urls=self.memory.seen_urls())
        if not selection or not selection.deals:
            self.log("Nothing new in the feeds")
            return None

        # Drop anything already in memory, and anything the scanner listed twice
        # in this pass — it does that when one product appears on several feeds.
        candidates, batch_keys = [], set()
        for deal in selection.deals:
            deal_keys = keys(deal)
            if self.memory.has_seen(deal) or deal_keys & batch_keys:
                continue
            batch_keys |= deal_keys
            candidates.append(deal)

        skipped = len(selection.deals) - len(candidates)
        if skipped:
            self.log(f"Skipped {skipped} duplicate or already-seen deals")
        if not candidates:
            return None

        opportunities = [self.value(deal) for deal in candidates[: settings.shortlist_size]]
        opportunities.sort(key=lambda o: o.discount, reverse=True)
        best = opportunities[0]
        self.log(
            f"Best candidate is ${best.discount:,.2f} under estimate "
            f"(estimator spread ${best.spread:,.0f})"
        )

        for opportunity in opportunities:
            self.memory.add(opportunity)

        if best.discount < settings.discount_threshold:
            self.log(f"Below the ${settings.discount_threshold:,.0f} threshold, no alert")
            return None

        self.messenger.alert(best)
        self.memory.mark_alerted(best)
        return best
