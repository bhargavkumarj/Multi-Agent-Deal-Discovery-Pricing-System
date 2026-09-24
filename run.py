"""Run the deal discovery agents: one cycle, or continuously."""

import argparse
import time

from core import logs
from core.config import settings
from core.framework import DealDiscovery
from core.memory import Memory


def show(memory: Memory, limit: int) -> None:
    if not memory:
        print("Nothing in memory yet.")
        return
    print(f"{len(memory)} deals in memory, best {limit} by discount:\n")
    for opportunity in memory.best(limit):
        flag = "alerted" if memory.was_alerted(opportunity) else "       "
        print(
            f"  {flag}  ${opportunity.discount:>8,.2f} off   "
            f"paid ${opportunity.deal.price:>8,.2f}   "
            f"worth ${opportunity.estimate:>8,.2f}   "
            f"{opportunity.deal.product_description[:60]}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loop", action="store_true", help="keep running")
    parser.add_argument("--every", type=int, default=600, help="seconds between cycles")
    parser.add_argument("--threshold", type=float, default=settings.discount_threshold)
    parser.add_argument("--no-specialist", action="store_true", help="skip the Modal model")
    parser.add_argument("--no-neural", action="store_true", help="skip the local neural net")
    parser.add_argument("--no-notify", action="store_true", help="compose alerts but do not send")
    parser.add_argument("--memory", action="store_true", help="print memory and exit")
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    if args.memory:
        logs.configure()
        show(Memory(), args.limit)
        return

    settings.discount_threshold = args.threshold
    settings.notify = not args.no_notify

    framework = DealDiscovery(
        use_specialist=not args.no_specialist, use_neural=not args.no_neural
    )

    while True:
        framework.cycle()
        if not args.loop:
            break
        print(f"\nSleeping {args.every}s\n")
        time.sleep(args.every)

    show(framework.memory, args.limit)


if __name__ == "__main__":
    main()
