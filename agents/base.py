import logging

from core import logs


class Agent:
    """Shared logging so every line in the transcript says who produced it."""

    name = "Agent"
    colour = logs.WHITE

    def log(self, message: str) -> None:
        logging.info(f"{logs.BG_BLACK}{self.colour}[{self.name}] {message}{logs.RESET}")
