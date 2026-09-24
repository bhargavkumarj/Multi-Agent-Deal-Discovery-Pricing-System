import logging
import sys

RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
WHITE = "\033[37m"
BG_BLACK = "\033[40m"
BG_BLUE = "\033[44m"
RESET = "\033[0m"

# The UI renders the same log lines as html, so terminal colours map to css ones.
HTML_COLOURS = {
    BG_BLACK + RED: "#dd0000",
    BG_BLACK + GREEN: "#00c853",
    BG_BLACK + YELLOW: "#d6c000",
    BG_BLACK + BLUE: "#4488ff",
    BG_BLACK + MAGENTA: "#aa00dd",
    BG_BLACK + CYAN: "#00cccc",
    BG_BLACK + WHITE: "#87ceeb",
    BG_BLUE + WHITE: "#ff7800",
}


def configure(level: int = logging.INFO) -> None:
    root = logging.getLogger()
    if any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
    )
    root.addHandler(handler)
    root.setLevel(level)


def to_html(message: str) -> str:
    for code, colour in HTML_COLOURS.items():
        message = message.replace(code, f'<span style="color: {colour}">')
    return message.replace(RESET, "</span>")
