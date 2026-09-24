import requests
from openai import OpenAI

from agents.base import Agent
from agents.deals import Opportunity
from core import logs
from core.config import openai_client, settings

PUSHOVER_URL = "https://api.pushover.net/1/messages.json"

PROMPT = """Write a two sentence push notification about this deal. Say what the item is,
what it costs and roughly how much below its estimated value it is. Be direct, no emoji,
no exclamation marks.

Item: {description}
Price: ${price:,.2f}
Estimated value: ${estimate:,.2f}"""


class MessagingAgent(Agent):
    name = "Messenger"
    colour = logs.WHITE

    def __init__(self, client: OpenAI | None = None):
        self.client = client or openai_client()
        self.model = settings.messenger_model
        if not settings.pushover_configured:
            self.log("Pushover is not configured, alerts will be logged only")

    def compose(self, opportunity: Opportunity) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": PROMPT.format(
                        description=opportunity.deal.product_description,
                        price=opportunity.deal.price,
                        estimate=opportunity.estimate,
                    ),
                }
            ],
        )
        return response.choices[0].message.content.strip()

    def push(self, text: str) -> bool:
        if not settings.pushover_configured:
            return False
        response = requests.post(
            PUSHOVER_URL,
            data={
                "user": settings.pushover_user,
                "token": settings.pushover_token,
                "message": text,
                "sound": "cashregister",
            },
            timeout=15,
        )
        return response.ok

    def alert(self, opportunity: Opportunity) -> str:
        text = self.compose(opportunity)
        message = f"{text}\n{opportunity.deal.url}"
        delivered = self.push(message[:900]) if settings.notify else False
        self.log(f"{'Sent' if delivered else 'Composed'} alert: {text[:90]}")
        return message
