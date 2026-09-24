import math

import numpy as np
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import HashingVectorizer

from agents.base import Agent
from core import logs
from core.config import settings

FEATURES = 5_000
WEIGHTS = "residual_net.pt"


class ResidualBlock(nn.Module):
    def __init__(self, width: int, dropout: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(width, width),
            nn.LayerNorm(width),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(width, width),
            nn.LayerNorm(width),
        )
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.activation(self.block(x) + x)


class PriceNet(nn.Module):
    def __init__(self, inputs: int = FEATURES, width: int = 1024, blocks: int = 4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Linear(inputs, width), nn.LayerNorm(width), nn.ReLU(), nn.Dropout(0.2)
        )
        self.blocks = nn.ModuleList(ResidualBlock(width) for _ in range(blocks))
        self.head = nn.Linear(width, 1)

    def forward(self, x):
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x)


class NeuralAgent(Agent):
    """The cheap local estimator: a residual net over hashed text features.

    It is the weakest of the three on its own, but it costs nothing per call and
    it is wrong in different ways to the LLM, which is what makes it worth
    averaging in.
    """

    name = "Neural"
    colour = logs.MAGENTA

    def __init__(self, weights_path=None):
        path = weights_path or settings.artifacts / WEIGHTS
        if not path.exists():
            raise FileNotFoundError(
                f"No trained weights at {path}. Train the residual net in the price "
                "prediction project and copy residual_net.pt into artifacts/."
            )

        self.device = (
            torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("mps")
            if torch.backends.mps.is_available()
            else torch.device("cpu")
        )
        blob = torch.load(path, map_location=self.device)
        self.model = PriceNet().to(self.device)
        self.model.load_state_dict(blob["state"])
        self.model.eval()
        self.mean = float(blob["mean"])
        self.std = float(blob["std"])
        self.vectorizer = HashingVectorizer(
            n_features=FEATURES, stop_words="english", binary=True
        )
        np.random.seed(42)
        self.log(f"Loaded weights on {self.device}")

    def price(self, description: str) -> float:
        with torch.no_grad():
            matrix = self.vectorizer.transform([description])
            x = torch.FloatTensor(matrix.toarray()).to(self.device)
            scaled = float(self.model(x)[0].item())
            estimate = max(math.expm1(scaled * self.std + self.mean), 0.0)
        self.log(f"Estimates ${estimate:,.2f}")
        return estimate
