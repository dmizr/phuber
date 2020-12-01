import math

import torch
import torch.nn as nn


class CrossEntropy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # Use log softmax as it has better numerical properties (avoids NaN)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = self.log_softmax(input)
        p = p[torch.arange(p.shape[0]), target]
        loss = -p
        return torch.mean(loss)


class PHuberCrossEntropy(nn.Module):
    def __init__(self, tau: float = 10) -> None:
        super().__init__()
        self.tau = tau
        # Use log softmax as it has better numerical properties (avoids NaN)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        p = self.log_softmax(input)
        p = p[torch.arange(p.shape[0]), target]

        # Adapt formula as p is in log-scale
        loss = torch.where(
            p <= math.log(1 / self.tau),
            -self.tau * torch.exp(p) + math.log(self.tau) + 1,
            -p,
        )

        return torch.mean(loss)
