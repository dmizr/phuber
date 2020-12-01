import math

import torch
import torch.nn as nn


class CrossEntropy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = self.softmax(input)
        p = p[torch.arange(p.shape[0]), target]
        loss = -torch.log(p)
        return torch.mean(loss)


class PHuberCrossEntropy(nn.Module):
    def __init__(self, tau: float = 10) -> None:
        super().__init__()
        self.tau = tau
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = self.softmax(input)
        p = p[torch.arange(p.shape[0]), target]

        loss = torch.where(
            p <= 1 / self.tau,
            -self.tau * p + math.log(self.tau) + 1,
            -torch.log(p),
        )
        return torch.mean(loss)
