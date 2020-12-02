import math

import torch
import torch.nn as nn


class CrossEntropy(nn.Module):
    """Computes the cross-entropy loss

    Shape:
        - Input: the raw, unnormalized score for each class.
                tensor of size :math:`(minibatch, C)`, with C the number of classes
        - Target: the labels, tensor of size :math:`(minibatch)`, where each value
                is :math:`0 \leq targets[i] \leq C-1`
        - Output: scalar
    """

    def __init__(self) -> None:
        super().__init__()
        # Use log softmax as it has better numerical properties
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = self.log_softmax(input)
        p = p[torch.arange(p.shape[0]), target]

        loss = -p

        return torch.mean(loss)


class PHuberCrossEntropy(nn.Module):
    """Computes the partially Huberised (PHuber) cross-entropy loss, from
    `"Can gradient clipping mitigate label noise?"
    <https://openreview.net/pdf?id=rklB76EKPr>`_

    Args:
        tau: clipping threshold, must be > 1


    Shape:
        - Input: the raw, unnormalized score for each class.
                tensor of size :math:`(minibatch, C)`, with C the number of classes
        - Target: the labels, tensor of size :math:`(minibatch)`, where each value
                is :math:`0 \leq targets[i] \leq C-1`
        - Output: scalar
    """

    def __init__(self, tau: float = 10) -> None:
        super().__init__()
        self.tau = tau
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        p = self.softmax(input)
        p = p[torch.arange(p.shape[0]), target]

        loss = torch.empty_like(p)
        clip = p <= 1 / self.tau
        loss[clip] = -self.tau * p[clip] + math.log(self.tau) + 1
        loss[~clip] = -torch.log(p[~clip])

        return torch.mean(loss)
