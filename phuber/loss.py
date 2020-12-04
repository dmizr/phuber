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


class GeneralizedCrossEntropy(nn.Module):
    """Computes the generalized cross-entropy loss, from `
    "Generalized Cross Entropy Loss for Training Deep Neural Networks with Noisy Labels"
    <https://arxiv.org/abs/1805.07836>`_

    Args:
        q: Box-Cox transformation parameter, :math:`\in (0,1]`


    Shape:
        - Input: the raw, unnormalized score for each class.
                tensor of size :math:`(minibatch, C)`, with C the number of classes
        - Target: the labels, tensor of size :math:`(minibatch)`, where each value
                is :math:`0 \leq targets[i] \leq C-1`
        - Output: scalar
    """

    def __init__(self, q: float = 0.7) -> None:
        super().__init__()
        self.q = q
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = self.softmax(input)
        p = p[torch.arange(p.shape[0]), target]

        loss = (1 - p ** self.q) / self.q

        return torch.mean(loss)


class Unhinged(nn.Module):
    """Computes the Unhinged (linear) loss, from
    `"Learning with Symmetric Label Noise: The Importance of Being Unhinged"
    <https://arxiv.org/abs/1505.07634>`_


    Shape:
        - Input: the raw, unnormalized score for each class.
                tensor of size :math:`(minibatch, C)`, with C the number of classes
        - Target: the labels, tensor of size :math:`(minibatch)`, where each value
                is :math:`0 \leq targets[i] \leq C-1`
        - Output: scalar
    """

    def __init__(self) -> None:
        super().__init__()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = self.softmax(input)
        p = p[torch.arange(p.shape[0]), target]

        loss = 1 - p

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

        # Probability threshold for the clipping
        self.prob_thresh = 1 / self.tau
        # Negative of the Fenchel conjugate of base loss at tau
        self.boundary_term = math.log(self.tau) + 1

        self.softmax = nn.Softmax(dim=1)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = self.softmax(input)
        p = p[torch.arange(p.shape[0]), target]

        loss = torch.empty_like(p)
        clip = p <= self.prob_thresh
        loss[clip] = -self.tau * p[clip] + self.boundary_term
        loss[~clip] = -torch.log(p[~clip])

        return torch.mean(loss)


class PHuberGeneralizedCrossEntropy(nn.Module):
    """Computes the partially Huberised (PHuber) generalized cross-entropy loss, from
    `"Can gradient clipping mitigate label noise?"
    <https://openreview.net/pdf?id=rklB76EKPr>`_

    Args:
        q: Box-Cox transformation parameter, :math:`\in (0,1]`
        tau: clipping threshold, must be > 1


    Shape:
        - Input: the raw, unnormalized score for each class.
                tensor of size :math:`(minibatch, C)`, with C the number of classes
        - Target: the labels, tensor of size :math:`(minibatch)`, where each value
                is :math:`0 \leq targets[i] \leq C-1`
        - Output: scalar
    """

    def __init__(self, q: float = 0.7, tau: float = 10) -> None:
        super().__init__()
        self.q = q
        self.tau = tau

        # Probability threshold for the clipping
        self.prob_thresh = tau ** (1 / (q - 1))
        # Negative of the Fenchel conjugate of base loss at tau
        self.boundary_term = tau * self.prob_thresh + (1 - self.prob_thresh ** q) / q

        self.softmax = nn.Softmax(dim=1)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = self.softmax(input)
        p = p[torch.arange(p.shape[0]), target]

        loss = torch.empty_like(p)
        clip = p <= self.prob_thresh
        loss[clip] = -self.tau * p[clip] + self.boundary_term
        loss[~clip] = (1 - p[~clip] ** self.q) / self.q

        return torch.mean(loss)
