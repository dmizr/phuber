from typing import Callable
import numpy as np

from plots.loss import (
    logistic_loss,
    huberized_loss,
    partially_huberized_loss,
    logistic_gradient,
    huberized_gradient,
    partially_huberized_gradient
)
from plots.dataset import long_servedio
from plots.utils import sigmoid

def linear_regression(
    samples: np.ndarray,
    labels: np.ndarray,
    loss_fn: Callable,
    grad_fn: Callable,
    gamma: float = 1e-2,
    max_steps: int = 3000) -> np.ndarray:
    """Trains a linear classifier"""
    weights = np.zeros((samples.shape[1],))
    prev = None
    for i in range(max_steps):
        # sample from dataset
        index = np.random.choice(samples.shape[0])
        x, y = samples[index], labels[index]

        # update the model
        gradient = grad_fn(x, y, weights)
        weights -= gamma * gradient
    return weights

def linear_evaluate(samples: np.ndarray, labels: np.ndarray, weights: np.ndarray) -> float:
    """Evaluates the given linear classifier"""
    preds = sigmoid(samples @ weights)
    preds[preds > 0.5] = 1
    preds[preds <= 0.5] = -1
    return np.mean(np.equal(preds, labels))