from typing import Callable, Tuple
import numpy as np

from plots.utils import sigmoid

def train_linear(
    samples: np.ndarray,
    labels: np.ndarray,
    loss_fn: Callable,
    grad_fn: Callable,
    gamma: float = 200,
    max_steps: int = 10000) -> np.ndarray:
    """Trains a linear classifier"""
    weights = np.zeros((samples.shape[1],))
    prev = None
    for i in range(max_steps):
        # sample from dataset
        index = np.random.choice(samples.shape[0])
        x, y = samples[index:index+1], labels[index:index+1]
        
        # update the model
        gradient = grad_fn(x, y, weights)
        weights -= gamma * np.mean(gradient, axis=0)

        if i % 4000 == 0:
            gamma *= 0.5
    return weights

def evaluate_linear(
    samples: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    loss_fn: Callable) -> Tuple[float, float]:
    """Evaluates the given linear classifier"""
    z = samples @ weights
    preds = np.ones_like(z)
    preds[z <= 0] = -1
    return np.mean(loss_fn(z * labels)), np.mean(np.equal(preds, labels))