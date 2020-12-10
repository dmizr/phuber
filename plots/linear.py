from typing import Callable, Tuple

import numpy as np
import scipy.optimize as optimize


def train_linear(
    samples: np.ndarray, labels: np.ndarray, loss_fn: Callable
) -> np.ndarray:
    """Trains a linear classifier"""
    weights = np.zeros((samples.shape[1],))

    def fun(weights, samples, labels, loss_fn):
        z = samples @ weights
        loss = np.mean(loss_fn(z * labels))
        return loss

    opt_result = optimize.minimize(
        fun, weights, (samples, labels, loss_fn), method="SLSQP"
    )

    return opt_result["x"]


def evaluate_linear(
    samples: np.ndarray, labels: np.ndarray, weights: np.ndarray, loss_fn: Callable
) -> Tuple[float, float]:
    """Evaluates the given linear classifier"""
    z = samples @ weights
    preds = np.ones_like(z)
    preds[z <= 0] = -1
    return np.mean(loss_fn(z * labels)), np.mean(np.equal(preds, labels))
