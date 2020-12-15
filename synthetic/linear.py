from typing import Callable, Tuple

import numpy as np
import scipy.optimize as optimize


def linear_objective(
    weights: np.ndarray, samples: np.ndarray, labels: np.ndarray, loss_fn: Callable
) -> float:
    """Calculates the final loss objective for a linear model"""
    z = (samples @ weights) * labels  #  calculate scores for loss
    loss = np.mean(loss_fn(z)).item()  #  mean loss value over the minibatch
    return loss


def train_linear_sgd(
    samples: np.ndarray,
    labels: np.ndarray,
    loss_fn: Callable,
    grad_fn: Callable,
    lr: float = 0.1,
    num_steps: int = 1000,
) -> Tuple[np.ndarray, float]:
    """Trains a linear classifier with Stochastic Gradient Descent.

    Returns:
        Tuple containing the final weights of the model and the associated loss
    """
    weights = np.zeros((samples.shape[1],))
    for i in range(num_steps):
        # sample from dataset
        index = np.random.choice(samples.shape[0])
        x, y = samples[index : index + 1], labels[index : index + 1]

        #  update the model
        gradient = grad_fn(x, y, weights)
        weights -= lr * np.mean(gradient, axis=0)
    return weights, linear_objective(samples, labels, weights, loss_fn)


def train_linear_slsqp(
    samples: np.ndarray, labels: np.ndarray, loss_fn: Callable
) -> Tuple[np.ndarray, float]:
    """Trains a linear classifier with SLSQP from scipy

    Returns:
        Tuple containing the final weights of the model and the associated loss
    """
    weights = np.zeros((samples.shape[1],))

    # optimize with scipy SLSQP
    opt_result = optimize.minimize(
        linear_objective, weights, (samples, labels, loss_fn), method="SLSQP"
    )

    # return weights of the linear classifier
    weights = opt_result["x"]
    return weights, linear_objective(weights, samples, labels, loss_fn)


def evaluate_linear(
    samples: np.ndarray, labels: np.ndarray, weights: np.ndarray, loss_fn: Callable
) -> Tuple[float, float]:
    """Evaluates the given linear classifier"""
    z = samples @ weights  # get logits

    # calculate predictions ((z <= 0) => -1, (z > 0) => +1)
    preds = np.ones_like(z)
    preds[z <= 0] = -1

    # calculate loss & acc
    loss = np.mean(loss_fn(z * labels)).item()
    acc = np.mean(np.equal(preds, labels)).item()

    return loss, acc
