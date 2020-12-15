from typing import Union

import matplotlib.pyplot as plt
import numpy as np


def sigmoid(z: Union[np.ndarray, float]) -> np.ndarray:
    """Applies sigmoid function on given z."""

    # Clip to avoid numerical errors
    z = np.clip(z, -500, 500)
    out = 1.0 / (1.0 + np.exp(-z))
    return out


def inverse_sigmoid(z: Union[np.ndarray, float]) -> np.ndarray:
    """Applies inverse sigmoid function on given z."""
    return np.log(z / (1 - z))


def plot_boundaries(w: np.ndarray, samples: np.ndarray, labels: np.ndarray) -> None:
    """Plots decision boundaries of a linear model"""
    # Plot when normalized
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    eps = 1e-6

    # Plot decision boundary
    x0_min, x0_max = -1, 1
    x1_min, x1_max = -1, 1
    x0_diff = x0_max - x0_min
    x1_diff = x1_max - x1_min

    xx, yy = np.mgrid[
        x0_min - x0_diff / 10: x0_max + x0_diff / 10 + eps: x0_diff / 400,
        x1_min - x1_diff / 10: x1_max + x1_diff / 10 + eps: x1_diff / 400,
    ]
    grid = np.c_[xx.ravel(), yy.ravel()]
    out = (grid @ w).reshape(xx.shape)
    out = np.where(out > 0, 1, -1)

    contour = ax.contourf(xx, yy, out, 25, cmap="GnBu", vmin=-1, vmax=1, alpha=0.4)
    ax_c = fig.colorbar(contour)
    ax_c.set_ticks(np.arange(-1, 1.01, 0.1))

    plt.scatter(
        samples[labels == -1, 0], samples[labels == -1, 1], s=0.2, c="green", alpha=0.5
    )
    plt.scatter(
        samples[labels == 1, 0], samples[labels == 1, 1], s=0.2, c="blue", alpha=0.5
    )

    # End of plotting decision boundary
    plt.show()


def plot_data(samples: np.ndarray, labels: np.ndarray) -> None:
    """Plots data points"""
    plt.scatter(samples[labels == -1, 0], samples[labels == -1, 1], s=0.1, c="blue")
    plt.scatter(samples[labels == 1, 0], samples[labels == 1, 1], s=0.1, c="red")
    plt.show()
