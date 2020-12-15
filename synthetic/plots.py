from typing import List

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


def plot_boundaries(
    w: np.ndarray,
    samples: np.ndarray,
    labels: np.ndarray,
    show=True,
    save=False,
    save_name="boundaries.png",
) -> None:
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
        x0_min - x0_diff / 10 : x0_max + x0_diff / 10 + eps : x0_diff / 400,
        x1_min - x1_diff / 10 : x1_max + x1_diff / 10 + eps : x1_diff / 400,
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
    if save:
        plt.savefig(save_name)

    if show:
        plt.show()


def plot_data(
    samples: np.ndarray,
    labels: np.ndarray,
    show=True,
    save=False,
    save_name="data.png",
) -> None:
    """Plots data points"""
    plt.scatter(samples[labels == -1, 0], samples[labels == -1, 1], s=0.1, c="blue")
    plt.scatter(samples[labels == 1, 0], samples[labels == 1, 1], s=0.1, c="red")
    # End of plotting decision boundary
    if save:
        plt.savefig(save_name)

    if show:
        plt.show()


def boxplot_long_servedio(
    test_accs: List[List[float]],
    losses_text: List[str],
    show=True,
    save=False,
    save_name="result.png",
) -> None:
    """Displays the boxplot from the Long & Servedio synthetic experiment"""

    sns.set_theme(style="darkgrid")
    fig, ax = plt.subplots()
    ax = sns.boxplot(data=test_accs, showfliers=False, ax=ax)
    sns.despine(fig)
    ax.set_xticklabels(losses_text, rotation=8)

    if save:
        plt.savefig(save_name, dpi=100)

    if show:
        plt.show()
