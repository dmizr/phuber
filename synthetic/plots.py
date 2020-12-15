from typing import Dict, List

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

    # Plot decision boundary by generating meshgrid of points
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

    if save:
        plt.savefig(save_name)

    if show:
        plt.show()


def long_servedio_boxplot(
    test_accs: List[List[float]],
    losses_text: List[str],
    show=True,
    save=False,
    save_name="result.png",
    seaborn_context="paper",
) -> None:
    """Displays the boxplot from the Long & Servedio synthetic experiment"""

    sns.set_context(seaborn_context)
    sns.set_style("darkgrid")

    fig, ax = plt.subplots()
    ax = sns.boxplot(data=test_accs, showfliers=False, ax=ax)
    sns.despine(fig)
    ax.set_xticklabels(losses_text, rotation=8)
    fig.tight_layout()

    if save:
        plt.savefig(save_name, dpi=100)

    if show:
        plt.show()


def outliers_lineplot(
    thetas: np.ndarray,
    losses: Dict[str, list],
    show=True,
    save=False,
    save_name="result.png",
    seaborn_context="notebook",
) -> None:
    """
    Displays the lineplot from the Outliers synthetic experiment
    """

    sns.set_context(seaborn_context)
    sns.set_style("darkgrid")

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(thetas, losses["logistic_inliers"], color="green", linewidth=2)
    ax.plot(thetas, losses["huber_inliers"], color="red", linewidth=2)
    ax.plot(thetas, losses["phuber_inliers"], color="blue", linewidth=2)
    ax.plot(thetas, losses["logistic_all"], "--", color="green", linewidth=2)
    ax.plot(thetas, losses["huber_all"], "--", color="red", linewidth=2)
    ax.plot(thetas, losses["phuber_all"], "--", color="blue", linewidth=2)

    ax.legend(["Logistic", "Huber", "Partial Huber"])
    ax.set_xlabel(r"$\Theta$")
    ax.set_ylabel(r"R($\Theta$)")
    ax.set_xlim([-2.0, 2.0])
    ax.set_ylim([0, 1.3])
    fig.tight_layout()

    if save:
        plt.savefig(save_name, dpi=100)

    if show:
        plt.show()
