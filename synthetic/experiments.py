import logging
from collections import defaultdict

import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm

from synthetic.dataset import long_servedio_dataset, outlier_dataset
from synthetic.linear import evaluate_linear, train_linear_sgd, train_linear_slsqp
from synthetic.loss import (
    huberised_gradient,
    huberised_loss,
    logistic_gradient,
    logistic_loss,
    partially_huberised_gradient,
    partially_huberised_loss,
)
from synthetic.plots import long_servedio_boxplot, outliers_lineplot, plot_boundaries


def long_servedio_experiment(cfg: DictConfig) -> None:
    """
    Long & Servedio (2010) synthetic experiment

    Args:
        cfg: Hydra config

    """
    logger = logging.getLogger()

    # prepare losses and gradients
    losses_text = ["Logistic", "Huberised", "Partial Huberised"]
    loss_fns = [logistic_loss, huberised_loss, partially_huberised_loss]
    grad_fns = [
        logistic_gradient,
        huberised_gradient,
        partially_huberised_gradient,
    ]

    # containers for results
    train_accs, train_losses = [[] for _ in range(3)], [[] for _ in range(3)]
    test_accs, test_losses = [[] for _ in range(3)], [[] for _ in range(3)]

    for _ in tqdm(range(cfg.n_repeat)):
        #  generate train and test sets
        train_samples, train_labels = long_servedio_dataset(
            N=cfg.n_train,
            gamma=cfg.gamma,
            var=cfg.var,
            corrupt_prob=cfg.corrupt_prob,
            noise_seed=cfg.seed,
        )
        test_samples, test_labels = long_servedio_dataset(
            N=cfg.n_test,
            gamma=cfg.gamma,
            var=cfg.var,
            corrupt_prob=0.0,
            noise_seed=cfg.seed + 1 if cfg.seed else None,
        )

        #  iterate over losses
        for i in range(3):
            # train linear model
            if cfg.method == "slsqp":
                weights, _ = train_linear_slsqp(
                    samples=train_samples, labels=train_labels, loss_fn=loss_fns[i]
                )
            elif cfg.method == "sgd":
                weights, _ = train_linear_sgd(
                    samples=train_samples,
                    labels=train_labels,
                    loss_fn=loss_fns[i],
                    grad_fn=grad_fns[i],
                    lr=cfg.lr if cfg.lr is None else 0.01,
                    num_steps=cfg.num_steps if cfg.num_steps is None else 3000,
                )
            else:
                raise ValueError("Only slsqp or sgd is supported for this experiment!")

            # Plot boundaries optionally
            if cfg.plot_boundaries:
                plot_boundaries(
                    weights,
                    train_samples,
                    train_labels,
                    show=cfg.show_fig,
                    save=cfg.save_fig,
                    save_name=f"boundaries_{losses_text[i]}.png",
                )

            # evaluate on train
            loss, acc = evaluate_linear(
                train_samples, train_labels, weights, loss_fns[i]
            )
            train_losses[i].append(loss)
            train_accs[i].append(acc)

            #  evaluate on test
            loss, acc = evaluate_linear(test_samples, test_labels, weights, loss_fns[i])
            test_losses[i].append(loss)
            test_accs[i].append(acc)

    #  Results summary
    logger.info(f"Summary of {cfg.n_repeat} trials:\n")
    for i in range(3):
        logger.info(losses_text[i])
        logger.info(
            f"Train Loss: {np.mean(train_losses[i])} +- {np.var(train_losses[i])}"
        )
        logger.info(f"Train Acc: {np.mean(train_accs[i])} +-  {np.var(train_accs[i])}")
        logger.info(f"Test Loss: {np.mean(test_losses[i])} +- {np.var(test_losses[i])}")
        logger.info(f"Test Acc:  {np.mean(test_accs[i])} +- {np.var(test_accs[i])}")

    # Display boxplot from paper
    long_servedio_boxplot(test_accs, losses_text, show=cfg.show_fig, save=cfg.save_fig)


def outliers_experiment(cfg: DictConfig):
    """
    Outliers synthetic experiment from Ding (2013)

    Args:
        cfg: Hydra config
    """

    inlier_feats, inlier_labels, outlier_feats, outlier_labels = outlier_dataset(
        cfg.seed
    )

    all_feats = np.concatenate([inlier_feats, outlier_feats])
    all_labels = np.concatenate([inlier_labels, outlier_labels])

    thetas = np.arange(-2.0, 2.0, 0.02)

    losses = defaultdict(lambda: [])

    for i in thetas:
        # Inliers
        losses["logistic_inliers"].append(
            np.mean(logistic_loss(inlier_labels * inlier_feats * i))
        )

        losses["huber_inliers"].append(
            np.mean(huberised_loss(inlier_labels * inlier_feats * i))
        )

        losses["phuber_inliers"].append(
            np.mean(partially_huberised_loss(inlier_labels * inlier_feats * i))
        )

        # Inliers + outliers
        losses["logistic_all"].append(
            np.mean(logistic_loss(all_labels * all_feats * i))
        )
        losses["huber_all"].append(np.mean(huberised_loss(all_labels * all_feats * i)))

        losses["phuber_all"].append(
            np.mean(partially_huberised_loss(all_labels * all_feats * i))
        )

    outliers_lineplot(thetas, losses, show=cfg.show_fig, save=cfg.save_fig)
