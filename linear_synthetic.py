import hydra
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from omegaconf import DictConfig

from synthetic.dataset import long_servedio
from synthetic.linear import evaluate_linear, train_linear_sgd, train_linear_slsqp
from synthetic.loss import (
    huberized_gradient,
    huberized_loss,
    logistic_gradient,
    logistic_loss,
    partially_huberized_gradient,
    partially_huberized_loss,
)
from synthetic.utils import plot_boundaries


@hydra.main(config_path="conf", config_name="synthetic/linear")
def synthetic_linear(cfg: DictConfig) -> None:
    # prepare losses and gradients
    losses_text = ["Logistic", "Huberised", "Partial Huberised"]
    loss_fncs = [logistic_loss, huberized_loss, partially_huberized_loss]
    grad_fncs = [
        logistic_gradient,
        huberized_gradient,
        partially_huberized_gradient,
    ]

    # containers for results
    train_accs, train_lss = [[] for i in range(3)], [[] for i in range(3)]
    test_accs, test_lss = [[] for i in range(3)], [[] for i in range(3)]

    for n in range(cfg.n_repeat):
        #  generate train and test sets
        train_samples, train_labels = long_servedio(
            N=cfg.n_train,
            corrupt_prob=cfg.corrupt_prob,
            gamma=cfg.gamma,
            var=cfg.var,
            noise_seed=cfg.seed,
        )
        test_samples, test_labels = long_servedio(
            N=cfg.n_test,
            corrupt_prob=cfg.corrupt_prob,
            gamma=cfg.gamma,
            var=cfg.var,
            noise_seed=cfg.seed + 1 if cfg.seed else None,
        )

        #  iterate over losses
        for i in range(3):
            # train linear model
            if cfg.method == "slsqp":
                weights, _ = train_linear_slsqp(
                    samples=train_samples,
                    labels=train_labels,
                    loss_fn=loss_fncs[i]
                )
            elif cfg.method == "sgd":
                weights, _ = train_linear_sgd(
                    samples=train_samples,
                    labels=train_labels,
                    loss_fn=loss_fncs[i],
                    grad_fn=grad_fncs[i],
                    lr=cfg.lr or 0.1,
                    num_steps=cfg.num_steps or 3000,
                )
            else:
                raise ValueError("Only slsqp or sgd is supported for method!")

            # Plot boundaries optionally
            if cfg.plot:
                plot_boundaries(weights, train_samples, train_labels)

            # evaluate on train
            loss, acc = evaluate_linear(
                train_samples, train_labels, weights, loss_fncs[i]
            )
            train_lss[i].append(loss)
            train_accs[i].append(acc)

            #  evaluate on test
            loss, acc = evaluate_linear(
                test_samples, test_labels, weights, loss_fncs[i]
            )
            test_lss[i].append(loss)
            test_accs[i].append(acc)

    # log summary
    print(f"Summary of {cfg.n_repeat} trials:\n")
    for i in range(3):
        print(losses_text[i])
        print("Train Loss: ", np.mean(train_lss[i]), "+-", np.var(train_lss[i]))
        print("Train Acc: ", np.mean(train_accs[i]), "+-", np.var(train_accs[i]))
        print("Test Loss: ", np.mean(test_lss[i]), "+-", np.var(test_lss[i]))
        print("Test Acc: ", np.mean(test_accs[i]), "+-", np.var(test_accs[i]))
        print()

    # plot results
    ax = sns.boxplot(data=test_accs)
    ax.set_xticklabels(losses_text, rotation=8)
    plt.show()


if __name__ == "__main__":
    synthetic_linear()
