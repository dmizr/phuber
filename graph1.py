import seaborn as sns
import matplotlib.pyplot as plt

from plots.loss import (
    logistic_loss,
    huberized_loss,
    partially_huberized_loss,
    logistic_gradient,
    huberized_gradient,
    partially_huberized_gradient
)
from plots.dataset import long_servedio
from plots.linear import linear_regression, linear_evaluate


if __name__ == "__main__":
    n_repeat = 500
    losses_text = ['Logistic', 'Huberised', 'Partial Huberised']
    loss_fncs = [logistic_loss, huberized_loss, partially_huberized_loss]
    gradient_fncs = [logistic_gradient, huberized_gradient, partially_huberized_gradient]
    accuracies = [ [] for i in range(len(losses_text)) ]

    for _ in range(n_repeat):
        # generate train and test sets
        train_samples, train_labels = long_servedio(N=1000)
        test_samples, test_labels = long_servedio(N=500, corrupt_prob=0)

        for i in range(3):
            # train linear model
            weights = linear_regression(
                train_samples,
                train_labels,
                loss_fncs[i],
                gradient_fncs[i])
            
            # evaluate linear model
            acc = linear_evaluate(
                test_samples,
                test_labels,
                weights)
            accuracies[i].append(acc)

    ax = sns.boxplot(data=accuracies)
    ax.set_xticklabels(losses_text, rotation=8)
    plt.show()
