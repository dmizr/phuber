import numpy as np
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
from plots.dataset import long_servedio, show_data
from plots.linear import train_linear, evaluate_linear


if __name__ == "__main__":
    n_repeat = 100
    losses_text = ['Logistic', 'Huberised', 'Partial Huberised']
    loss_fncs = [logistic_loss, huberized_loss, partially_huberized_loss]
    gradient_fncs = [logistic_gradient, huberized_gradient, partially_huberized_gradient]
    train_accs, train_lss = [ [] for i in range(3) ], [ [] for i in range(3) ]
    test_accs, test_lss = [ [] for i in range(3) ], [ [] for i in range(3) ]

    for n in range(n_repeat):
        # generate train and test sets
        train_samples, train_labels = long_servedio(N=1000, corrupt_prob=0.2, gamma=1./24., var=0.0, noise_seed=None)
        test_samples, test_labels = long_servedio(N=500, corrupt_prob=0, gamma=1./24., var=0.0, noise_seed=None)

        for i in range(3):
            # train linear model
            weights = train_linear(
                train_samples,
                train_labels,
                loss_fncs[i])

            # evaluate on train
            loss, acc = evaluate_linear(
                train_samples,
                train_labels,
                weights,
                loss_fncs[i])
            train_lss[i].append(loss)
            train_accs[i].append(acc)
            
            # evaluate on test
            loss, acc = evaluate_linear(
                test_samples,
                test_labels,
                weights,
                loss_fncs[i])
            test_lss[i].append(loss)
            test_accs[i].append(acc)

    print(f'Summary of {n_repeat} trials:\n')
    for i in range(3):
        print(losses_text[i])
        print('Train Loss: ', np.mean(train_lss[i]), '+-', np.var(train_lss[i]))
        print('Train Acc: ', np.mean(train_accs[i]), '+-', np.var(train_accs[i]))
        print('Test Loss: ', np.mean(test_lss[i]), '+-', np.var(test_lss[i]))
        print('Test Acc: ', np.mean(test_accs[i]), '+-', np.var(test_accs[i]))
        print()

    ax = sns.boxplot(data=test_accs)
    ax.set_xticklabels(losses_text, rotation=8)
    plt.show()

