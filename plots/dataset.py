from typing import Optional, Tuple

import numpy as np


def long_servedio(
    N: int = 1000,
    gamma: float = 1.0 / 24.0,
    var: float = 0.01,
    corrupt_prob: float = 0.45,
    noise_seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generates 2-dimensional 6-atoms Long & Servedio dataset, from`
    "Random Classification Noise Defeats All Convex Potential Boosters"
    <http://www.cs.columbia.edu/~rocco/Public/icml08-cameraready.pdf>`_

    Args:
        N: number of samples
        gamma: specifies location of each atom
        var: variance of each atom
        corrupt_prob: applied label noise
        noise_seed: seed for applying label noise

    Returns:
        samples: numpy array of size :math:`(N, 2)`
        labels: numpy array of size :math:`(N)` with :math:`\pm1` labels
    """

    if noise_seed is not None:
        np.random.seed(noise_seed)

    # means of the atoms (following Long & Servedio)
    means = np.array(
        [
            [1, 0],
            [-1, 0],
            [gamma, 5 * gamma],
            [-gamma, -5 * gamma],
            [gamma, -gamma],
            [-gamma, gamma],
        ]
    )
    # each atom is isotropic gaussian
    cov = var * np.eye(2)

    # weights of the atoms (following Long & Servedio)
    weights = np.array([1.0, 1.0, 1.0, 1.0, 2.0, 2.0]) / 8.0

    samples, labels = [], []
    for i in range(N):
        # choose the atom
        m = means[np.random.choice(np.arange(len(means)), p=weights)]

        # sample from the atom
        x = np.random.multivariate_normal(m, cov)

        #  extract label x > 0
        label = 1 if x[0] >= 0 else -1

        # randomly flip with corrupt probability
        flip = np.random.choice([-1, 1], p=[1 - corrupt_prob, corrupt_prob])
        label = label * flip

        # store sample and label
        samples.append(x)
        labels.append(label)

    return np.array(samples), np.array(labels)
