import math

import numpy as np

from synthetic.utils import inverse_sigmoid, sigmoid

"""
Calculates losses and gradient of losses used in "Synthetic datasets"
experiments (Section 5) from,
`"Can gradient clipping mitigate label noise?"
<https://openreview.net/pdf?id=rklB76EKPr>`_.

All of the losses below defined for a binary classification task with +1, -1
labels. We expect model logits to be multiplied with labels beforehand.
"""


def logistic_loss(z: np.ndarray) -> np.ndarray:
    """Returns the logistic loss for given scores.

    Shape:
        - Input: the raw, unnormalized prediction scores (model logits * labels).
                numpy array of size :math:`(minibatch)`
        - Output: the loss values for prediction scores
                numpy array of size :math:`(minibatch)`
    """
    z = np.clip(z, -500, 500)  # to avoid numerical errors
    return np.log(1 + np.exp(-z))


def huberized_loss(z: np.ndarray, tau: float = 0.1) -> np.ndarray:
    """Returns the partially huberized logistic loss for given scores.

    Shape:
        - Input: the raw, unnormalized prediction scores (model logits * labels).
                numpy array of size :math:`(minibatch)`
        - Output: the loss values for prediction scores
                numpy array of size :math:`(minibatch)`
    """
    return np.where(
        z <= -inverse_sigmoid(tau),  # linearization boundary
        -tau * z - math.log(1 - tau) - tau * inverse_sigmoid(tau),  # huber
        logistic_loss(z),  # default logistic loss
    )


def partially_huberized_loss(z: np.ndarray, tau: float = 2.0) -> np.ndarray:
    """Returns the partially huberized logistic loss for given scores.

    Shape:
        - Input: the raw, unnormalized prediction scores (model logits * labels).
                numpy array of size :math:`(minibatch)`
        - Output: the loss values for prediction scores
                numpy array of size :math:`(minibatch)`
    """
    return np.where(
        z <= inverse_sigmoid(1 / tau),  # linearization boundary
        -tau * sigmoid(z) + math.log(tau) + 1,  #  partial huber
        logistic_loss(z),  # default: logistic loss
    )


def logistic_gradient(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Returns the gradient of logistic loss for a linear scorer.

    Shape:
        - x: samples to be considered for gradient.
                numpy array of size :math:`(minibatch, W)`,
                with W the number of features
        - y: labels of the given samples (+1 or -1).
                numpy array of size :math:`(minibatch)`
        - w: weights of linear scorer.
                numpy array of size :math:`(W)`
        - Output: the gradients for each sample and weight
                numpy array of size :math:`(minibatch, W)`
    """
    z = y * (x @ w)  # get model scores
    z = sigmoid(-z)  #  1 - sigmoid(z)
    y, z = y[:, np.newaxis], z[:, np.newaxis]
    return (-z * y) * x  #  -z from d(logistic)/d(scores), y * x from d(scores)/d(w)


def huberized_gradient(
    x: np.ndarray, y: np.ndarray, w: np.ndarray, tau: float = 1
) -> np.ndarray:
    """Returns the gradient of huberized loss for a linear scorer.

    Shape:
        - x: samples to be considered for gradient.
                numpy array of size :math:`(minibatch, W)`,
                with W the number of features
        - y: labels of the given samples (+1 or -1).
                numpy array of size :math:`(minibatch)`
        - w: weights of linear scorer.
                numpy array of size :math:`(W)`
        - Output: the gradients for each sample and weight
                numpy array of size :math:`(minibatch, W)`
    """
    z = y * (x @ w)  # get model scores
    z = sigmoid(-z)  #  1 - sigmoid(z)
    y, z = y[:, np.newaxis], z[:, np.newaxis]
    return np.where(
        z >= tau,  # linearization boundary
        (-tau * y) * x,  #  huber gradient
        (-z * y) * x,  # logistic gradient
    )


def partially_huberized_gradient(
    x: np.ndarray, y: np.ndarray, w: np.ndarray, tau: float = 2
) -> np.ndarray:
    """Returns the gradient of partially huberized loss for a linear scorer.

    Shape:
        - x: samples to be considered for gradient.
                numpy array of size :math:`(minibatch, W)`,
                with W the number of features
        - y: labels of the given samples (+1 or -1).
                numpy array of size :math:`(minibatch)`
        - w: weights of linear scorer.
                numpy array of size :math:`(W)`
        - Output: the gradients for each sample and weight
                numpy array of size :math:`(minibatch, W)`
    """
    z = y * (x @ w)  # get model scores
    z = sigmoid(z)
    y, z = y[:, np.newaxis], z[:, np.newaxis]  # to broadcast
    return np.where(
        z <= 1.0 / tau,  # linearization boundary
        (-tau * z * (1 - z) * y) * x,  # partial huber gradient
        (-(1 - z) * y) * x,  # logistic gradient (1-z) to get sigmoid(-z)
    )
