import numpy as np

from synthetic.utils import logit, sigmoid

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


def huberised_loss(z: np.ndarray, tau: float = 0.1) -> np.ndarray:
    """Returns the partially huberized logistic loss for given scores.

    Shape:
        - Input: the raw, unnormalized prediction scores (model logits * labels).
                numpy array of size :math:`(minibatch)`
        - Output: the loss values for prediction scores
                numpy array of size :math:`(minibatch)`
    """
    return np.where(
        # linearisation boundary
        z <= -logit(tau),
        # huber
        -tau * z - np.log(1 - tau) - tau * logit(tau),
        logistic_loss(z),
    )


def partially_huberised_loss(z: np.ndarray, tau: float = 1.1) -> np.ndarray:
    """Returns the partially huberized logistic loss for given scores.

    Shape:
        - Input: the raw, unnormalized prediction scores (model logits * labels).
                numpy array of size :math:`(minibatch)`
        - Output: the loss values for prediction scores
                numpy array of size :math:`(minibatch)`
    """
    return np.where(
        # linearization boundary
        z <= logit(1 / tau),
        #  partial huber
        -tau * sigmoid(z) + np.log(tau) + 1,
        logistic_loss(z),
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
    # Get model scores
    z = y * (x @ w)
    z = sigmoid(-z)
    y, z = y[:, np.newaxis], z[:, np.newaxis]
    #  -z from d(logistic)/d(scores), y * x from d(scores)/d(w)
    return (-z * y) * x


def huberised_gradient(
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
    # Get model scores
    z = y * (x @ w)
    z = sigmoid(-z)
    y, z = y[:, np.newaxis], z[:, np.newaxis]
    return np.where(
        # linearisation boundary
        z >= tau,
        # huber gradient
        (-tau * y) * x,
        # logistic gradient
        (-z * y) * x,
    )


def partially_huberised_gradient(
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
    # Get model scores
    z = y * (x @ w)
    z = sigmoid(z)
    y, z = y[:, np.newaxis], z[:, np.newaxis]
    return np.where(
        # linearisation boundary
        z <= 1.0 / tau,
        # partial huber gradient
        (-tau * z * (1 - z) * y) * x,
        # logistic gradient (1-z) to get sigmoid(-z)
        (-(1 - z) * y) * x,
    )


def empirical_risk_logistic_loss(labels, feats, theta):
    risk = np.mean(logistic_loss(labels * feats * theta))
    return risk
