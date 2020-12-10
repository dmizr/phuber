import math
import numpy as np

from plots.utils import sigmoid, inverse_sigmoid

def logistic_loss(z: np.ndarray) -> np.ndarray:
    """Returns the loss for logistic loss
    
    Args:
        z: logits

    Returns:
        logistic loss
    """
    z = np.clip(z, -500, 500)
    return np.log(1 + np.exp(-z))

def huberized_loss(z: np.ndarray, tau: float = 0.1) -> np.ndarray:
    """Returns the loss for partially huberized logistic loss
    
    Args:
        z: logits
        tau: huberization hyperparam

    Returns:
        huberized logistic loss
    """
    return np.where(
        z <= -inverse_sigmoid(tau),
        -tau * z - math.log(1 - tau) - tau * inverse_sigmoid(tau),
        logistic_loss(z))
        
def partially_huberized_loss(z: np.ndarray, tau: float = 2.0) -> np.ndarray:
    """Returns the loss for partially huberized logistic loss
    
    Args:
        z: logits
        tau: partial huberization hyperparam

    Returns:
        partially huberized logistic loss
    """
    return np.where(
        z <= inverse_sigmoid(1 / tau),
        -tau * sigmoid(z) + math.log(tau) + 1,
        logistic_loss(z))

def logistic_gradient(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Returns the gradient of logistic loss w.r.t weights
    
    Args:
        x: samples
        y: labels
        w: linear model weights

    Returns:
        dw: gradient of weights
    """
    z = y * (x @ w)
    z = sigmoid(-z) # 1 - sigmoid(z)
    y, z = y[:, np.newaxis], z[:, np.newaxis]
    return (-z * y) * x
    
def huberized_gradient(x: np.ndarray, y: np.ndarray, w: np.ndarray, tau: float = 1) -> np.ndarray:
    """Returns the gradient of huberized loss w.r.t weights
    
    Args:
        x: samples
        y: labels
        w: linear model weights

    Returns:
        dw: gradient of weights
    """
    z = y * (x @ w)
    z = sigmoid(-z) # 1 - sigmoid(z)
    y, z = y[:, np.newaxis], z[:, np.newaxis]
    return np.where(
        z >= tau,
        (-tau * y) * x,
        (-z * y) * x)
    
def partially_huberized_gradient(x: np.ndarray, y: np.ndarray, w: np.ndarray, tau: float = 2) -> np.ndarray:
    """Returns the gradient of partially huberized loss w.r.t weights
    
    Args:
        x: samples
        y: labels
        w: linear model weights

    Returns:
        dw: gradient of weights
    """
    z = y * (x @ w)
    z = sigmoid(z)
    y, z = y[:, np.newaxis], z[:, np.newaxis]
    return np.where(
        z <= 1. / tau,
        (-tau * z * (1 - z) * y) * x,
        (-(1 - z) * y) * x)
