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

    return np.log(1 + np.exp(-z))

def huberized_loss(z: np.ndarray, tau: float = 0.9) -> np.ndarray:
    """Returns the loss for partially huberized logistic loss
    
    Args:
        z: logits
        tau: huberization hyperparam

    Returns:
        huberized logistic loss
    """

    if (z <= -inverse_sigmoid(tau)):
        return -tau * z - math.log(1 - tau) - tau * inverse_sigmoid(tau)
    else :
        return logistic_loss(z)
        
def partially_huberized_loss(z: np.ndarray, tau: float = 2.0) -> np.ndarray:
    """Returns the loss for partially huberized logistic loss
    
    Args:
        z: logits
        tau: partial huberization hyperparam

    Returns:
        partially huberized logistic loss
    """

    if (z <= inverse_sigmoid(1 / tau)):
        return -tau * sigmoid(z) + math.log(tau) + 1
    else:
        return logistic_loss(z)

def logistic_gradient(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Returns the gradient of logistic loss w.r.t weights
    
    Args:
        x: samples
        y: labels
        w: linear model weights

    Returns:
        dw: gradient weight vector
    """
    z = y * (x @ w)
    z = sigmoid(-z) # 1 - sigmoid(z)
    return -z * y * x
    
def huberized_gradient(x: np.ndarray, y: np.ndarray, w: np.ndarray, tau: float = 0.9) -> np.ndarray:
    """Returns the gradient of huberized loss w.r.t weights
    
    Args:
        x: samples
        y: labels
        w: linear model weights

    Returns:
        dw: gradient weight vector
    """
    z = y * (x @ w)
    z = sigmoid(-z) # 1 - sigmoid(z)
    if z >= tau:
        return -tau * y * x
    else:
        return -z * y * x
    
def partially_huberized_gradient(x: np.ndarray, y: np.ndarray, w: np.ndarray, tau: float = 2) -> np.ndarray:
    """Returns the gradient of partially huberized loss w.r.t weights
    
    Args:
        x: samples
        y: labels
        w: linear model weights

    Returns:
        dw: gradient weight vector
    """
    z = y * (x @ w)
    z = sigmoid(z)
    if z <= 1. / tau:
        return -tau * z * (1 - z) * y * x
    else:
        return -(1 - z) * y * x
