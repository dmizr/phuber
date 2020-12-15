from typing import Union

import numpy as np


def sigmoid(z: Union[np.ndarray, float]) -> np.ndarray:
    """Applies sigmoid function on given z."""

    # Clip to avoid numerical errors
    z = np.clip(z, -500, 500)
    out = 1.0 / (1.0 + np.exp(-z))
    return out


def inverse_sigmoid(z: Union[np.ndarray, float]) -> np.ndarray:
    """Applies inverse sigmoid function on given z."""
    return np.log(z / (1 - z))
