from typing import Union

import numpy as np


def sigmoid(z: Union[np.ndarray, float]) -> np.ndarray:
    """Applies sigmoid function on given z."""

    # Clip to avoid numerical errors
    z = np.clip(z, -500, 500)
    out = 1.0 / (1.0 + np.exp(-z))
    return out


def logit(z: Union[np.ndarray, float]) -> np.ndarray:
    """Applies inverse sigmoid function (log-odds) on given z."""

    # Clip to avoid numerical errors
    z = np.clip(z, 1e-9, 1 - 1e-9)
    log_odds = np.log(z / (1 - z))
    return log_odds
