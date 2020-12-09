import numpy as np

def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1. / (1. + np.exp(-z))

def inverse_sigmoid(z: float) -> float:
    return np.log(z / (1 - z))