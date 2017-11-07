import numpy as np


def normalize(v, axis=None):
    norm = np.linalg.norm(v, ord=1, axis=axis)
    if axis:
        norm = np.expand_dims(norm, axis=axis)
    return v/norm
