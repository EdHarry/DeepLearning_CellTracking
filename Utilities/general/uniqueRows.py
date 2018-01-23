__author__ = 'edwardharry'

import scipy


def uniqueRows(a: scipy.ndarray):
    return scipy.unique(scipy.ascontiguousarray(a).view(scipy.dtype((scipy.void, a.dtype.itemsize * a.shape[1])))).view(
        a.dtype).reshape(-1, a.shape[1])