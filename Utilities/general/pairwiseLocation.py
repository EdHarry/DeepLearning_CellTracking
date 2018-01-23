import scipy
from scipy.spatial.distance import pdist
from scipy.misc import comb

__author__ = 'edwardharry'


def pairwiseLocation(coords: scipy.ndarray, cutoff):
    def fun(x, y):
        return scipy.nonzero(x > y)[0][0]

    vfun = scipy.vectorize(fun)
    vfun.excluded.add(0)

    n = coords.shape[0]
    dis = pdist(coords)
    selectionArray = scipy.cumsum(scipy.arange(n - 1, 0, -1))
    selection = scipy.nonzero(dis < cutoff)[0]

    if len(selection) == 0:
        return scipy.zeros((0, 2))

    i = vfun(selectionArray, selection)
    j = selection - comb(n, 2) + comb(n - i, 2) + i + 1

    return scipy.concatenate((i[:, scipy.newaxis], j[:, scipy.newaxis].astype(scipy.int64)), axis=1)
