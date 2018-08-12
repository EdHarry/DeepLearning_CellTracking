import lapjv

import scipy
import scipy.sparse


__author__ = 'edwardharry'


def lap(mat, nonlinkMarker=-1, augment=False):
    if scipy.sparse.issparse(mat):
        tmp = scipy.ones((mat.shape[0], mat.shape[1])) * nonlinkMarker
        select = scipy.nonzero(mat)
        mat = mat.toarray()
        tmp[select] = mat[select]
        mat = tmp

    select = mat == nonlinkMarker
    mat[select] = -scipy.inf
    maxCost = scipy.amax(mat.flatten())
    mat[select] = 1.2 * maxCost

    ret = lapjv.lap(mat, augment, maxCost * 1.1)

    return ret[1:]
