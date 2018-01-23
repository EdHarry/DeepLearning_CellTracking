import scipy

__author__ = 'edwardharry'  # http://stackoverflow.com/questions/15864082/python-equivalent-of-matlabs-ismember-function?rq=1


def ismember(a, b):
    bind = {}
    for i, elt in enumerate(b):
        if elt not in bind:
            bind[elt] = i
    return [(bind.get(itm, -1) + 1) for itm in a]  # None can be replaced by any other "not in b" value


def asvoid(arr):  # http://stackoverflow.com/questions/22699756/python-version-of-ismember-with-rows-and-index
    """
    View the array as dtype np.void (bytes)
    This views the last axis of ND-arrays as bytes so you can perform comparisons on
    the entire row.
    http://stackoverflow.com/a/16840350/190597 (Jaime, 2013-05)
    Warning: When using asvoid for comparison, note that float zeros may compare UNEQUALLY
    >>> asvoid([-0.]) == asvoid([0.])
    array([False], dtype=bool)
    """
    arr = scipy.ascontiguousarray(arr)
    return arr.view(scipy.dtype((scipy.void, arr.dtype.itemsize * arr.shape[-1])))


def in1d_index(a, b):
    voida, voidb = map(asvoid, (a, b))
    return scipy.where(scipy.in1d(voidb, voida))[0]


def in1d_index_map(a, b):
    idx = in1d_index(b, a)
    z = scipy.zeros(a.shape[0]).astype(scipy.int64) - 1
    for i in idx:
        z[i] = in1d_index(a[i, :], b)[0]
    return z