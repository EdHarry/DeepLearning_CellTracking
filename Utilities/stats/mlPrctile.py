import scipy

__author__ = 'edwardharry'  # MATLAB's algorithm for percentiles


def percentile(x, p):
    if len(x) == 0:
        return scipy.nan

    if type(p) is not scipy.ndarray:
        p = scipy.array([p])
        isSingle = True
    else:
        isSingle = False

    y = scipy.zeros(p.shape)
    x = x.flatten()
    x = scipy.sort(x)
    n = x.shape[0]
    r = (p / 100) * n
    k = scipy.floor(r + 0.5).astype(scipy.int64)
    sk = scipy.nonzero(k < 1)[0]
    if sk.shape[0] > 0:
        y[sk] = x[0]
    lk = scipy.nonzero(k >= n)[0]
    if lk.shape[0] > 0:
        y[lk] = x[-1]
    mk = scipy.nonzero(scipy.logical_and(k >= 1, k < n))[0]
    if mk.shape[0] > 0:
        r[mk] = r[mk] - k[mk]
        y[mk] = (0.5 - r[mk]) * x[k[mk] - 1] + (0.5 + r[mk]) * x[k[mk]]

    if isSingle:
        y = y[0]

    return y


