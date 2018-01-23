from scipy import int64
from scipy.ndimage.measurements import label, find_objects

__author__ = 'edwardharry'


def connComp(seq):
    seq = seq.astype(int64)
    n = label(seq, output=seq)
    idxs = find_objects(seq)
    slices = [s[0] for s in idxs]

    class ConnComp(object):
        def __init__(self, n, slices):
            self.n = n
            self.slices = slices

    return ConnComp(n, slices)
