import scipy
import scipy.sparse

import Utilities.graphTracks.floodFillGraph as FloodFill


__author__ = 'edwardharry'


def findTrackConflicts(idxMat=scipy.sparse.lil_matrix((1, 1))):
    nTracks = idxMat.shape[0]
    nFrames = idxMat.shape[1]

    confMat = scipy.sparse.lil_matrix((nTracks, nTracks), dtype=scipy.bool_)

    for iFrame in range(nFrames):

        col = idxMat[:, iFrame]
        col_nZ = col[scipy.nonzero(col[:, 0])[0], 0].toarray()[:, 0]
        col_unique = scipy.unique(col_nZ)

        if len(col_unique) != len(col_nZ):
            for idx in col_unique:

                ids = idx == col

                if ids.sum() > 1:

                    ids = scipy.nonzero(ids)[0]
                    nIds = len(ids)

                    for i in range(nIds - 1):
                        for j in range(i + 1, nIds):
                            confMat[ids[i], ids[j]] = True
                            confMat[ids[j], ids[i]] = True

    conflicts = FloodFill.floodFillGraph(confMat)

    idx = -1

    while idx < (len(conflicts) - 1):
        idx += 1

        if len(conflicts[idx]) == 1:
            conflicts.pop(idx)
            idx -= 1

    return conflicts