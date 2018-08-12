import scipy

__author__ = 'edwardharry'


def constructTrackMatFromCoords(coords):
    nFrames = len(coords)
    isSingle = type(coords[0]) is not list

    if isSingle:

        try:
            coords = scipy.concatenate(coords, axis=1)

        except:
            print("cannot track coords with missing timepoints")
            return None

        nTracks, nDim = coords.shape
        nDim /= nFrames

        trackMat = scipy.nan * scipy.ones((nTracks, nFrames * 8))

        for iDim in range(int(nDim)):
            trackMat[:, iDim:trackMat.shape[1]:8] = coords[:, iDim:coords.shape[1]:nDim]

    else:
        tmp = scipy.concatenate([coords[iFrame][0] for iFrame in range(nFrames)], axis=0)
        nTracks1 = tmp.shape[0]
        tmp = scipy.concatenate([coords[iFrame][1] for iFrame in range(nFrames)], axis=0)
        nTracks2 = tmp.shape[0]

        trackMat1 = scipy.nan * scipy.ones((nTracks1, nFrames * 8))
        trackMat2 = scipy.nan * scipy.ones((nTracks2, nFrames * 8))

        idx1 = 0
        idx2 = 0

        for iFrame in range(nFrames):
            coords1 = coords[iFrame][0]
            coords2 = coords[iFrame][1]
            nC1 = coords1.shape[0]
            nC2 = coords2.shape[0]
            trackMat1[idx1:(idx1 + nC1), (iFrame * 8):((iFrame * 8) + 2)] = coords1
            trackMat2[idx2:(idx2 + nC2), (iFrame * 8):((iFrame * 8) + 2)] = coords2
            idx1 += nC1
            idx2 += nC2

        trackMat = (trackMat1, trackMat2)

    return trackMat
