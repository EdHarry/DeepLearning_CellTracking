import sys
import scipy
import scipy.sparse

from Utilities.general.ismember import ismember, in1d_index
from Utilities.general.uniqueRows import uniqueRows
from Utilities.graphTracks.findAllPaths import findAllPaths

__author__ = 'edwardharry'


class GenerateCojoinedTracksReturnValue:
    def __init__(self, extraTracks, earlyExit):
        self.extraTracks = extraTracks
        self.earlyExit = earlyExit


def generateCojoinedTracks(idxMat, extraString=""):
    nMaxNodes = 100
    extraTracks = idxMat.copy()
    nTracks = idxMat.shape[0]
    nFrames = idxMat.shape[1]

    framesToConsider = scipy.arange(1, nFrames - 1)
    kFrame = 2

    statusString = extraString + "Generating compound tracks, {0:.2f}%...".format(0)
    sys.stdout.write(statusString + '\r')
    sys.stdout.flush()

    while kFrame < nFrames:
        confs = None

        for jFrame in scipy.intersect1d(scipy.arange(kFrame - 1, nFrames - 1), framesToConsider):
            col = extraTracks[:, jFrame]
            col_nZ = col[scipy.logical_and(col != 0, ~scipy.isnan(col))]
            col_unique = scipy.unique(col_nZ)

            if len(col_unique) != len(col_nZ):

                for idx in col_unique:
                    ids = idx == col

                    if ids.sum() > 1:
                        ids = scipy.nonzero(ids)[0]
                        nIds = len(ids)

                        if confs is None:
                            confs = scipy.concatenate((scipy.ravel_multi_index((ids[:-1], ids[1:]), (nTracks, nTracks))[
                                                       :, scipy.newaxis], jFrame * scipy.ones((nIds - 1, 1))), axis=1)
                        else:
                            confs = scipy.concatenate((confs, scipy.concatenate((scipy.ravel_multi_index(
                                (ids[:-1], ids[1:]), (nTracks, nTracks))[:, scipy.newaxis],
                                                                                 jFrame * scipy.ones((nIds - 1, 1))),
                                                                                axis=1)))

        if confs is None:
            break

        confs = confs[confs[:, 0].argsort(), :]
        confsDiff = scipy.diff(confs, 1, 0)
        confsDiff[confsDiff[:, 0] != 0, 1] = 0
        confs[:-1, :][confsDiff[:, 1] == 1, :] = scipy.nan
        confs = confs[~scipy.isnan(confs[:, 0]), :]

        i = scipy.unravel_index(confs[:, 0].astype(scipy.int64), (nTracks, nTracks))
        confs = scipy.concatenate((i[0][:, scipy.newaxis], i[1][:, scipy.newaxis], confs[:, 1][:, scipy.newaxis]),
                                  axis=1)
        confs.view('i8,i8,i8').sort(order=['f2', 'f0', 'f1'], axis=0)

        nExtraNodes = 2 * nTracks
        framesToConsider = scipy.unique(confs[:, 2]).astype('int')

        loop = True
        jFrame = kFrame - 1
        previousSelect = None

        while loop:
            jFrame += 1
            select = scipy.array(ismember(confs[:, 2], scipy.arange(kFrame - 1, jFrame))) > 0
            nNodes = uniqueRows(scipy.concatenate((confs[select, :][:, [0, 2]], confs[select, :][:, [1, 2]]))).shape[0]

            if nNodes > nMaxNodes and scipy.any(select):
                loop = False

                if previousSelect is not None:
                    select = previousSelect.copy()
                    jFrame -= 1
                    nNodes = nNodesPrevious

            elif jFrame == nFrames - 1:
                loop = False

            elif scipy.any(select):
                previousSelect = select.copy()
                nNodesPrevious = nNodes

        nNodes += nExtraNodes
        confs = confs[select, :]
        kFrame = jFrame + 1

        i = []
        j = []
        trackMap = -scipy.ones((nNodes, 2)).astype('int')

        previousFrames = []
        idx = 0
        buildTransMat = False

        for iFrame in scipy.unique(confs[:, 2]):
            confs_sub = confs[confs[:, 2] == iFrame, :2]

            for trans in confs_sub:

                for trackId in trans:
                    trackIds = scipy.array([trackId, iFrame]).astype('int')

                    if in1d_index(trackMap, trackIds).shape[0] == 0:
                        idx += 1
                        trackMap[idx - 1, :] = trackIds

                thisFrameIdx = trackMap[:, 1] == iFrame
                iNode = scipy.nonzero(scipy.logical_and(trackMap[:, 0] == trans[0], thisFrameIdx))[0].tolist()
                jNode = scipy.nonzero(scipy.logical_and(trackMap[:, 0] == trans[1], thisFrameIdx))[0].tolist()
                i += iNode
                i += jNode
                j += jNode
                j += iNode

            if buildTransMat:

                for trackId in scipy.unique(confs_sub.flatten()):
                    thisFrameIdx = trackMap[:, 1] == iFrame
                    jNode = scipy.logical_and(trackMap[:, 0] == trackId, thisFrameIdx)

                    if scipy.any(jNode):
                        loop = True
                        idx2 = 0
                        n = len(previousFrames)

                        while idx2 < n and loop:
                            idx2 += 1
                            previousFrame = previousFrames[idx2 - 1]
                            previousFrameIdx = trackMap[:, 1] == previousFrame
                            iNode = scipy.logical_and(trackMap[:, 0] == trackId, previousFrameIdx)

                            if scipy.any(iNode):
                                loop = False
                                iNode = scipy.nonzero(iNode)[0].tolist()
                                jNode = scipy.nonzero(jNode)[0].tolist()
                                i += iNode
                                j += jNode

            else:
                buildTransMat = True

            previousFrames += [iFrame]

        trackMap[idx:idx + nTracks, :] = scipy.concatenate(
            (scipy.arange(nTracks)[:, scipy.newaxis], scipy.zeros((nTracks, 1))), axis=1)
        trackMap[idx + nTracks:, :] = scipy.concatenate(
            (scipy.arange(nTracks)[:, scipy.newaxis], (nFrames - 1) * scipy.ones((nTracks, 1))), axis=1)

        startTime = [0, nFrames - 1]
        dir = [1, -1]

        for ii in range(2):

            for iTrack in range(nTracks):
                thisTrackIdx = trackMap[:, 0] == iTrack
                iNode = scipy.logical_and(trackMap[:, 1] == startTime[ii], thisTrackIdx)
                linked = False
                idx2 = 0

                while not linked:
                    idx2 += 1
                    nextTP = startTime[ii] + (dir[ii] * idx2)
                    jNode = scipy.logical_and(trackMap[:, 1] == nextTP, thisTrackIdx)

                    if scipy.any(jNode):

                        if ii == 1:
                            tmp = iNode
                            iNode = jNode
                            jNode = tmp

                        iNode = scipy.nonzero(iNode)[0].tolist()
                        jNode = scipy.nonzero(jNode)[0].tolist()
                        i += iNode
                        j += jNode
                        linked = True

        retVal = findAllPaths(scipy.sparse.coo_matrix((scipy.ones(len(i)), (i, j)), shape=(nNodes, nNodes)).tolil(),
                              scipy.arange(idx, idx + nTracks), scipy.arange(idx + nTracks, nNodes))
        paths = retVal.pathsOutput
        earlyExit = retVal.earlyExit
        del retVal
        if earlyExit:
            return GenerateCojoinedTracksReturnValue([], True)

        nPaths = len(paths)
        extraPaths_ = scipy.zeros((nPaths, nFrames))

        for iPath in range(nPaths):
            path = paths[iPath]

            for node in path:
                map = trackMap[node, :]
                extraPaths_[iPath, map[1]:] = extraTracks[map[0], map[1]:]

        extraTracks = uniqueRows(extraPaths_)
        nTracks = extraTracks.shape[0]

        statusString = extraString + "Generating compound tracks, {0:.2f}%...".format(min((kFrame + 1), nFrames) * 100 / nFrames)
        sys.stdout.write(statusString + '\r')
        sys.stdout.flush()

    print(statusString)
    return GenerateCojoinedTracksReturnValue(
        extraTracks[
        scipy.setdiff1d(scipy.arange(extraTracks.shape[0]), in1d_index(idxMat, extraTracks.astype(scipy.int64))),
        :].astype(scipy.int64),
        False)
