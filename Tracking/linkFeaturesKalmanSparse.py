import scipy
import scipy.sparse
import scipy.misc
from scipy.spatial import distance

from Utilities.linearAssesment.lap import lap

from Utilities.general.ConsoleOutput import ConsolePrintSingleLine

__author__ = 'edwardharry'


class LinkFeaturesKalmanSparseReturnValue(object):
    def __init__(self, trackedFeatureIndx, trackedFeatureInfo, kalmanFilterInfo, nnDistFeatures, prevCost, errFlag):
        self.trackedFeatureIndx = trackedFeatureIndx
        self.trackedFeatureInfo = trackedFeatureInfo
        self.kalmanFilterInfo = kalmanFilterInfo
        self.nnDistFeatures = nnDistFeatures
        self.prevCost = prevCost
        self.errFlag = errFlag


class PrevCost(object):
    def __init__(self, all, max, allAux):
        self.all = all
        self.max = max
        self.allAux = allAux


# noinspection PyNoneFunctionAssignment
def linkFeaturesKalmanSparse(movieInfo, costMatName, costMatParam, kalmanFunctions=None, probDim=None,
                             filterInfoPrev=None, prevCost=None):
    trackedFeatureIndx = []
    trackedFeatureInfo = []
    kalmanFilterInfo = []
    nnDistFeatures = []
    errFlag = []

    if kalmanFunctions is None or len(kalmanFunctions) == 0:
        kalmanFunctions = []
        selfAdaptive = False
    else:
        selfAdaptive = True

    if hasattr(movieInfo[0], 'zCoord'):
        probDimT = 3
    else:
        probDimT = 2

    if probDim is None:
        probDim = probDimT
    elif probDim == 3 and probDimT == 2:
        print("--linkFeaturesKalmanSparse: Inconsistency in input. Problem 3D but no z-coordinates.")
        return LinkFeaturesKalmanSparseReturnValue(trackedFeatureIndx, trackedFeatureInfo, kalmanFilterInfo,
                                                   nnDistFeatures, prevCost, True)

    if filterInfoPrev is None or len(filterInfoPrev) == 0:
        filterInfoPrev = []
        usePriorInfo = False
    else:
        usePriorInfo = True

    numFrames = len(movieInfo)

    if not hasattr(movieInfo[0], 'num'):
        for iFrame in range(numFrames):
            movieInfo[iFrame].num = movieInfo[iFrame].xCoord.shape[0]

    if not hasattr(movieInfo[0], 'allCoord'):
        if probDim == 2:
            for iFrame in range(numFrames):
                movieInfo[iFrame].allCoords = scipy.concatenate(([movieInfo[iFrame].xCoord, movieInfo[iFrame].yCoord]),
                                                                axis=1)
        elif probDim == 3:
            for iFrame in range(numFrames):
                movieInfo[iFrame].allCoords = scipy.concatenate(
                    ([movieInfo[iFrame].xCoord, movieInfo[iFrame].yCoord, movieInfo[iFrame].zCoord]), axis=1)

    if not hasattr(movieInfo[0], 'nnDist'):
        for iFrame in range(numFrames):
            if movieInfo[iFrame].num == 0:
                nnDist = scipy.zeros((0, 1))
            elif movieInfo[iFrame].num == 1:
                nnDist = scipy.array([[1000000000]])
            else:
                nnDist = distance.squareform(distance.pdist(movieInfo[iFrame].allCoord[:, 0::2]))
                nnDist = scipy.sort(nnDist, axis=1)
                nnDist = nnDist[:, 1]

            movieInfo[iFrame].nnDist = nnDist

    numFeatures = scipy.array([i.num for i in movieInfo])

    if selfAdaptive:
        kalmanFilterInfo = kalmanFunctions.reserveMem(numFrames, numFeatures, probDim)

    else:

        kalmanFilterInfo = scipy.zeros((numFrames, 1))

    trackedFeatureIndx = scipy.arange(1, movieInfo[0].num + 1).astype(scipy.int32)
    trackedFeatureIndx = scipy.reshape(trackedFeatureIndx, (trackedFeatureIndx.shape[0], 1))

    nnDistFeatures = movieInfo[0].nnDist

    if selfAdaptive:

        if usePriorInfo:
            kalmanFilterInfo[0].stateVec = filterInfoPrev[0].stateVec
            kalmanFilterInfo[0].stateCov = filterInfoPrev[0].stateCov
            kalmanFilterInfo[0].noiseVar = filterInfoPrev[0].noiseVar
        else:
            retVal = kalmanFunctions.initialise(movieInfo[0], probDim, costMatParam)
            filterInit = retVal.kalmanFilterInfo
            errFlag = retVal.errFlag

            kalmanFilterInfo[0].stateVec = filterInit.stateVec
            kalmanFilterInfo[0].stateCov = filterInit.stateCov
            kalmanFilterInfo[0].noiseVar = filterInit.noiseVar

    if prevCost is None or len(prevCost) == 0:
        prevCost = scipy.nan * scipy.ones((movieInfo[0].num, 1))
    else:
        prevCost = scipy.amax(prevCost.flatten()) * scipy.ones((movieInfo[0].num, 1))

    prevCostStruct = PrevCost(prevCost, scipy.array([scipy.amax(prevCost.flatten())]), scipy.array([]))

    featLifetime = scipy.ones((movieInfo[0].num, 1))

    numTracksWorstCase = scipy.round_(numFeatures.sum() / 10).astype(scipy.int64)

    trackedFeatureIndxAux = scipy.zeros((numTracksWorstCase, numFrames))
    nnDistFeaturesAux = scipy.nan * scipy.ones((numTracksWorstCase, numFrames))
    prevCostAux = nnDistFeaturesAux.copy()
    rowEnd = numTracksWorstCase

    ConsolePrintSingleLine("Linking frame to frame...")

    for iFrame in range(numFrames - 1):

        numFeaturesFrame1 = movieInfo[iFrame].num
        numFeaturesFrame2 = movieInfo[iFrame + 1].num

        if numFeaturesFrame1 != 0:

            if numFeaturesFrame2 != 0:

                retVal = costMatName(movieInfo, kalmanFilterInfo[iFrame], costMatParam,
                                     nnDistFeatures[:numFeaturesFrame1, :], probDim, prevCostStruct, featLifetime,
                                     trackedFeatureIndx, iFrame)
                costMat = retVal.costMat
                propagationScheme = retVal.propagationScheme
                kalmanFilterInfoTmp = retVal.kalmanFilterInfoFrame2
                nonlinkMarker = retVal.nonlinkMarker
                errFlag = retVal.errFlag
                del retVal

                if scipy.any(costMat.flatten() != nonlinkMarker):

                    link21 = lap(costMat, nonlinkMarker)
                    link21 = link21[1]
                    indx2C = scipy.nonzero(scipy.logical_and(link21[:numFeaturesFrame2] != -1,
                                                             link21[:numFeaturesFrame2] < numFeaturesFrame1))[0]
                    indx1C = link21[indx2C]

                    numExistTracks = trackedFeatureIndx.shape[0]
                    indx1U = scipy.setdiff1d(scipy.arange(0, numExistTracks), indx1C)
                    numRows = len(indx1U)

                    rowStart = rowEnd - numRows + 1

                    while rowStart <= 1:
                        trackedFeatureIndxAux = scipy.concatenate(
                            (scipy.zeros((numTracksWorstCase, numFrames)), trackedFeatureIndxAux), axis=0)
                        nnDistFeaturesAux = scipy.concatenate(
                            (scipy.nan * scipy.ones((numTracksWorstCase, numFrames)), nnDistFeaturesAux), axis=0)
                        prevCostAux = scipy.concatenate(
                            (scipy.nan * scipy.ones((numTracksWorstCase, numFrames)), prevCostAux), axis=0)
                        rowEnd = rowEnd + numTracksWorstCase
                        rowStart = rowStart + numTracksWorstCase

                    trackedFeatureIndxAux[rowStart - 1:rowEnd, :iFrame + 1] = trackedFeatureIndx[indx1U, :]
                    tmp = scipy.zeros((numFeaturesFrame2, iFrame + 2))
                    tmp[:numFeaturesFrame2, iFrame + 1] = scipy.arange(1, numFeaturesFrame2 + 1).transpose()
                    tmp[indx2C, :iFrame + 1] = trackedFeatureIndx[indx1C, :]
                    trackedFeatureIndx = tmp.copy()

                    nnDistFeaturesAux[rowStart - 1:rowEnd, :iFrame + 1] = nnDistFeatures[indx1U, :]
                    tmp = scipy.nan * scipy.ones((numFeaturesFrame2, iFrame + 2))
                    tmp[:numFeaturesFrame2, iFrame + 1] = movieInfo[iFrame + 1].nnDist[:, 0]
                    tmp[indx2C, :iFrame + 1] = nnDistFeatures[indx1C, :]
                    nnDistFeatures = tmp.copy()

                    prevCostAux[rowStart - 1:rowEnd, :iFrame + 1] = prevCost[indx1U, :]
                    tmp = scipy.nan * scipy.ones((numFeaturesFrame2, iFrame + 2))

                    for i in range(len(indx2C)):
                        tmp[indx2C[i], iFrame + 1] = costMat[indx1C[i], indx2C[i]]

                    tmp[indx2C, :iFrame + 1] = prevCost[indx1C, :]
                    prevCost = tmp.copy()

                    rowEnd = rowStart - 1
                    featLifetime = scipy.ones((numFeaturesFrame2, 1))

                    for iFeat in range(numFeaturesFrame2):
                        featLifetime[iFeat] = len(scipy.nonzero(trackedFeatureIndx[iFeat, :] != 0)[0])

                    if selfAdaptive:

                        if usePriorInfo:

                            retVal = kalmanFunctions.calcGain(trackedFeatureIndx[:numFeaturesFrame2, :],
                                                              movieInfo[iFrame + 1], kalmanFilterInfoTmp,
                                                              propagationScheme, kalmanFilterInfo, probDim,
                                                              filterInfoPrev[iFrame + 1], costMatParam,
                                                              kalmanFunctions.initialise)
                        else:

                            retVal = kalmanFunctions.calcGain(trackedFeatureIndx[:numFeaturesFrame2, :],
                                                              movieInfo[iFrame + 1], kalmanFilterInfoTmp,
                                                              propagationScheme, kalmanFilterInfo, probDim, [],
                                                              costMatParam, kalmanFunctions.initialise)

                        kalmanFilterInfo = retVal.kalmanFilterInfo
                        errFlag = retVal.errFlag
                        del retVal


                else:

                    numRows = trackedFeatureIndx.shape[0]
                    rowStart = rowEnd - numRows + 1

                    while rowStart <= 1:
                        trackedFeatureIndxAux = scipy.concatenate(
                            (scipy.zeros((numTracksWorstCase, numFrames)), trackedFeatureIndxAux), axis=0)
                        nnDistFeaturesAux = scipy.concatenate(
                            (scipy.nan * scipy.ones((numTracksWorstCase, numFrames)), nnDistFeaturesAux), axis=0)
                        prevCostAux = scipy.concatenate(
                            (scipy.nan * scipy.ones((numTracksWorstCase, numFrames)), prevCostAux), axis=0)
                        rowEnd = rowEnd + numTracksWorstCase
                        rowStart = rowStart + numTracksWorstCase

                    trackedFeatureIndxAux[rowStart - 1:rowEnd, :iFrame + 1] = trackedFeatureIndx
                    trackedFeatureIndx = scipy.zeros((numFeaturesFrame2, iFrame + 2))
                    trackedFeatureIndx[:numFeaturesFrame2, iFrame + 1] = scipy.arange(1,
                                                                                      numFeaturesFrame2 + 1).transpose()

                    nnDistFeaturesAux[rowStart - 1:rowEnd, :iFrame + 1] = nnDistFeatures
                    nnDistFeatures = scipy.nan * scipy.ones((numFeaturesFrame2, iFrame + 2))
                    nnDistFeatures[:numFeaturesFrame2, iFrame + 1] = movieInfo[iFrame + 1].nnDist

                    prevCostAux[rowStart - 1:rowEnd, :iFrame + 1] = prevCost
                    prevCost = scipy.nan * scipy.ones((numFeaturesFrame2, iFrame + 2))

                    rowEnd = rowStart - 1

                    featLifetime = scipy.ones((numFeaturesFrame2, 1))

                    if selfAdaptive:

                        if usePriorInfo:
                            kalmanFilterInfo[iFrame + 1].stateVec = filterInfoPrev[iFrame + 1].stateVec
                            kalmanFilterInfo[iFrame + 1].stateCov = filterInfoPrev[iFrame + 1].stateCov
                            kalmanFilterInfo[iFrame + 1].noiseVar = filterInfoPrev[iFrame + 1].noiseVar
                        else:
                            retVal = kalmanFunctions.initialise(movieInfo[iFrame + 1], probDim, costMatParam)
                            filterInit = retVal.kalmanFilterInfo
                            errFlag = retVal.errFlag

                            kalmanFilterInfo[iFrame + 1].stateVec = filterInit.stateVec
                            kalmanFilterInfo[iFrame + 1].stateCov = filterInit.stateCov
                            kalmanFilterInfo[iFrame + 1].noiseVar = filterInit.noiseVar

            else:

                numRows = trackedFeatureIndx.shape[0]
                rowStart = rowEnd - numRows + 1

                while rowStart <= 1:
                    trackedFeatureIndxAux = scipy.concatenate(
                        (scipy.zeros((numTracksWorstCase, numFrames)), trackedFeatureIndxAux), axis=0)
                    nnDistFeaturesAux = scipy.concatenate(
                        (scipy.nan * scipy.ones((numTracksWorstCase, numFrames)), nnDistFeaturesAux), axis=0)
                    prevCostAux = scipy.concatenate(
                        (scipy.nan * scipy.ones((numTracksWorstCase, numFrames)), prevCostAux), axis=0)
                    rowEnd = rowEnd + numTracksWorstCase
                    rowStart = rowStart + numTracksWorstCase

                trackedFeatureIndxAux[rowStart - 1:rowEnd, :iFrame + 1] = trackedFeatureIndx
                trackedFeatureIndx = scipy.zeros((numFeaturesFrame2, iFrame + 2))
                nnDistFeaturesAux[rowStart - 1:rowEnd, :iFrame + 1] = nnDistFeatures
                nnDistFeatures = scipy.nan * scipy.ones((numFeaturesFrame2, iFrame + 2))

                prevCostAux[rowStart - 1:rowEnd, :iFrame + 1] = prevCost
                prevCost = scipy.nan * scipy.ones((numFeaturesFrame2, iFrame + 2))

                rowEnd = rowStart - 1
                featLifetime = []

        else:

            if numFeaturesFrame2 != 0:

                trackedFeatureIndx = scipy.zeros((numFeaturesFrame2, iFrame + 2))
                trackedFeatureIndx[:numFeaturesFrame2, iFrame + 1] = scipy.arange(1,
                                                                                  numFeaturesFrame2 + 1).transpose().astype(
                    scipy.int64)
                nnDistFeatures = scipy.nan * scipy.ones((numFeaturesFrame2, iFrame + 2))
                nnDistFeatures[:numFeaturesFrame2, iFrame + 1] = movieInfo[iFrame + 1].nnDist.flatten()

                prevCost = scipy.nan * scipy.ones((numFeaturesFrame2, iFrame + 2))
                featLifetime = scipy.ones((numFeaturesFrame2, 1))

                if selfAdaptive:

                    if usePriorInfo:
                        kalmanFilterInfo[iFrame + 1].stateVec = filterInfoPrev[iFrame + 1].stateVec
                        kalmanFilterInfo[iFrame + 1].stateCov = filterInfoPrev[iFrame + 1].stateCov
                        kalmanFilterInfo[iFrame + 1].noiseVar = filterInfoPrev[iFrame + 1].noiseVar
                    else:
                        retVal = kalmanFunctions.initialise(movieInfo[iFrame + 1], probDim, costMatParam)
                        filterInit = retVal.kalmanFilterInfo
                        errFlag = retVal.errFlag

                        kalmanFilterInfo[iFrame + 1].stateVec = filterInit.stateVec
                        kalmanFilterInfo[iFrame + 1].stateCov = filterInit.stateCov
                        kalmanFilterInfo[iFrame + 1].noiseVar = filterInit.noiseVar

            else:

                trackedFeatureIndx = scipy.zeros((numFeaturesFrame2, iFrame + 2))
                nnDistFeatures = scipy.nan * scipy.ones((numFeaturesFrame2, iFrame + 2))
                prevCost = scipy.nan * scipy.ones((numFeaturesFrame2, iFrame + 2))
                featLifetime = []

        prevCostStruct.all = prevCost
        prevCostStruct.max = scipy.nanmax(
            scipy.concatenate(([prevCostStruct.max], scipy.reshape(prevCost[:, -1], (prevCost.shape[0], 1))), axis=0),
            axis=0)
        prevCostStruct.allAux = prevCostAux

        ConsolePrintSingleLine("Linking frame to frame (" + str((iFrame + 1) * 100 / numFrames) + "%)")

    numRows = trackedFeatureIndx.shape[0]
    rowStart = rowEnd - numRows + 1

    while rowStart <= 1:
        trackedFeatureIndxAux = scipy.concatenate((scipy.zeros((numRows, numFrames)), trackedFeatureIndxAux), axis=0)
        nnDistFeaturesAux = scipy.concatenate((scipy.nan * scipy.ones((numRows, numFrames)), nnDistFeaturesAux), axis=0)
        prevCostAux = scipy.concatenate((scipy.nan * scipy.ones((numRows, numFrames)), prevCostAux), axis=0)
        rowEnd = rowEnd + numRows
        rowStart = rowStart + numRows

    trackedFeatureIndxAux[rowStart - 1:rowEnd, :] = trackedFeatureIndx
    nnDistFeaturesAux[rowStart - 1:rowEnd, :] = nnDistFeatures
    prevCostAux[rowStart - 1:rowEnd, :] = prevCost

    trackedFeatureIndx = trackedFeatureIndxAux[rowStart - 1:, :].astype(scipy.int32)
    del trackedFeatureIndxAux
    nnDistFeatures = nnDistFeaturesAux[rowStart - 1:, :]
    del nnDistFeaturesAux
    prevCost = prevCostAux[rowStart - 1:, :]
    del prevCostAux

    numTracks = trackedFeatureIndx.shape[0]

    frameStart = scipy.zeros(numTracks)

    for i in range(numTracks):
        frameStart[i] = scipy.nonzero(trackedFeatureIndx[i, :] != 0)[0][0]
    indx = scipy.argsort(frameStart)

    trackedFeatureIndx = trackedFeatureIndx[indx, :]
    nnDistFeatures = nnDistFeatures[indx, :]
    prevCost = prevCost[indx, :]

    trackedFeatureInfo = scipy.sparse.lil_matrix((numTracks, 8 * numFrames))

    if probDim == 2:

        for iFrame in range(numFrames):

            indx1 = scipy.nonzero(trackedFeatureIndx[:, iFrame] != 0)[0]

            if len(indx1) > 0:
                indx2 = scipy.floor(trackedFeatureIndx[indx1, iFrame]).astype(scipy.int64) - 1

                trackedFeatureInfo[indx1, 8 * iFrame:8 * (iFrame + 1)] = scipy.concatenate((movieInfo[iFrame].allCoord[
                                                                                            indx2, :2 * probDim:2],
                                                                                            scipy.zeros(
                                                                                                (len(indx2), 1)),
                                                                                            scipy.reshape(
                                                                                                movieInfo[iFrame].amp[
                                                                                                    indx2, 0],
                                                                                                (len(indx2), 1)),
                                                                                            movieInfo[iFrame].allCoord[
                                                                                            indx2, 1:2 * probDim:2],
                                                                                            scipy.zeros(
                                                                                                (len(indx2), 1)),
                                                                                            scipy.reshape(
                                                                                                movieInfo[iFrame].amp[
                                                                                                    indx2, 1],
                                                                                                (len(indx2), 1))),
                                                                                           axis=1)

    else:

        for iFrame in range(numFrames):

            indx1 = scipy.nonzero(trackedFeatureIndx[:, iFrame] != 0)[0]

            if len(indx1) > 0:
                indx2 = scipy.floor(trackedFeatureIndx[indx1, iFrame]).astype(scipy.int64) - 1

                trackedFeatureInfo[indx1, 8 * iFrame:8 * (iFrame + 1)] = scipy.concatenate((movieInfo[iFrame].allCoord[
                                                                                            indx2, :2 * probDim:2],
                                                                                            scipy.reshape(
                                                                                                movieInfo[iFrame].amp[
                                                                                                    indx2, 0],
                                                                                                (len(indx2), 1)),
                                                                                            movieInfo[iFrame].allCoord[
                                                                                            indx2, 1:2 * probDim:2],
                                                                                            scipy.reshape(
                                                                                                movieInfo[iFrame].amp[
                                                                                                    indx2, 1],
                                                                                                (len(indx2), 1))),
                                                                                           axis=1)

    if selfAdaptive and not usePriorInfo:

        for iFrame in range(numFrames):
            kalmanFilterInfo[iFrame].noiseVar = scipy.absolute(kalmanFilterInfo[iFrame].noiseVar)

    return LinkFeaturesKalmanSparseReturnValue(trackedFeatureIndx, trackedFeatureInfo, kalmanFilterInfo, nnDistFeatures,
                                               prevCost, errFlag)
