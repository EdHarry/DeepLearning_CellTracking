import warnings

import scipy
import scipy.sparse
from scipy.spatial import distance

from Utrack.kalmanFunctions.KalmanFilterInfo import KalmanFilterInfoFrame
from Utilities.stats import mlPrctile

__author__ = 'edwardharry'


class CostMatRandomDirectedSwitchingMotionLinkReturnValue(object):
    def __init__(self, costMat, propagationScheme, kalmanFilterInfoFrame2, nonlinkMarker, errFlag):
        self.costMat = costMat
        self.propagationScheme = propagationScheme
        self.kalmanFilterInfoFrame2 = kalmanFilterInfoFrame2
        self.nonlinkMarker = nonlinkMarker
        self.errFlag = errFlag


def costMatRandomDirectedSwitchingMotionLink(movieInfo, kalmanFilterInfoFrame1, costMatParam, nnDistFeatures, probDim,
                                             prevCost, featLifetime, trackedFeatureIndx, currentFrame):
    errFlag = False

    linearMotion = costMatParam.linearMotion
    minSearchRadius = costMatParam.minSearchRadius
    maxSearchRadius = costMatParam.maxSearchRadius
    brownStdMult = costMatParam.brownStdMult
    useLocalDensity = costMatParam.useLocalDensity
    nnWindow = costMatParam.nnWindow

    if useLocalDensity:
        closestDistScale = 2
        maxStdMult = 100

    if hasattr(costMatParam, 'lftCdf') and costMatParam.lftCdf is not None and len(costMatParam.lftCdf) > 0:
        lftCdf = costMatParam.lftCdf
    else:
        lftCdf = scipy.array([])

    frameNum = nnDistFeatures.shape[1]
    tmpNN = scipy.maximum(1, frameNum - nnWindow)
    nnDistTracks = scipy.nanmin(nnDistFeatures[:, tmpNN - 1:], axis=1)
    nnDistTracks = scipy.reshape(nnDistTracks, (nnDistTracks.shape[0], 1))

    movieInfo = movieInfo[currentFrame:currentFrame + 2]

    vecSize = 2 * probDim
    numSchemes = 1

    if linearMotion == 0:

        transMat = scipy.zeros((vecSize, vecSize, 1))
        transMat[:, :, 0] = scipy.eye(vecSize)
        numSchemes = 1

    elif linearMotion == 1:

        transMat = scipy.zeros((vecSize, vecSize, 2))
        transMat[:, :, 0] = scipy.eye(vecSize) + scipy.diag(scipy.ones(probDim), probDim)
        transMat[:, :, 1] = scipy.eye(vecSize)
        numSchemes = 2

    elif linearMotion == 2:

        transMat = scipy.zeros((vecSize, vecSize, 3))
        transMat[:, :, 0] = scipy.eye(vecSize) + scipy.diag(scipy.ones(probDim), probDim)
        transMat[:, :, 1] = scipy.eye(vecSize) + scipy.diag(-scipy.ones(probDim), probDim)
        transMat[:, :, 2] = scipy.eye(vecSize)
        numSchemes = 3

    observationMat = scipy.concatenate((scipy.eye(probDim), scipy.zeros((probDim, probDim))), axis=1)

    numFeaturesFrame1 = movieInfo[0].num
    numFeaturesFrame2 = movieInfo[1].num

    kalmanFilterInfoFrame2 = KalmanFilterInfoFrame(scipy.zeros((numFeaturesFrame1, vecSize, numSchemes)),
                                                   scipy.zeros((vecSize, vecSize, numFeaturesFrame1, numSchemes)),
                                                   scipy.zeros((numFeaturesFrame1, probDim, numSchemes)))

    for iFeature in range(numFeaturesFrame1):

        stateOld = scipy.mat(kalmanFilterInfoFrame1.stateVec[iFeature, :]).transpose()
        stateCovOld = scipy.mat(kalmanFilterInfoFrame1.stateCov[:, :, iFeature])
        noiseVar = scipy.mat(scipy.absolute(kalmanFilterInfoFrame1.noiseVar[:, :, iFeature]))

        for iScheme in range(numSchemes):
            stateVec = transMat[:, :, iScheme] * stateOld

            stateCov = transMat[:, :, iScheme] * stateCovOld * transMat[:, :, iScheme].transpose() + noiseVar

            obsVec = observationMat * stateVec

            kalmanFilterInfoFrame2.stateVec[iFeature, :, iScheme] = stateVec.transpose()
            kalmanFilterInfoFrame2.stateCov[:, :, iFeature, iScheme] = stateCov
            kalmanFilterInfoFrame2.obsVec[iFeature, :, iScheme] = obsVec.transpose()

    propagatedPos = kalmanFilterInfoFrame2.obsVec

    coord2 = movieInfo[1].allCoord[:, ::2]

    costMatTmp = scipy.zeros((propagatedPos.shape[0], coord2.shape[0], numSchemes))

    for iScheme in range(numSchemes):
        coord1 = propagatedPos[:, :, iScheme]
        costMatTmp[:, :, iScheme] = distance.cdist(coord1, coord2)

        if movieInfo[0].previousTrackFunc is not None:
            for index1 in range(coord1.shape[0]):
                _, afterIdx = movieInfo[0].PreviousTracks(currentFrame, index1)
                if afterIdx is not None:
                    for index2 in range(coord2.shape[0]):
                        if index2 != afterIdx:
                            costMatTmp[index1, index2, iScheme] = scipy.inf

            for index2 in range(coord2.shape[0]):
                beforeIdx, _ = movieInfo[0].PreviousTracks(currentFrame + 1, index2)
                if beforeIdx is not None and beforeIdx < 0:
                    for index1 in range(coord1.shape[0]):
                        costMatTmp[index1, index2, iScheme] = scipy.inf

    costMat = scipy.amin(costMatTmp, axis=2)
    propagationScheme = scipy.argmin(costMatTmp, axis=2)

    notFirstAppearance = kalmanFilterInfoFrame1.noiseVar[0, 0, :] >= 0
    notFirstAppearance = scipy.reshape(notFirstAppearance, (notFirstAppearance.shape[0], 1))

    kalmanStd = scipy.sqrt(probDim * scipy.absolute(kalmanFilterInfoFrame1.noiseVar[0, 0, :]))
    kalmanStd = scipy.reshape(kalmanStd, (kalmanStd.shape[0], 1))

    stdMultInd = scipy.tile(brownStdMult, (numFeaturesFrame1, 1))

    if useLocalDensity:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            ratioDist2Std = nnDistTracks / kalmanStd / closestDistScale
            ratioDist2Std[ratioDist2Std > maxStdMult] = maxStdMult

        stdMultInd = scipy.amax(scipy.concatenate((stdMultInd, ratioDist2Std), axis=1), axis=1)
        stdMultInd = scipy.reshape(stdMultInd, (stdMultInd.shape[0], 1))

    searchRadius = stdMultInd * kalmanStd

    searchRadius[scipy.logical_and((searchRadius > maxSearchRadius), notFirstAppearance)] = maxSearchRadius
    searchRadius[scipy.logical_and((searchRadius < minSearchRadius), notFirstAppearance)] = minSearchRadius

    searchRadius = scipy.tile(searchRadius, (1, numFeaturesFrame2))

    select = costMat > searchRadius

    costMat = scipy.power(costMat, 2)

    costMat[select] = scipy.nan

    if len(lftCdf) > 0:
        oneMinusLftCdf = 1 - lftCdf

        oneOverLftPen = oneMinusLftCdf[featLifetime]

        costMat /= scipy.tile(oneOverLftPen, (1, numFeaturesFrame2))

        costMat[scipy.isinf(costMat)] = scipy.nan

    maxCost = 1.05 * scipy.maximum(mlPrctile.percentile(costMat[~scipy.isnan(costMat)].flatten(), 100),
                                   scipy.spacing(1))

    if scipy.isnan(maxCost):
        maxCost = scipy.spacing(1)

    deathCost = maxCost * scipy.ones(numFeaturesFrame1)
    birthCost = maxCost * scipy.ones(numFeaturesFrame2)

    deathBlock = scipy.diag(deathCost)
    deathBlock[deathBlock == 0] = scipy.nan
    birthBlock = scipy.diag(birthCost)
    birthBlock[birthBlock == 0] = scipy.nan

    lrBlock = costMat.copy().transpose()
    lrBlock[~scipy.isnan(lrBlock)] = maxCost

    costMat = scipy.vstack((scipy.hstack((costMat, deathBlock)), scipy.hstack((birthBlock, lrBlock))))

    if scipy.all(scipy.isnan(costMat.flatten())):
        nonlinkMarker = -5
    else:
        nonlinkMarker = scipy.minimum(scipy.floor(scipy.amin(costMat[~scipy.isnan(costMat)].flatten())) - 5, -5)

    costMat[scipy.isnan(costMat)] = nonlinkMarker

    return CostMatRandomDirectedSwitchingMotionLinkReturnValue(costMat, propagationScheme, kalmanFilterInfoFrame2,
                                                               nonlinkMarker, errFlag)
