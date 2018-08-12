import scipy
import scipy.linalg

__author__ = 'edwardharry'


class KalmanGainLinearMotionReturnValue(object):
    def __init__(self, kalmanFilterInfo, errFlag):
        self.kalmanFilterInfo = kalmanFilterInfo
        self.errFlag = errFlag


class FeatureInfo(object):
    def __init__(self, allCoord, num):
        self.allCoord = allCoord
        self.num = num


def kalmanGainLinearMotion(trackedFeatureIndx, frameInfo, kalmanFilterInfoTmp, propagationScheme, kalmanFilterInfoIn,
                           probDim, filterInfoPrev, costMatParam, initFunctionName):
    errFlag = False

    if len(filterInfoPrev) == 0:
        usePriorInfo = False
    else:
        usePriorInfo = True

    for iFrame in range(len(kalmanFilterInfoIn)):
        kalmanFilterInfoIn[iFrame].noiseVar = scipy.absolute(kalmanFilterInfoIn[iFrame].noiseVar)

    trackedFeatureIndx = trackedFeatureIndx.astype(scipy.int32)

    kalmanFilterInfoOut = kalmanFilterInfoIn

    numFeatures = trackedFeatureIndx.shape[0]
    iFrame = trackedFeatureIndx.shape[1] - 1

    observationMat = scipy.mat(scipy.concatenate((scipy.eye(probDim), scipy.zeros((probDim, probDim))), axis=1))

    for iFeature in range(numFeatures):

        iFeaturePrev = trackedFeatureIndx[iFeature, -2]

        if iFeaturePrev != 0:

            iFeaturePrev -= 1

            iScheme = propagationScheme[iFeaturePrev, iFeature]
            kalmanFilterInfoOut[iFrame].scheme[iFeature, 0] = iScheme
            kalmanFilterInfoOut[iFrame - 1].scheme[iFeaturePrev, 1] = iScheme

            stateVecOld = scipy.mat(kalmanFilterInfoTmp.stateVec[iFeaturePrev, :, iScheme]).transpose()
            stateCovOld = scipy.mat(kalmanFilterInfoTmp.stateCov[:, :, iFeaturePrev, iScheme])
            obsVecOld = scipy.mat(kalmanFilterInfoTmp.obsVec[iFeaturePrev, :, iScheme]).transpose()

            kalmanGain = scipy.mat(scipy.linalg.solve((
                                                          observationMat * stateCovOld * observationMat.transpose() + scipy.diag(
                                                              scipy.spacing(1) + frameInfo.allCoord[iFeature,
                                                                                 1::2] ** 2)).transpose(),
                                                      (
                                                      stateCovOld * observationMat.transpose()).transpose())).transpose()

            stateNoise = kalmanGain * (scipy.reshape(frameInfo.allCoord[iFeature, ::2],
                                                     (frameInfo.allCoord[iFeature, ::2].shape[0], 1)) - obsVecOld)
            kalmanFilterInfoOut[iFrame - 1].stateNoise[iFeaturePrev, :] = scipy.array(stateNoise.transpose())[0, :]

            stateVec = stateVecOld + stateNoise

            stateCov = stateCovOld - (kalmanGain * observationMat * stateCovOld)

            indx = trackedFeatureIndx[iFeature, :-1]

            indxLength = len(scipy.nonzero(indx)[0])

            stateNoiseAll = scipy.zeros((indxLength, 2 * probDim))
            j = -1
            for i in range(iFrame - indxLength, iFrame):
                j += 1
                stateNoiseAll[j, :] = kalmanFilterInfoOut[i].stateNoise[indx[i] - 1, :]

            stateNoisePos = stateNoiseAll[:, :probDim]
            stateNoiseVel = stateNoiseAll[:, probDim:(2 * probDim)]

            noiseVar = scipy.zeros((2 * probDim))
            noiseVar[:probDim] = scipy.var(stateNoisePos.flatten(), ddof=1)
            noiseVar[probDim:(2 * probDim)] = scipy.var(stateNoiseVel.flatten(), ddof=1)

            kalmanFilterInfoOut[iFrame].stateVec[iFeature, :] = scipy.array(stateVec.transpose())[0, :]
            kalmanFilterInfoOut[iFrame].stateCov[:, :, iFeature] = scipy.array(stateCov)
            kalmanFilterInfoOut[iFrame].noiseVar[:, :, iFeature] = scipy.diag(noiseVar)

        else:

            if usePriorInfo:
                kalmanFilterInfoOut[iFrame].stateVec[iFeature, :] = filterInfoPrev.stateVec[iFeature, :]
                kalmanFilterInfoOut[iFrame].stateCov[:, :, iFeature] = filterInfoPrev.stateCov[:, :, iFeature]
                kalmanFilterInfoOut[iFrame].noiseVar[:, :, iFeature] = filterInfoPrev.noiseVar[:, :, iFeature]
            else:
                allCoord = frameInfo.allCoord[iFeature, :]
                allCoord = scipy.reshape(allCoord, (1, allCoord.shape[0]))
                featureInfo = FeatureInfo(allCoord, 1)
                retVal = initFunctionName(featureInfo, probDim, costMatParam)
                filterTmp = retVal.kalmanFilterInfo
                errFlag = retVal.errFlag

                kalmanFilterInfoOut[iFrame].stateVec[iFeature, :] = filterTmp.stateVec
                kalmanFilterInfoOut[iFrame].stateCov[:, :, iFeature] = filterTmp.stateCov[:, :, 0]
                kalmanFilterInfoOut[iFrame].noiseVar[:, :, iFeature] = filterTmp.noiseVar[:, :, 0]

        tmpState = kalmanFilterInfoOut[iFrame].stateVec[iFeature, :]
        tmpState[:probDim] = frameInfo.allCoord[iFeature, ::2]
        kalmanFilterInfoOut[iFrame].stateVec[iFeature, :] = tmpState

    return KalmanGainLinearMotionReturnValue(kalmanFilterInfoOut, errFlag)
