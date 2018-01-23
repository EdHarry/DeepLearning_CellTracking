import scipy

from Utrack.kalmanFunctions.KalmanFilterInfo import KalmanFilterInfo


__author__ = 'edwardharry'


class KalmanInitLinearMotionReturnValue(object):
    def __init__(self, kalmanFilterInfo, errFlag):
        self.kalmanFilterInfo = kalmanFilterInfo
        self.errFlag = errFlag


def kalmanInitLinearMotion(frameInfo, probDim, costMatParam):
    kalmanFilterInfo = []
    errFlag = False

    minSearchRadius = costMatParam.minSearchRadius
    maxSearchRadius = costMatParam.maxSearchRadius
    brownStdMult = costMatParam.brownStdMult

    if hasattr(costMatParam, 'kalmanInitParam'):
        initParam = costMatParam.kalmanInitParam

        if hasattr(initParam, 'convergePoint'):
            convergePoint = initParam.convergePoint
        else:
            convergePoint = scipy.array([])

        if hasattr(initParam, 'initVelocity'):
            initVelGuess = initParam.initVelocity
        else:
            initVelGuess = scipy.array([])

        if hasattr(initParam, 'searchRadiusFirstIteration') and len(initParam.searchRadiusFirstIteration) > 0:
            searchRadiusFirstIteration = initParam.searchRadiusFirstIteration
            noiseVarInit = -(searchRadiusFirstIteration / brownStdMult) ** 2 / probDim
        else:
            noiseVarInit = (maxSearchRadius / brownStdMult) ** 2 / probDim

    else:
        convergePoint = scipy.array([])
        initVelGuess = scipy.array([])
        noiseVarInit = (scipy.mean((minSearchRadius, maxSearchRadius), axis=0) / brownStdMult) ** 2 / probDim

    if len(convergePoint) > 0:
        nRow = convergePoint.shape[0]
        nCol = convergePoint.shape[1]
        if nRow != 0 and nCol == 1:
            convergePoint = convergePoint.transpose()

        nCol = convergePoint.shape[1]
        if nCol != probDim:
            print('--kalmanInitLinearMotion: initParam.convergePoint of wrong dimension!')
            errFlag = True

    if len(initVelGuess) > 0:
        nRow = initVelGuess.shape[0]
        nCol = initVelGuess.shape[1]
        if nRow != 0 and nCol == 1:
            initVelGuess = initVelGuess.transpose()

        nCol = initVelGuess.shape[1]
        if nCol != probDim:
            print('--kalmanInitLinearMotion: initParam.initVelocity of wrong dimension!')
            errFlag = True

    if errFlag:
        print('--kalmanInitLinearMotion: Please fix input parameters.')
        return KalmanInitLinearMotionReturnValue(kalmanFilterInfo, True)

    numFeatures = frameInfo.num

    if len(initVelGuess) == 0:

        if len(convergePoint) == 0:

            velocityInit = scipy.zeros((numFeatures, probDim))

        else:

            speedInit = 1

            displacement = convergePoint - frameInfo.allCoord[:, ::2]
            distance = scipy.sqrt((displacement ** 2).sum(axis=1))

            velocityInit = speedInit * displacement / distance

    else:

        velocityInit = scipy.tile(initVelGuess, (numFeatures, 1))

    stateVec = scipy.concatenate((frameInfo.allCoord[:, ::2], velocityInit), axis=1)

    stateCov = scipy.zeros((2 * probDim, 2 * probDim, numFeatures))
    noiseVar = stateCov.copy()

    for iFeature in range(numFeatures):
        posVar = frameInfo.allCoord[iFeature, 1::2] ** 2
        posVar = scipy.maximum(posVar, scipy.spacing(1))
        stateCov[:, :, iFeature] = scipy.diag(scipy.concatenate((posVar, 4 * scipy.ones(probDim)), axis=0))
        noiseVar[:, :, iFeature] = scipy.diag(noiseVarInit * scipy.ones((2 * probDim)))

    return KalmanInitLinearMotionReturnValue(KalmanFilterInfo(stateVec, stateCov, noiseVar, None, None), errFlag)