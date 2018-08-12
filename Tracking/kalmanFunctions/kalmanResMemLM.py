import scipy

from Utrack.kalmanFunctions.KalmanFilterInfo import KalmanFilterInfo

__author__ = 'edwardharry'


def kalmanResMemLM(numFrames, numFeatures, probDim):
    vecSize = 2 * probDim

    return [KalmanFilterInfo(scipy.zeros((numFeatures[iFrame], vecSize)),
                             scipy.zeros((vecSize, vecSize, numFeatures[iFrame])),
                             scipy.zeros((vecSize, vecSize, numFeatures[iFrame])),
                             scipy.zeros((numFeatures[iFrame], vecSize)), scipy.zeros((numFeatures[iFrame], 2))) for
            iFrame in range(numFrames)]
