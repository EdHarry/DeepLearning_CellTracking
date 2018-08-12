__author__ = 'edwardharry'


def kalmanReverseLinearMotion(kalmanFilterInfo, probDim):
    kalmanFilterInfo.reverse()

    for iFrame in range(len(kalmanFilterInfo)):
        kalmanFilterInfo[iFrame].stateVec[:, probDim:] = -kalmanFilterInfo[iFrame].stateVec[:, probDim:]

    return kalmanFilterInfo
