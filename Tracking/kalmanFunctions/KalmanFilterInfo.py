__author__ = 'edwardharry'


class KalmanFilterInfo(object):
    def __init__(self, stateVec, stateCov, noiseVar, stateNoise, scheme):
        self.stateVec = stateVec
        self.stateCov = stateCov
        self.noiseVar = noiseVar
        self.stateNoise = stateNoise
        self.scheme = scheme

    def __len__(self):
        return 1


class KalmanFilterInfoFrame(object):
    def __init__(self, stateVec, stateCov, obsVec):
        self.stateVec = stateVec
        self.stateCov = stateCov
        self.obsVec = obsVec

    def __len__(self):
        return 1
