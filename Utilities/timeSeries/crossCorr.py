import scipy

__author__ = 'edwardharry'


def crossCorr(traj1, traj2, maxLag: int):
    if type(traj1) is scipy.ndarray:
        traj1 = [traj1]

    if type(traj2) is scipy.ndarray:
        traj2 = [traj2]

    numTraj1 = len(traj1)
    trajLength1 = [len(obs) for obs in traj1]
    numTraj2 = len(traj2)
    trajLength2 = [len(obs) for obs in traj2]

    if scipy.any((scipy.concatenate(([numTraj1], trajLength1)) - scipy.concatenate(([numTraj2], trajLength2))) != 0):
        print("Both trajectories must have the same number of data points.")
        return [], True

    trajLength1 = scipy.array(trajLength1)

    if scipy.sum(trajLength1[trajLength1 > maxLag] - maxLag) < 3 * maxLag:
        print("Trajectories not long enough for chosen maxLag.")
        return [], True

    traj1 = [obs - scipy.nanmean(obs) for obs in traj1]
    traj2 = [obs - scipy.nanmean(obs) for obs in traj2]

    traj1Std = scipy.sqrt(scipy.nanmean(scipy.concatenate(traj1) ** 2))
    traj2Std = scipy.sqrt(scipy.nanmean(scipy.concatenate(traj2) ** 2))

    gamma = scipy.zeros(((2 * maxLag) + 1, 2))

    for lag in range(-maxLag, 0):
        vecMult = scipy.array([])

        for j in range(numTraj1):

            if trajLength1[j] > scipy.absolute(lag):
                vec1 = traj1[j][scipy.absolute(lag):]
                vec2 = traj2[j][:-scipy.absolute(lag)]

                vecMult = scipy.concatenate((vecMult, vec1 * vec2))

        vecMult = vecMult[~scipy.isnan(vecMult)] / traj1Std / traj2Std
        gamma[lag + maxLag, :] = scipy.array(
            [scipy.mean(vecMult), scipy.std(vecMult, ddof=1) / scipy.sqrt(len(vecMult))])

    for lag in range(maxLag + 1):
        vecMult = scipy.array([])

        for j in range(numTraj1):

            if trajLength1[j] > lag:
                vec1 = traj1[j][:len(traj1[j]) - lag]
                vec2 = traj2[j][lag:]

                vecMult = scipy.concatenate((vecMult, vec1 * vec2))

        vecMult = vecMult[~scipy.isnan(vecMult)] / traj1Std / traj2Std
        gamma[lag + maxLag, :] = scipy.array(
            [scipy.mean(vecMult), scipy.std(vecMult, ddof=1) / scipy.sqrt(len(vecMult))])

    return gamma, False



