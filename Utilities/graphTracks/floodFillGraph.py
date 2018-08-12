import scipy
import scipy.sparse

import Utilities.general.ismember as IsM

__author__ = 'edwardharry'


def floodFillGraph(c=scipy.sparse.lil_matrix((1, 1))):
    numS = c.shape[0]
    idx = [i for i in range(numS)]

    col = [0]
    loop = True

    while loop:
        colTemp = []

        for q in range(len(col)):
            connected = scipy.nonzero(c[:, col[q]])[0].tolist()
            colTemp += connected

        col2 = scipy.unique(col + colTemp).tolist()

        if len(col2) == len(col):
            loop = False

        col = col2

    subGraphs = [col]

    notIncluded = scipy.array(idx)[scipy.nonzero([i == 0 for i in IsM.ismember(idx, col)])[0]].tolist()

    while len(notIncluded) > 0:
        col = [notIncluded[0]]
        loop = True

        while loop:
            colTemp = []

            for q in range(len(col)):
                connected = scipy.nonzero(c[:, col[q]])[0].tolist()
                colTemp += connected

            col2 = scipy.unique(col + colTemp).tolist()

            if len(col2) == len(col):
                loop = False

            col = col2

        subGraphs.append(col)

        colAll = []

        for w in range(len(subGraphs)):
            colTemp = subGraphs[w]
            colAll += colTemp

        notIncluded = scipy.array(idx)[scipy.nonzero([i == 0 for i in IsM.ismember(idx, colAll)])[0]].tolist()

    return subGraphs
