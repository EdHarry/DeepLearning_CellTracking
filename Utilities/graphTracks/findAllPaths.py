import scipy
import scipy.sparse

import Utilities.graphTracks.allPaths as AllPaths
import Utilities.general.ismember as IsM

__author__ = 'edwardharry'


class FindAllPathsReturnValue(object):
    def __init__(self, pathsOutput, earlyExit):
        self.pathsOutput = pathsOutput
        self.earlyExit = earlyExit


def findAllPaths(graphMat=scipy.sparse.lil_matrix((1, 1)), startIdx=scipy.array([0]), endIdx=scipy.array([0])):
    def getPaths():
        paths = []

        p = AllPaths.allPaths(subGraphMat, scipy.array([subStartIdxs.tolist()]).transpose(), subEndIdxs, N)
        for i in range(0, len(p)):
            paths = paths + p[i].tolist()

        return paths

    def filterByStart():
        return [scipy.array(IsM.ismember([i[0]], startIdx)).astype(scipy.bool_).tolist()[0] for i in subPaths]

    def filterByEnd():
        return [scipy.array(IsM.ismember([i[-1]], endIdx)).astype(scipy.bool_).tolist()[0] for i in subPaths]

    maxGraphSize = 10
    maximumMatrixProcessingStats = scipy.array([0.2, 56])
    earlyExit = False

    nNodes = graphMat.shape
    nNodes = nNodes[0]

    if nNodes % 2 == 1:
        graphMat = scipy.sparse.hstack(
            [scipy.sparse.vstack([graphMat, scipy.sparse.lil_matrix((1, nNodes), dtype=scipy.bool_)]),
             scipy.sparse.lil_matrix((nNodes + 1, 1), dtype=scipy.bool_)], 'lil')
        nNodes += 1

    nNodes = int(nNodes / 2)
    subStartIdxs = scipy.arange(0, nNodes)
    subEndIdxs = scipy.arange(0, nNodes)

    if nNodes >= maximumMatrixProcessingStats[1] / 2 and graphMat.getnnz() / (nNodes * nNodes * 4) >= \
            maximumMatrixProcessingStats[0]:
        return FindAllPathsReturnValue([], True)

    paths = list(list([] for i in range(2)) for i in range(2))
    for i in range(2):
        for j in range(2):
            iIndex = range((i * nNodes), (i * nNodes) + nNodes)
            jIndex = range((j * nNodes), (j * nNodes) + nNodes)

            subGraphMat = graphMat[iIndex, :]
            subGraphMat = subGraphMat[:, jIndex]

            if i == j:
                if nNodes > maxGraphSize:
                    findAllPathsReturn = findAllPaths(subGraphMat, subStartIdxs, subEndIdxs)
                    extraPaths = findAllPathsReturn.pathsOutput
                    earlyExit = findAllPathsReturn.earlyExit

                    if earlyExit:
                        return FindAllPathsReturnValue([], True)

                else:
                    N = nNodes
                    extraPaths = getPaths()

                if i == 1:
                    extraPaths = [scipy.array(k).__add__(nNodes).tolist() for k in extraPaths]
            else:
                N = 2
                extraPaths = getPaths()
                diagPaths = scipy.nonzero(subGraphMat.diagonal())[0]
                if diagPaths.shape[0] > 0:
                    extraPaths += scipy.array([diagPaths, diagPaths]).transpose().tolist()
                if j == 0:
                    extraPaths = [[k[0] + nNodes, k[1]] for k in extraPaths]
                else:
                    extraPaths = [[k[0], k[1] + nNodes] for k in extraPaths]

            paths[i][j] = extraPaths

    if scipy.array([len(i) for i in (paths[0] + paths[1])]).sum() == 0:
        return FindAllPathsReturnValue([], earlyExit)

    subMatTransitionPaths = [[[] for i in range(4)] for i in range(4)]
    paths = paths[0] + paths[1]
    N = 2

    for i in range(4):
        for j in range(4):
            if i != j:
                endIdx1 = [k[-1] for k in paths[i]]
                startIdx2 = [k[0] for k in paths[j]]
                nEnd = len(endIdx1)
                nStart = len(startIdx2)
                subNnodes = nEnd + nStart

                idxAll = scipy.zeros((min(100000000, nEnd * nStart)), dtype=scipy.int64)
                idAll = idxAll.copy()

                k = 0
                for idx in range(0, nEnd):
                    k += 1
                    id = scipy.nonzero(IsM.ismember(startIdx2, [endIdx1[idx]]))[0].__add__(nEnd).tolist()
                    nId = len(id)
                    if nId > 0:
                        idxAll[(k - 1):(nId + k - 1)] = scipy.ones((nId, 1), dtype=scipy.int64).__mul__(idx + 1)[:,
                                                        0].tolist()
                        idAll[(k - 1):(nId + k - 1)] = id
                    k = nId + k - 1
                idxId = scipy.nonzero(idxAll)[0]
                idxAll = idxAll[idxId].__add__(-1)
                idAll = idAll[idxId]
                del idxId
                subGraphMat = scipy.sparse.coo_matrix(
                    (scipy.full((len(idxAll)), True, dtype=scipy.bool_), (idxAll, idAll)), shape=(subNnodes, subNnodes))
                subStartIdxs = scipy.arange(0, nEnd)
                subEndIdxs = scipy.arange(nEnd, subNnodes)
                subPaths = scipy.array(getPaths())
                del subGraphMat
                del idAll
                del idxAll

                if len(subPaths) > 0:
                    subPaths[:, 1] = subPaths[:, 1] - nEnd
                else:
                    subPaths = []
                subMatTransitionPaths[i][j] = subPaths

    subMat = scipy.array([[(len(i) > 0) for i in j] for j in subMatTransitionPaths])

    pathsOutput = []
    currentTrackList = []
    currentSelect = []
    subTrackIds = []
    for i in range(0, 4):
        subPaths = paths[i]
        select = scipy.nonzero(filterByStart())[0]
        subPaths = scipy.array(subPaths)[select].tolist()
        if len(subPaths) > 0:
            pathsOutput = pathsOutput + scipy.array(subPaths)[scipy.nonzero(filterByEnd())[0]].tolist()
            currentTrackList += [subPaths]
            subTrackIds += [i]
            currentSelect += [select]

    if len(subTrackIds) == 0:
        return FindAllPathsReturnValue([], earlyExit)

    N = 1
    subEndIdxs = scipy.arange(0, 4)
    subStartIdxs = scipy.array([subTrackIds]).transpose()
    subTrackIds = subStartIdxs.copy()
    loop = True
    while loop:
        N += 1
        subStartIdxs = AllPaths.allPaths(subMat, subStartIdxs, subEndIdxs, N, N - 1, True)
        if len(subStartIdxs) > 0:
            subStartIdxs = scipy.array(subStartIdxs)[0]
        pathsAddedThisN = False
        idx = -1
        newSelect = [0 for i in range(len(subStartIdxs))]
        newPaths = newSelect.copy()

        while idx < (len(subStartIdxs) - 1):
            idx += 1
            path = subStartIdxs[idx, :]
            node = path[-1]
            previousNode = path[-2]
            selectCurrentTracks = IsM.in1d_index(path[0:-1], subTrackIds)
            subPaths = scipy.array(currentTrackList)[selectCurrentTracks].tolist()[0]
            select = scipy.array(currentSelect)[selectCurrentTracks].tolist()[0]
            transition_ = subMatTransitionPaths[previousNode][node]
            nSelect = len(select)
            transition = scipy.zeros((min(100000000, nSelect * len(transition_)), 2), dtype=scipy.int64)
            id = -1

            for k in range(0, nSelect):
                id += 1
                tmp = transition_[transition_[:, 0] == select[k], 1]
                nTmp = len(tmp)
                tmp = scipy.array([[k for i in range(nTmp)], tmp.tolist()]).transpose()
                transition[id:(nTmp + id), :] = tmp.__add__(1)
                id = nTmp + id - 1

            transition = transition[scipy.nonzero(transition[:, 0]), :][0].__add__(-1)
            select = transition[:, 1]
            subPaths = [subPaths[transition[i, 0]][0:-1] + paths[node][transition[i, 1]] for i in
                        range(0, len(transition))]
            goodPaths = scipy.nonzero([len(scipy.unique(i)) == len(i) for i in subPaths])[0]
            subPaths = scipy.array(subPaths)[goodPaths].tolist()
            select = select[goodPaths]
            newSelect[idx] = select
            newPaths[idx] = subPaths

            if len(subPaths) == 0:
                subStartIdxs = scipy.delete(subStartIdxs, idx, 0)
                newSelect.pop(idx)
                newPaths.pop(idx)
                idx -= 1
            else:
                pathsAddedThisN = True
                pathsOutput = pathsOutput + scipy.array(subPaths)[scipy.nonzero(filterByEnd())[0]].tolist()

        if not pathsAddedThisN:
            loop = False
        else:
            subTrackIds = subStartIdxs
            currentTrackList = newPaths
            currentSelect = newSelect

    if len(pathsOutput) == 0:
        pathsOutput = []

    return FindAllPathsReturnValue(pathsOutput, earlyExit)
