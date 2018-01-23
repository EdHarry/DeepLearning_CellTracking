import scipy
import scipy.sparse

import Utilities.general.ismember as isM


__author__ = 'edwardharry'


class Compound2GraphReturnObject(object):
    def __init__(self, trackGraph, subTrackIdx, startNodes, endNodes, splitNodes, mergeNodes):
        self.trackGraph = trackGraph
        self.subTrackIdx = subTrackIdx
        self.startNodes = startNodes
        self.endNodes = endNodes
        self.splitNodes = splitNodes
        self.mergeNodes = mergeNodes


def compound2Graph(seqOfEvents=scipy.array([[1]])):
    n = seqOfEvents.shape
    n = n[0]
    trackGraph = scipy.sparse.lil_matrix((n, n), dtype=scipy.bool_)
    subTrackIdx = list([] for i in range(0, n))
    startNodes = []
    endNodes = []
    splitNodes = []
    mergeNodes = []

    for iNode in range(0, n):
        myTime = seqOfEvents[iNode, 0]
        typeOfEvent = seqOfEvents[iNode, 1]
        myIdx = seqOfEvents[iNode, 2]
        targetIdx = seqOfEvents[iNode, 3]

        subTrackIdx[iNode].append(myIdx)

        if scipy.isnan(targetIdx):
            if typeOfEvent == 1:
                startNodes.append(iNode + 1)
            else:
                endNodes.append(iNode + 1)
        else:
            if typeOfEvent == 1:
                splitNodes.append(iNode + 1)
                idx = 0
                loop = True
                while loop:
                    idx -= 1
                    jNode = iNode + idx
                    if scipy.array(isM.ismember([targetIdx], subTrackIdx[jNode])).astype(scipy.bool_) and myTime != \
                            seqOfEvents[jNode, 0]:
                        loop = False
                        subTrackIdx[jNode].append(myIdx)
                        trackGraph[jNode, iNode] = True
            else:
                mergeNodes.append(iNode + 1)
                subTrackIdx[iNode].append(targetIdx)
                idx = 0
                loop = True
                while loop:
                    idx -= 1
                    jNode = iNode + idx
                    if scipy.array(isM.ismember([myIdx], subTrackIdx[jNode])).astype(scipy.bool_) and myTime != \
                            seqOfEvents[jNode, 0]:
                        loop = False
                        trackGraph[jNode, iNode] = True

    for subIdx in scipy.unique(seqOfEvents[:, 2]):
        idx = -1
        loop = True
        while loop:
            idx += 1
            if scipy.array(isM.ismember([subIdx], subTrackIdx[idx])).astype(scipy.bool_):
                loop = False

        loop = True
        while loop:
            jdx = idx
            loop2 = True
            while loop2:
                jdx += 1
                if jdx >= n or scipy.array(isM.ismember([subIdx], subTrackIdx[jdx])).astype(scipy.bool_):
                    loop2 = False
            if jdx >= n:
                loop = False
            else:
                trackGraph[idx, jdx] = True
                idx = jdx

    return Compound2GraphReturnObject(trackGraph, subTrackIdx, startNodes, endNodes, splitNodes, mergeNodes)



