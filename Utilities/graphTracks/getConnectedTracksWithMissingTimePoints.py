import scipy
from scipy.sparse import coo_matrix

from Utilities.graphTracks.floodFillGraph import floodFillGraph


__author__ = 'edwardharry'


def getConnectedTracksWithMissingTimePoints(idxMat):
    ret = scipy.nonzero(idxMat == 0)
    tracksToProcess = ret[0]
    framesToProcess = ret[1]
    del ret
    nNodes = len(tracksToProcess)

    if nNodes == 0:
        return []

    i = []
    j = []

    for iNode in range(nNodes - 1):
        myTrackId = tracksToProcess[iNode]
        myFrameId = framesToProcess[iNode]
        jNode = iNode
        loop = True

        while jNode < (nNodes - 1) and loop:
            jNode += 1

            if myTrackId == tracksToProcess[jNode]:
                i.append(iNode)
                i.append(jNode)
                j.append(jNode)
                j.append(iNode)
                loop = False

        jNode = iNode
        loop = True

        while jNode < (nNodes - 1) and loop:
            jNode += 1

            if myFrameId == framesToProcess[jNode]:
                i.append(iNode)
                i.append(jNode)
                j.append(jNode)
                j.append(iNode)
                loop = False

    islands = floodFillGraph(coo_matrix((scipy.ones(len(i)), (i, j)), shape=(nNodes, nNodes)).tolil())

    return [scipy.unique(tracksToProcess[iIsland]) for iIsland in islands]


