import scipy
from scipy.sparse import coo_matrix

from Utilities.graphTracks.floodFillGraph import floodFillGraph

__author__ = 'edwardharry'


class NetworkedFeaturesInfo(object):
    def __init__(self, networkedFeatures, indexToGroupMap):
        self.networkedFeatures = networkedFeatures
        self.indexToGroupMap = indexToGroupMap


def getNetworkedFeatures(trackMatFull):
    nTracks = trackMatFull.shape[0]
    nFrames = trackMatFull.shape[2]
    networkedFeatures = []
    indexToGroupMap = scipy.zeros((nTracks, nFrames))

    for iFrame in range(nFrames):
        feats = trackMatFull[:, :, iFrame].copy()
        anyZeros = scipy.any(feats == 0, axis=1)
        feats[anyZeros, :] = 0
        goodIdx = scipy.nonzero(~anyZeros)[0]
        feats_tmp = feats[goodIdx, :]
        nGoodIdx = len(goodIdx)
        nNodes = 2 * nGoodIdx

        if nNodes == 0:
            continue

        graphMatRows = scipy.arange(nNodes).tolist()
        graphMatCols = scipy.concatenate((scipy.arange(nGoodIdx, nNodes), scipy.arange(nGoodIdx))).tolist()

        for i in range(2):

            for idx in scipy.unique(feats_tmp[:, i]):
                ind = scipy.nonzero(feats_tmp[:, i] == idx)[0]
                nInd = len(ind)

                for iInd in range(nInd - 1):

                    for jInd in range(iInd + 1, nInd):
                        iNode = ind[scipy.array([iInd, jInd])] + (i * nGoodIdx)
                        graphMatRows.append(iNode[0])
                        graphMatRows.append(iNode[1])
                        graphMatCols.append(iNode[1])
                        graphMatCols.append(iNode[0])

        graphGroups = floodFillGraph(
            coo_matrix((scipy.ones(len(graphMatRows)), (graphMatRows, graphMatCols)), shape=(nNodes, nNodes)).tolil())
        nGroups = len(graphGroups)
        groups = []

        for iGroup in range(nGroups):
            idx = scipy.array(graphGroups[iGroup])
            idx = idx[idx < nGoodIdx]
            idx = goodIdx[idx]
            groups.append(feats[idx, :])
            feats[idx, 0] = iGroup + 1

        networkedFeatures.append(groups)
        indexToGroupMap[:, iFrame] = feats[:, 0]

    return NetworkedFeaturesInfo(networkedFeatures, indexToGroupMap)
