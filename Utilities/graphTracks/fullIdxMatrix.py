__author__ = 'edwardharry'

import copy

import scipy

from Utrack.utilityClasses import TracksFinal

from Utrack.convStruct2MatIgnoreMS import convStruct2MatIgnoreMS
from Utilities.graphTracks.compound2Graph import compound2Graph
from Utilities.graphTracks.findAllPaths import findAllPaths
from Utilities.general.ismember import ismember, in1d_index
from Utilities.general.uniqueRows import uniqueRows


def fullIdxMatrix(track: TracksFinal, onlySubTrackStartToConsider=None):
    trackStartTime = track.seqOfEvents[0, 0]
    trackedFeatureInfo = convStruct2MatIgnoreMS([track])
    trackedFeatureInfo = trackedFeatureInfo[1]
    trackGraphInfo = compound2Graph(track.seqOfEvents)
    startNodes = trackGraphInfo.startNodes
    endNodes = trackGraphInfo.endNodes
    splitNodes = trackGraphInfo.splitNodes
    mergeNodes = trackGraphInfo.mergeNodes

    def splitTrackHere(splitTime, splitEvent):
        subTrack = copy.deepcopy(track)
        eventsToRemove = scipy.logical_and(subTrack.seqOfEvents[:, 0] <= splitTime, subTrack.seqOfEvents[:, 1] == 2)
        subTracksToRemove = subTrack.seqOfEvents[eventsToRemove, 2]
        eventsToRemove = scipy.logical_and(scipy.logical_or(eventsToRemove, subTrack.seqOfEvents[:, 1] == 1),
                                           scipy.array(ismember(subTrack.seqOfEvents[:, 2], subTracksToRemove)) > 0)
        eventsToConvToStarts = scipy.logical_and(subTrack.seqOfEvents[:, 0] <= splitTime,
                                                 subTrack.seqOfEvents[:, 1] == 1, ~(
                scipy.array(ismember(subTrack.seqOfEvents[:, 2], subTracksToRemove)) > 0))

        subTrack.seqOfEvents[eventsToConvToStarts, 0] = splitTime
        subTrack.seqOfEvents[eventsToConvToStarts, 3] = scipy.nan
        subTrack.seqOfEvents[splitEvent, 3] = scipy.nan
        subTrack.seqOfEvents = subTrack.seqOfEvents[~eventsToRemove, :]

        for subTrackToRemove in scipy.sort(subTracksToRemove)[::-1]:
            subTrack.seqOfEvents[subTrack.seqOfEvents[:, 2] > subTrackToRemove, 2] -= 1
            subTrack.seqOfEvents[subTrack.seqOfEvents[:, 3] > subTrackToRemove, 3] -= 1
            subTrack.tracksFeatIndxCG = scipy.delete(subTrack.tracksFeatIndxCG, subTrackToRemove - 1, axis=0)
            subTrack.tracksCoordAmpCG = scipy.delete(subTrack.tracksCoordAmpCG, subTrackToRemove - 1, axis=0)

        timesToRemove = scipy.arange(splitTime - trackStartTime)
        subTrack.tracksFeatIndxCG = scipy.delete(subTrack.tracksFeatIndxCG, timesToRemove, axis=1)
        timesToRemove = scipy.arange(8 * (splitTime - trackStartTime))
        subTrack.tracksCoordAmpCG = scipy.delete(subTrack.tracksCoordAmpCG, timesToRemove, axis=1)

        return subTrack

    if onlySubTrackStartToConsider is None:
        allNodesStart = startNodes + endNodes + splitNodes + mergeNodes
        allNodesEnd = allNodesStart
    else:
        startNodes = scipy.array(startNodes)
        allNodesStart = startNodes[track.seqOfEvents[startNodes - 1, 2] == onlySubTrackStartToConsider].tolist()
        allNodesEnd = startNodes.tolist() + endNodes + splitNodes + mergeNodes

    retVal = findAllPaths(trackGraphInfo.trackGraph, scipy.array(allNodesStart) - 1, scipy.array(allNodesEnd) - 1)
    paths = retVal.pathsOutput
    earlyExit = retVal.earlyExit
    if earlyExit:
        return []

    nPaths = len(paths)
    trackIdxMat = scipy.zeros((nPaths * 4, trackedFeatureInfo.shape[1]))
    trackIdxMat_splitExtra = None
    processedSplits = scipy.array([[0, 0]])

    for iPath in range(nPaths):
        path = paths[iPath]

        for node in path:
            possibleSubTracks = scipy.array(trackGraphInfo.subTrackIdx[node]).astype(scipy.int64)
            time = int(track.seqOfEvents[node, 0])
            featIdx = trackedFeatureInfo[possibleSubTracks - 1, time - 1]
            selectedSubTrack = scipy.unique(possibleSubTracks[featIdx != 0])

            while len(selectedSubTrack) > 1:
                if track.seqOfEvents[node + 1, 0] == time:
                    selectedSubTrack = scipy.setdiff1d(selectedSubTrack, trackGraphInfo.subTrackIdx[node + 1])

            trackIdxMat[4 * iPath, time - 1:] = trackedFeatureInfo[selectedSubTrack - 1, time - 1:]

        path = scipy.array(path)
        path = path[[0, -1]]
        time = track.seqOfEvents[path, 0]
        isSplit = scipy.nonzero(ismember(path[:1], scipy.array(splitNodes) - 1))[0].shape[0] > 0
        isMerge = scipy.nonzero(ismember(path[1:2], scipy.array(mergeNodes) - 1))[0].shape[0] > 0

        if isSplit:
            splitFrom = track.seqOfEvents[path[0], 3]
            time[0] -= 1
            trackIdxMat[4 * iPath, time[0] - 1] = trackedFeatureInfo[splitFrom - 1, time[0] - 1]

            if in1d_index(processedSplits, scipy.array([time[0], path[0]])).shape[0] == 0:
                if trackIdxMat_splitExtra is None:
                    trackIdxMat_splitExtra = fullIdxMatrix(splitTrackHere(time[0], path[0]), splitFrom)
                else:
                    trackIdxMat_splitExtra = scipy.concatenate(
                        (trackIdxMat_splitExtra, fullIdxMatrix(splitTrackHere(time[0], path[0]), splitFrom)))

                processedSplits = scipy.concatenate((processedSplits, scipy.array([[time[0], path[0]]])))

        if isSplit and isMerge and time[1] - time[0] > 1:
            trackIdxMat[(4 * iPath) + 1, time[0]:time[1] - 1] = trackIdxMat[4 * iPath, time[0]:time[1] - 1]

        if isSplit:
            trackIdxMat[(4 * iPath) + 2, time[0]:time[1]] = trackIdxMat[4 * iPath, time[0]:time[1]]

        if isMerge:
            trackIdxMat[(4 * iPath) + 3, time[0] - 1:time[1] - 1] = trackIdxMat[4 * iPath, time[0] - 1:time[1] - 1]

    if trackIdxMat_splitExtra is None:
        trackIdxMat = uniqueRows(trackIdxMat)
    else:
        trackIdxMat = uniqueRows(scipy.concatenate((trackIdxMat, trackIdxMat_splitExtra)))

    allZero = scipy.all(trackIdxMat == 0, axis=1)
    return trackIdxMat[~allZero, :]
