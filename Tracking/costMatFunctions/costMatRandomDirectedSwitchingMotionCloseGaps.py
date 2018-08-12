import scipy
import scipy.sparse
from scipy.spatial import distance

from Utrack.getTrackSEL import getTrackSEL
from Utrack.asymDeterm2D3D import asymDeterm2D3D
from Utilities.sparse.sparseIndexing import sparse_sum_row, sparse_sum_col, sparse_min_col
from Utilities.stats import mlPrctile

__author__ = 'edwardharry'


class CostMatRandomDirectedSwitchingMotionCloseGapsReturnValue(object):
    def __init__(self, costMat, nonlinkMarker, indxMerge, numMerge, indxSplit, numSplit, errFlag):
        self.costMat = costMat
        self.nonlinkMarker = nonlinkMarker
        self.indxMerge = indxMerge
        self.numMerge = numMerge
        self.indxSplit = indxSplit
        self.numSplit = numSplit
        self.errFlag = errFlag


class TracksPerFrame(object):
    def __init__(self, starts, ends):
        self.starts = starts
        self.ends = ends


class EstimTrackTypeParamRDS(object):
    def __init__(self, trackType, xyzVelS, xyzVelE, noiseStd, trackCentre, trackMeanDispMag, errFlag):
        self.trackType = trackType
        self.xyzVelS = xyzVelS
        self.xyzVelE = xyzVelE
        self.noiseStd = noiseStd
        self.trackCentre = trackCentre
        self.trackMeanDispMag = trackMeanDispMag
        self.errFlag = errFlag


class GetSearchRegionRDS(object):
    def __init__(self, longVecS, longVecE, shortVecS, shortVecE, shortVecS3D, shortVecE3D, longVecSMS, longVecEMS,
                 shortVecSMS, shortVecEMS, shortVecS3DMS, shortVecE3DMS, longRedVecS, longRedVecE, longRedVecSMS,
                 longRedVecEMS):
        self.longVecS = longVecS
        self.longVecE = longVecE
        self.shortVecS = shortVecS
        self.shortVecE = shortVecE
        self.shortVecS3D = shortVecS3D
        self.shortVecE3D = shortVecE3D
        self.longVecSMS = longVecSMS
        self.longVecEMS = longVecEMS
        self.shortVecSMS = shortVecSMS
        self.shortVecEMS = shortVecEMS
        self.shortVecS3DMS = shortVecS3DMS
        self.shortVecE3DMS = shortVecE3DMS
        self.longRedVecS = longRedVecS
        self.longRedVecE = longRedVecE
        self.longRedVecSMS = longRedVecSMS
        self.longRedVecEMS = longRedVecEMS


def costMatRandomDirectedSwitchingMotionCloseGaps(trackedFeatInfo, trackedFeatIndx, trackStartTime, trackEndTime,
                                                  costMatParam, gapCloseParam, kalmanFilterInfo, nnDistLinkedFeat,
                                                  probDim, movieInfo):
    costMat = scipy.array([])
    nonlinkMarker = scipy.array([])
    indxMerge = scipy.array([])
    numMerge = scipy.array([])
    indxSplit = scipy.array([])
    numSplit = scipy.array([])
    errFlag = scipy.array([])

    linearMotion = costMatParam.linearMotion
    minSearchRadius = costMatParam.minSearchRadius
    maxSearchRadius = costMatParam.maxSearchRadius
    brownStdMult = costMatParam.brownStdMult
    brownScaling = costMatParam.brownScaling
    timeReachConfB = costMatParam.timeReachConfB
    lenForClassify = costMatParam.lenForClassify
    useLocalDensity = costMatParam.useLocalDensity
    linStdMult = costMatParam.linStdMult
    linScaling = costMatParam.linScaling
    timeReachConfL = costMatParam.timeReachConfL

    sin2AngleMax = (scipy.sin(costMatParam.maxAngleVV * scipy.pi / 180)) ** 2
    sin2AngleMaxVD = 0.5
    nnWindow = costMatParam.nnWindow

    if useLocalDensity:
        closestDistScale = 2
        maxStdMult = 100
    else:
        closestDistScale = scipy.array([])
        maxStdMult = scipy.array([])

    if hasattr(costMatParam, 'ampRatioLimit') and costMatParam.ampRatioLimit is not None and len(
            costMatParam.ampRatioLimit) > 0:
        minAmpRatio = costMatParam.ampRatioLimit[0]
        maxAmpRatio = costMatParam.ampRatioLimit[1]
        useAmp = True
    else:
        minAmpRatio = 0
        maxAmpRatio = scipy.inf
        useAmp = False

    if hasattr(costMatParam, 'lftCdf') and costMatParam.lftCdf is not None and len(costMatParam.lftCdf) > 0:
        lftCdf = costMatParam.lftCdf
        oneMinusLftCdf = 1 - lftCdf
    else:
        lftCdf = scipy.array([])

    if hasattr(costMatParam, 'gapPenalty') and costMatParam.gapPenalty is not None:
        gapPenalty = costMatParam.gapPenalty
    else:
        gapPenalty = 1

    if hasattr(costMatParam, 'resLimit') and costMatParam.resLimit is not None:
        resLimit = costMatParam.resLimit
    else:
        resLimit = 0

    timeWindow = gapCloseParam.timeWindow
    mergeSplit = gapCloseParam.mergeSplit

    timeReachConfB = scipy.minimum(timeReachConfB, timeWindow)
    timeReachConfL = scipy.minimum(timeReachConfL, timeWindow)

    numTracks = trackedFeatInfo.shape[0]
    numFrames = trackedFeatInfo.shape[1]
    numFrames /= 8
    numFrames = int(scipy.floor(numFrames))

    tracksPerFrame = [
        TracksPerFrame(scipy.nonzero(trackStartTime - 1 == iFrame)[0], scipy.nonzero(trackEndTime - 1 == iFrame)[0]) for
        iFrame in range(numFrames)]

    coordStart = scipy.zeros((numTracks, probDim))
    ampStart = scipy.zeros((numTracks, 1))
    coordEnd = scipy.zeros((numTracks, probDim))
    ampEnd = scipy.zeros((numTracks, 1))
    for iTrack in range(numTracks):
        coordStart[iTrack, :] = trackedFeatInfo[iTrack,
                                (trackStartTime[iTrack] - 1) * 8:(trackStartTime[iTrack] - 1) * 8 + probDim].toarray()
        ampStart[iTrack] = trackedFeatInfo[iTrack, (trackStartTime[iTrack] - 1) * 8 + 3]
        coordEnd[iTrack, :] = trackedFeatInfo[iTrack,
                              (trackEndTime[iTrack] - 1) * 8:(trackEndTime[iTrack] - 1) * 8 + probDim].toarray()
        ampEnd[iTrack] = trackedFeatInfo[iTrack, (trackEndTime[iTrack] - 1) * 8 + 3]

    retStruct = estimTrackTypeParamRDS(trackedFeatIndx, trackedFeatInfo, kalmanFilterInfo, lenForClassify, probDim)
    trackType = retStruct.trackType
    xyzVelS = retStruct.xyzVelS
    xyzVelE = retStruct.xyzVelE
    noiseStd = retStruct.noiseStd
    trackCentre = retStruct.trackCentre
    trackMeanDispMag = retStruct.trackMeanDispMag

    if linearMotion == 0:
        trackType[trackType == 1] = 0

    noiseStdAll = noiseStd[noiseStd != 1]
    undetBrownStd = mlPrctile.percentile(noiseStdAll, 10)

    indx = scipy.nonzero(noiseStd == 1 & ~scipy.isnan(trackMeanDispMag))[0]
    noiseStd[indx] = trackMeanDispMag[indx] / scipy.sqrt(2)

    meanDispAllTrack = scipy.nanmean(trackMeanDispMag)

    retStruct = getSearchRegionRDS(xyzVelS, xyzVelE, noiseStd, trackType, undetBrownStd, timeWindow, brownStdMult,
                                   linStdMult, timeReachConfB, timeReachConfL, minSearchRadius, maxSearchRadius,
                                   useLocalDensity, closestDistScale, maxStdMult, nnDistLinkedFeat, nnWindow,
                                   trackStartTime, trackEndTime, probDim, resLimit, brownScaling, linScaling,
                                   linearMotion)
    longVecSAll = retStruct.longVecS
    longVecEAll = retStruct.longVecE
    shortVecSAll = retStruct.shortVecS
    shortVecEAll = retStruct.shortVecE
    shortVecS3DAll = retStruct.shortVecS3D
    shortVecE3DAll = retStruct.shortVecE3D
    longVecSAllMS = retStruct.longVecSMS
    longVecEAllMS = retStruct.longVecEMS
    shortVecSAllMS = retStruct.shortVecSMS
    shortVecEAllMS = retStruct.shortVecEMS
    shortVecS3DAllMS = retStruct.shortVecS3DMS
    shortVecE3DAllMS = retStruct.shortVecE3DMS
    longRedVecSAll = retStruct.longRedVecS
    longRedVecEAll = retStruct.longRedVecE
    longRedVecSAllMS = retStruct.longRedVecSMS
    longRedVecEAllMS = retStruct.longRedVecEMS
    del retStruct

    indxEnd2 = scipy.array([]).astype(scipy.int32)
    indxStart2 = scipy.array([]).astype(scipy.int32)

    maxDispAllowed = scipy.maximum(
        scipy.amax(scipy.absolute(scipy.concatenate((xyzVelS.flatten(), xyzVelE.flatten()), axis=0))) * probDim *
        linStdMult[0] * 3, maxSearchRadius)

    for iFrame in range(numFrames - 1):

        endsToConsider = tracksPerFrame[iFrame].ends

        for jFrame in range(iFrame + 1, scipy.minimum(iFrame + timeWindow, numFrames)):

            startsToConsider = tracksPerFrame[jFrame].starts

            dispMat2 = distance.cdist(coordEnd[endsToConsider, :], coordStart[startsToConsider, :])

            tmpFrame = jFrame - iFrame
            indxEnd3 = scipy.nonzero(dispMat2 <= (maxDispAllowed * tmpFrame))
            indxStart3 = indxEnd3[1]
            indxEnd3 = indxEnd3[0]

            toDel = []
            toDelIndex = -1
            if movieInfo[0].previousTrackFunc is not None:
                for tIdx in endsToConsider[indxEnd3]:
                    idx = trackedFeatIndx[tIdx, iFrame]
                    toDelIndex += 1
                    _, afterIdx = movieInfo[0].PreviousTracks(iFrame, idx)
                    if afterIdx is not None and afterIdx < 0:
                        toDel.append(toDelIndex)

                toDelIndex = -1
                for tIdx in startsToConsider[indxStart3]:
                    idx = trackedFeatIndx[tIdx, jFrame]
                    toDelIndex += 1
                    beforeIdx, _ = movieInfo[0].PreviousTracks(jFrame, idx)
                    if beforeIdx is not None and beforeIdx < 0:
                        toDel.append(toDelIndex)

            indxStart3 = scipy.delete(indxStart3, toDel)
            indxEnd3 = scipy.delete(indxEnd3, toDel)

            if indxEnd3.shape[0] == 1:
                indxEnd3 = indxEnd3.transpose()
                indxStart3 = indxStart3.transpose()

            indxEnd2 = scipy.concatenate((indxEnd2, endsToConsider[indxEnd3]), axis=0)
            indxStart2 = scipy.concatenate((indxStart2, startsToConsider[indxStart3]), axis=0)

    numPairs = indxEnd2.shape[0]

    del dispMat2
    del maxDispAllowed

    indx1 = scipy.zeros((numPairs, 1))
    indx2 = scipy.zeros((numPairs, 1))
    cost = scipy.zeros((numPairs, 1))

    timeScalingLin = scipy.concatenate((scipy.arange(1, timeReachConfL + 1) ** linScaling[0],
                                        (timeReachConfL ** linScaling[0]) * (
                                            scipy.arange(2, timeWindow - timeReachConfL + 2) ** linScaling[1])), axis=0)

    timeScalingBrown = scipy.concatenate((scipy.arange(1, timeReachConfB + 1) ** brownScaling[0],
                                          (timeReachConfB ** brownScaling[0]) * (
                                              scipy.arange(2, timeWindow - timeReachConfB + 2) ** brownScaling[1])),
                                         axis=0)

    for iPair in range(numPairs):

        iStart = indxStart2[iPair]
        iEnd = indxEnd2[iPair]

        timeGap = trackStartTime[iStart] - trackEndTime[iEnd] - 1

        trackTypeS = trackType[iStart]
        trackTypeE = trackType[iEnd]

        dispVec = scipy.mat(coordStart[iStart, :] - coordEnd[iEnd, :])

        if scipy.any(scipy.isnan(dispVec)):
            dispVecMag = scipy.nan
        else:
            dispVecMag = distance.norm(dispVec)

        parallelToS = (dispVec * scipy.mat(xyzVelS[iStart, :]).transpose()) > 0
        parallelToE = (dispVec * scipy.mat(xyzVelE[iEnd, :]).transpose()) > 0

        if linearMotion == 1 and not parallelToS:
            longVecS = scipy.mat(longRedVecSAll[:, timeGap, iStart]).transpose()
        else:
            longVecS = scipy.mat(longVecSAll[:, timeGap, iStart]).transpose()
        shortVecS = scipy.mat(shortVecSAll[:, timeGap, iStart]).transpose()

        if linearMotion == 1 and not parallelToE:
            longVecE = scipy.mat(longRedVecEAll[:, timeGap, iEnd]).transpose()
        else:
            longVecE = scipy.mat(longVecEAll[:, timeGap, iEnd]).transpose()
        shortVecE = scipy.mat(shortVecEAll[:, timeGap, iEnd]).transpose()

        if scipy.any(scipy.isnan(longVecS)):
            longVecMagS = scipy.nan
        else:
            longVecMagS = distance.norm(longVecS)

        if scipy.any(scipy.isnan(shortVecS)):
            shortVecMagS = scipy.nan
        else:
            shortVecMagS = distance.norm(shortVecS)

        if scipy.any(scipy.isnan(longVecE)):
            longVecMagE = scipy.nan
        else:
            longVecMagE = distance.norm(longVecE)

        if scipy.any(scipy.isnan(shortVecE)):
            shortVecMagE = scipy.nan
        else:
            shortVecMagE = distance.norm(shortVecE)

        projStartLong = scipy.absolute(dispVec * longVecS) / longVecMagS
        projStartShort = scipy.absolute(dispVec * shortVecS) / shortVecMagS

        projEndLong = scipy.absolute(dispVec * longVecE) / longVecMagE
        projEndShort = scipy.absolute(dispVec * shortVecE) / shortVecMagE

        if probDim == 3:

            shortVecS3D = scipy.mat(shortVecS3DAll[:, timeGap, iStart]).transpose()
            shortVecE3D = scipy.mat(shortVecE3DAll[:, timeGap, iEnd]).transpose()
            shortVecMagS3D = scipy.sqrt(shortVecS3D.transpose() * shortVecS3D)
            shortVecMagE3D = scipy.sqrt(shortVecE3D.transpose() * shortVecE3D)
            projStartShort3D = scipy.absolute(dispVec * shortVecS3D) / shortVecMagS3D
            projEndShort3D = scipy.absolute(dispVec * shortVecE3D) / shortVecMagE3D

        else:

            shortVecMagS3D = 0
            shortVecMagE3D = 0
            projStartShort3D = 0
            projEndShort3D = 0

        cen2cenVec = dispVec.copy()
        cen2cenVecMag = scipy.sqrt(cen2cenVec * cen2cenVec.transpose())

        if trackTypeE == 1:

            if trackTypeS == 1:

                cosAngle = longVecE.transpose() * longVecS / (longVecMagE * longVecMagS)
                sin2Angle = 1 - (cosAngle ** 2)

                sin2AngleE = 1 - (cen2cenVec * longVecE / (longVecMagE * cen2cenVecMag)) ** 2
                sin2AngleS = 1 - (cen2cenVec * longVecS / (longVecMagS * cen2cenVecMag)) ** 2

                possibleLink = ((projEndLong <= longVecMagE) and (projEndShort <= shortVecMagE) and (
                    projEndShort3D <= shortVecMagE3D)) and (
                                   (projStartLong <= longVecMagS) and (projStartShort <= shortVecMagS) and (
                                       projStartShort3D <= shortVecMagS3D)) and (sin2Angle <= sin2AngleMax) and (
                                   (sin2AngleE <= sin2AngleMaxVD) and (sin2AngleS <= sin2AngleMaxVD))

                if linearMotion == 1:
                    possibleLink = possibleLink and (cosAngle >= 0)

            elif trackTypeS == 0:

                sin2AngleE = 1 - (cen2cenVec * longVecE / (longVecMagE * cen2cenVecMag)) ** 2

                possibleLink = ((projEndLong <= longVecMagE) and (projEndShort <= shortVecMagE) and (
                    projEndShort3D <= shortVecMagE3D)) and (dispVecMag <= longVecMagS) and (
                    sin2AngleE <= sin2AngleMaxVD)

            else:

                sin2AngleE = 1 - (cen2cenVec * longVecE / (longVecMagE * cen2cenVecMag)) ** 2

                possibleLink = ((projEndLong <= longVecMagE) and (projEndShort <= shortVecMagE) and (
                    projEndShort3D <= shortVecMagE3D)) and (sin2AngleE <= sin2AngleMaxVD)

        elif trackTypeE == 0:

            if trackTypeS == 1:

                sin2AngleS = 1 - (cen2cenVec * longVecS / (longVecMagS * cen2cenVecMag)) ** 2

                possibleLink = (dispVecMag <= longVecMagE) and (
                    (projStartLong <= longVecMagS) and (projStartShort <= shortVecMagS) and (
                        projStartShort3D <= shortVecMagS3D)) and (sin2AngleS <= sin2AngleMaxVD)

            elif trackTypeS == 0:

                possibleLink = (dispVecMag <= longVecMagE) and (dispVecMag <= longVecMagS)

            else:

                possibleLink = (dispVecMag <= longVecMagE)

        else:

            if trackTypeS == 1:

                sin2AngleS = 1 - (cen2cenVec * longVecS / (longVecMagS * cen2cenVecMag)) ** 2

                possibleLink = ((projStartLong <= longVecMagS) and (projStartShort <= shortVecMagS) and (
                    projStartShort3D <= shortVecMagS3D)) and (sin2AngleS <= sin2AngleMaxVD)

            elif trackTypeS == 0:

                possibleLink = (dispVecMag <= longVecMagS)

            else:

                possibleLink = (dispVecMag <= longVecMagE) and (dispVecMag <= longVecMagS)

        if possibleLink:

            meanDispTrack1 = trackMeanDispMag[iStart]
            meanDispTrack1[scipy.isnan(meanDispTrack1)] = meanDispAllTrack
            meanDispTrack2 = trackMeanDispMag[iEnd]
            meanDispTrack2[scipy.isnan(meanDispTrack2)] = meanDispAllTrack
            meanDisp2Tracks = scipy.mean(scipy.concatenate((meanDispTrack1, meanDispTrack2), axis=0), axis=0)

            dispVecMag2 = dispVecMag ** 2

            if trackTypeE == 1 and trackTypeS == 1:
                cost12 = dispVecMag2 * (1 + scipy.mean(
                    scipy.concatenate((sin2Angle, scipy.concatenate((sin2AngleE, sin2AngleS), axis=0)), axis=0),
                    axis=0)) / (timeScalingLin[timeGap] * meanDisp2Tracks) ** 2
            elif trackTypeE == 1:
                cost12 = dispVecMag2 * (1 + sin2AngleE) / (scipy.mean(scipy.concatenate(
                    (timeScalingLin[timeGap] * meanDispTrack2, timeScalingBrown[timeGap] * meanDispTrack1), axis=0),
                    axis=0)) ** 2
            elif trackTypeS == 1:
                cost12 = dispVecMag2 * (1 + sin2AngleS) / (scipy.mean(scipy.concatenate(
                    (timeScalingLin[timeGap] * meanDispTrack1, timeScalingBrown[timeGap] * meanDispTrack2), axis=0),
                    axis=0)) ** 2
            else:
                cost12 = dispVecMag2 / (timeScalingBrown[timeGap] * meanDisp2Tracks) ** 2

            if len(lftCdf) > 0:
                cost12 = cost12 / oneMinusLftCdf[trackEndTime[iStart] - trackStartTime[iEnd] + 1]

            if scipy.isfinite(cost12):
                cost12 *= gapPenalty ** timeGap

                cost[iPair] = cost12

                indx1[iPair] = iEnd + 1
                indx2[iPair] = iStart + 1

    possiblePairs = scipy.nonzero(indx1 != 0)
    indx1 = indx1[possiblePairs] - 1
    indx2 = indx2[possiblePairs] - 1
    cost = cost[possiblePairs]

    del possiblePairs

    numMerge = 0
    indxMerge = scipy.array([])
    altCostMerge = scipy.array([])
    numSplit = 0
    indxSplit = scipy.array([])
    altCostSplit = scipy.array([])

    if mergeSplit > 0:

        maxDispAllowed = scipy.maximum(
            scipy.amax(scipy.absolute(scipy.concatenate((xyzVelS.flatten(), xyzVelE.flatten()), axis=0)),
                axis=0) * probDim * linStdMult[0], maxSearchRadius)
        maxDispAllowed = scipy.maximum(maxDispAllowed, resLimit)

        if mergeSplit == 1 or mergeSplit == 2:

            for endTime in range(numFrames - 1):

                endsToConsider = tracksPerFrame[endTime].ends

                mergesToConsider = scipy.intersect1d(
                    scipy.concatenate([i.starts for i in tracksPerFrame[0:endTime + 1]]),
                    scipy.concatenate([i.ends for i in tracksPerFrame[endTime + 1:]]))

                timeIndx = (endTime + 1) * 8

                tmp = trackedFeatInfo[mergesToConsider, timeIndx:timeIndx + probDim]
                if scipy.sparse.issparse(tmp):
                    tmp = tmp.toarray()

                dispMat2 = distance.cdist(coordEnd[endsToConsider, :], tmp)

                indxEnd2 = scipy.nonzero(dispMat2 <= maxDispAllowed)
                indxMerge2 = indxEnd2[1]
                indxEnd2 = indxEnd2[0]
                numPairs = len(indxEnd2)

                del dispMat2

                indxEnd2 = endsToConsider[indxEnd2]
                indxMerge2 = mergesToConsider[indxMerge2]

                indx1MS = scipy.zeros((numPairs, 1)).astype(scipy.int32)
                indx2MS = scipy.zeros((numPairs, 1)).astype(scipy.int32)
                costMS = scipy.zeros((numPairs, 1))
                altCostMS = scipy.zeros((numPairs, 1))
                indxMSMS = scipy.zeros((numPairs, 1)).astype(scipy.int32)

                for iPair in range(numPairs):

                    iEnd = indxEnd2[iPair]
                    iMerge = indxMerge2[iPair]

                    tmp = trackedFeatInfo[iMerge, timeIndx:timeIndx + probDim]
                    if scipy.sparse.issparse(tmp):
                        tmp = tmp.toarray()

                    dispVec = scipy.mat(tmp - coordEnd[iEnd, :])
                    dispVecMag = scipy.sqrt(dispVec * dispVec.transpose())

                    parallelToE = (dispVec * scipy.mat(xyzVelE[iEnd, :]).transpose()) > 0

                    if linearMotion == 1 and not parallelToE:
                        longVecE = scipy.mat(longRedVecEAllMS[:, 0, iEnd]).transpose()
                    else:
                        longVecE = scipy.mat(longVecEAllMS[:, 0, iEnd]).transpose()
                    shortVecE = scipy.mat(shortVecEAllMS[:, 0, iEnd]).transpose()

                    longVecMagE = scipy.sqrt(longVecE.transpose() * longVecE)
                    shortVecMagE = scipy.sqrt(shortVecE.transpose() * shortVecE)

                    projEndLong = scipy.absolute(dispVec * longVecE) / longVecMagE
                    projEndShort = scipy.absolute(dispVec * shortVecE) / shortVecMagE

                    indxBefore = ((8 * endTime) + 3) - (8 * scipy.arange(0, 5))
                    indxBefore = indxBefore[indxBefore >= 0]
                    ampE = trackedFeatInfo[iEnd, indxBefore]
                    if scipy.sparse.issparse(ampE):
                        ampE = ampE.toarray()
                    ampE = scipy.mean(ampE[ampE != 0])

                    ampM1 = trackedFeatInfo[iMerge, indxBefore]
                    if scipy.sparse.issparse(ampM1):
                        ampM1 = ampM1.toarray()
                    ampM1 = scipy.mean(ampM1[ampM1 != 0])
                    indxAfter = ((8 * (endTime + 1)) + 3) + (8 * scipy.arange(0, 5))
                    indxAfter = indxAfter[indxAfter < (8 * numFrames)]
                    ampM = trackedFeatInfo[iMerge, indxAfter]
                    if scipy.sparse.issparse(ampM):
                        ampM = ampM.toarray()
                    ampM = scipy.mean(ampM[ampM != 0])

                    ampRatio = ampM / (ampE + ampM1)

                    ampRatioIndME = ampM / ampE
                    ampRatioIndMM1 = ampM / ampM1

                    if not useAmp:
                        ampRatio = 1
                        ampM = 1
                        ampM1 = 1
                        ampRatioIndME = 1.1
                        ampRatioIndMM1 = 1.1

                    if trackType[iEnd] == 1:

                        if probDim == 3:
                            shortVecE3D = scipy.mat(shortVecE3DAllMS[:, 0, iEnd]).transpose()
                            shortVecMagE3D = scipy.sqrt(shortVecE3D.transpose() * shortVecE3D)
                            projEndShort3D = scipy.absolute(dispVec * shortVecE3D) / shortVecMagE3D
                        else:
                            shortVecMagE3D = 0
                            projEndShort3D = 0

                        cen2cenVec = dispVec.copy()
                        cen2cenVecMag = scipy.sqrt(cen2cenVec * cen2cenVec.transpose())

                        sin2AngleE = 1 - (cen2cenVec * longVecE / (longVecMagE * cen2cenVecMag)) ** 2

                        possibleLink = projEndLong <= longVecMagE and projEndShort <= shortVecMagE and projEndShort3D <= shortVecMagE3D and sin2AngleE <= sin2AngleMaxVD and minAmpRatio <= ampRatio <= maxAmpRatio and ampRatioIndME > 1 and ampRatioIndMM1 > 1 and scipy.absolute(
                            ampRatio - 1) < scipy.absolute(ampRatioIndMM1 - 1)

                    else:

                        sin2AngleE = 0

                        possibleLink = dispVecMag <= longVecMagE and minAmpRatio <= ampRatio <= maxAmpRatio and ampRatioIndME > 1 and ampRatioIndMM1 > 1 and scipy.absolute(
                            ampRatio - 1) < scipy.absolute(ampRatioIndMM1 - 1)

                    if possibleLink:

                        dispVecMag2 = dispVecMag ** 2
                        ampCost = ampRatio
                        if ampCost < 1:
                            ampCost **= -2
                        meanDisp2Tracks = trackMeanDispMag[iEnd]
                        if scipy.isnan(meanDisp2Tracks):
                            meanDisp2Tracks = meanDispAllTrack
                        cost12 = dispVecMag2 * ampCost * (1 + sin2AngleE) / (meanDisp2Tracks ** 2)

                        if len(lftCdf) > 0:
                            cost12 = cost12 / oneMinusLftCdf[trackEndTime[iMerge] - trackStartTime[iEnd] + 1]

                        if ~scipy.isinf(cost12):

                            costMS[iPair] = cost12

                            prevAppearance = scipy.nonzero(indxMSMS == (iMerge + 1))[0]

                            if len(prevAppearance) == 0:

                                numMerge += 1

                                indxMSMS[iPair] = iMerge + 1

                                indx1MS[iPair] = iEnd + 1
                                indx2MS[iPair] = numMerge + numTracks

                                trackCoord = trackedFeatInfo[indxMSMS[iPair] - 1, :]
                                if scipy.sparse.issparse(trackCoord):
                                    trackCoord = trackCoord.toarray()
                                    trackCoord = scipy.reshape(trackCoord, (-1, 8)).transpose()
                                    trackCoord[trackCoord == 0] = scipy.nan
                                    if probDim == 2:
                                        trackCoord[2, :] = 0
                                        trackCoord[6, :] = 0
                                else:
                                    trackCoord = scipy.reshape(trackCoord, (-1, 8)).transpose()
                                dispVecMag2 = (scipy.diff(trackCoord, 1, 1)) ** 2
                                dispVecMag2 = scipy.nanmean(dispVecMag2, 1)
                                dispVecMag2 = scipy.sum(dispVecMag2[:probDim])

                                dispVecMag2 = scipy.maximum(dispVecMag2, resLimit ** 2)

                                ampCost = ampM / ampM1
                                if ampCost < 1:
                                    ampCost **= -2

                                meanDisp1Track = trackMeanDispMag[indxMSMS[iPair, 0] - 1, 0]
                                if scipy.isnan(meanDisp1Track):
                                    meanDisp1Track = meanDispAllTrack

                                cost12 = dispVecMag2 * ampCost / (meanDisp1Track ** 2)

                                altCostMS[iPair] = cost12

                            else:

                                indx1MS[iPair] = iEnd + 1
                                indx2MS[iPair] = indx2MS[prevAppearance]

                possiblePairs = scipy.nonzero(indx1MS != 0)
                indx1MS = indx1MS[possiblePairs] - 1
                indx2MS = indx2MS[possiblePairs] - 1
                costMS = costMS[possiblePairs]
                possibleMerges = scipy.nonzero(indxMSMS != 0)
                indxMSMS = indxMSMS[possibleMerges] - 1
                altCostMS = altCostMS[possibleMerges]
                del possiblePairs
                del possibleMerges

                indx1 = scipy.concatenate((indx1, indx1MS), axis=0)
                indx2 = scipy.concatenate((indx2, indx2MS), axis=0)
                cost = scipy.concatenate((cost, costMS), axis=0)
                altCostMerge = scipy.concatenate((altCostMerge, altCostMS), axis=0)
                indxMerge = scipy.concatenate((indxMerge, indxMSMS), axis=0)

        if mergeSplit == 1 or mergeSplit == 3:

            for startTime in range(1, numFrames):

                startsToConsider = tracksPerFrame[startTime].starts
                splitsToConsider = scipy.intersect1d(scipy.concatenate([i.starts for i in tracksPerFrame[:startTime]]),
                                                     scipy.concatenate([i.ends for i in tracksPerFrame[startTime:]]))

                timeIndx = (startTime - 1) * 8

                tmp = trackedFeatInfo[splitsToConsider, timeIndx:timeIndx + probDim]
                if scipy.sparse.issparse(tmp):
                    tmp = tmp.toarray()
                dispMat2 = distance.cdist(coordStart[startsToConsider, :], tmp)

                indxStart2 = scipy.nonzero(dispMat2 <= maxDispAllowed)
                indxSplit2 = indxStart2[1]
                indxStart2 = indxStart2[0]
                numPairs = len(indxStart2)

                del dispMat2

                indxStart2 = startsToConsider[indxStart2]
                indxSplit2 = splitsToConsider[indxSplit2]

                indx1MS = scipy.zeros((numPairs, 1)).astype(scipy.int32)
                indx2MS = scipy.zeros((numPairs, 1)).astype(scipy.int32)
                costMS = scipy.zeros((numPairs, 1))
                altCostMS = scipy.zeros((numPairs, 1))
                indxMSMS = scipy.zeros((numPairs, 1)).astype(scipy.int32)

                for iPair in range(numPairs):

                    iStart = indxStart2[iPair]
                    iSplit = indxSplit2[iPair]

                    idxStart = trackedFeatIndx[iStart, startTime]
                    idxSplit = trackedFeatIndx[iSplit, startTime - 1]

                    tmp = trackedFeatInfo[iSplit, timeIndx:timeIndx + probDim]
                    if scipy.sparse.issparse(tmp):
                        tmp = tmp.toarray()
                    dispVec = scipy.mat(coordStart[iStart, :] - tmp)
                    dispVecMag = scipy.sqrt(dispVec * dispVec.transpose())

                    parallelToS = (dispVec * scipy.mat(xyzVelS[iStart, :]).transpose()) > 0

                    if linearMotion == 1 and not parallelToS:
                        longVecS = scipy.mat(longRedVecSAllMS[:, 0, iStart]).transpose()
                    else:
                        longVecS = scipy.mat(longVecSAllMS[:, 0, iStart]).transpose()
                    shortVecS = scipy.mat(shortVecSAllMS[:, 0, iStart]).transpose()

                    longVecMagS = scipy.sqrt(longVecS.transpose() * longVecS)
                    shortVecMagS = scipy.sqrt(shortVecS.transpose() * shortVecS)

                    projStartLong = scipy.absolute(dispVec * longVecS) / longVecMagS
                    projStartShort = scipy.absolute(dispVec * shortVecS) / shortVecMagS

                    indxAfter = ((8 * startTime) + 3) + (8 * scipy.arange(0, 5))
                    indxAfter = indxAfter[indxAfter < (8 * numFrames)]
                    ampS = trackedFeatInfo[iStart, indxAfter]
                    if scipy.sparse.issparse(ampS):
                        ampS = ampS.toarray()
                    ampS = scipy.mean(ampS[ampS != 0])

                    ampSp1 = trackedFeatInfo[iSplit, indxAfter]
                    if scipy.sparse.issparse(ampSp1):
                        ampSp1 = ampSp1.toarray()
                    ampSp1 = scipy.mean(ampSp1 != 0)
                    indxBefore = ((8 * (startTime - 1)) + 3) - (8 * scipy.arange(0, 5))
                    indxBefore = indxBefore[indxBefore > 0]
                    ampSp = trackedFeatInfo[iSplit, indxBefore]
                    if scipy.sparse.issparse(ampSp):
                        ampSp = ampSp.toarray()
                    ampSp = scipy.mean(ampSp[ampSp != 0])

                    ampRatio = ampSp / (ampS * ampSp1)

                    ampRatioIndSpS = ampSp / ampS
                    ampRatioIndSpSp1 = ampSp / ampSp1

                    if not useAmp:
                        ampRatio = 1
                        ampSp = 1
                        ampSp1 = 1
                        ampRatioIndSpS = 1.1
                        ampRatioIndSpSp1 = 1.1

                    if trackType[iStart] == 1:

                        if probDim == 3:
                            shortVecS3D = scipy.mat(shortVecS3DAllMS[:, 0, iStart]).transpose()
                            shortVecMagS3D = scipy.sqrt(shortVecS3D.transpose() * shortVecS3D)
                            projStartShort3D = scipy.absolute(dispVec * shortVecS3D) / shortVecMagS3D
                        else:
                            shortVecMagS3D = 0
                            projStartShort3D = 0

                        cen2cenVec = dispVec.copy()
                        cen2cenVecMag = scipy.sqrt(cen2cenVec * cen2cenVec.transpose())

                        sin2AngleS = 1 - (cen2cenVec * longVecS / (longVecMagS * cen2cenVecMag)) ** 2

                        possibleLink = projStartLong <= longVecMagS and projStartShort <= shortVecMagS and projStartShort3D <= shortVecMagS3D and sin2AngleS <= sin2AngleMaxVD and minAmpRatio <= ampRatio <= maxAmpRatio and ampRatioIndSpS > 1 and ampRatioIndSpSp1 > 1 and scipy.absolute(
                            ampRatio - 1) < scipy.absolute(ampRatioIndSpSp1 - 1)

                    else:

                        sin2AngleS = 0

                        possibleLink = dispVecMag <= longVecMagS and minAmpRatio <= ampRatio <= maxAmpRatio and ampRatioIndSpS > 1 and ampRatioIndSpSp1 > 1 and scipy.absolute(
                            ampRatio - 1) < scipy.absolute(ampRatioIndSpSp1 - 1)

                    if possibleLink and movieInfo[0].splitsFunc is not None:
                        splitFrame = movieInfo[0].GetSplitFrame(idxSplit, idxStart, startTime)
                        possibleLink = splitFrame is None or splitFrame == startTime

                    if possibleLink:

                        dispVecMag2 = dispVecMag ** 2
                        ampCost = ampRatio
                        if ampCost < 1:
                            ampCost **= -2
                        meanDisp2Tracks = trackMeanDispMag[iStart]
                        if scipy.isnan(meanDisp2Tracks):
                            meanDisp2Tracks = meanDispAllTrack
                        cost12 = dispVecMag2 * ampCost * (1 + sin2AngleS) / (meanDisp2Tracks ** 2)

                        if len(lftCdf) > 0:
                            cost12 = cost12 / oneMinusLftCdf[trackEndTime[iStart] - trackStartTime[iSplit] + 1]

                        if ~scipy.isinf(cost12):

                            costMS[iPair] = cost12

                            prevAppearance = scipy.nonzero(indxMSMS == (iSplit + 1))[0]

                            if len(prevAppearance) == 0:

                                numSplit += 1

                                indxMSMS[iPair] = iSplit + 1

                                indx1MS[iPair] = numSplit + numTracks
                                indx2MS[iPair] = iStart + 1

                                trackCoord = trackedFeatInfo[indxMSMS[iPair], :]
                                if scipy.sparse.issparse(trackCoord):
                                    trackCoord = trackCoord.toarray()
                                    trackCoord = scipy.reshape(trackCoord, (-1, 8)).transpose()
                                    trackCoord[trackCoord == 0] = scipy.nan
                                    if probDim == 2:
                                        trackCoord[2, :] = 0
                                        trackCoord[6, :] = 0
                                else:
                                    trackCoord = scipy.reshape(trackCoord, (-1, 8)).transpose()
                                dispVecMag2 = scipy.diff(trackCoord, 1, 1) ** 2
                                dispVecMag2 = scipy.nanmean(dispVecMag2, 1)
                                dispVecMag2 = scipy.sum(dispVecMag2[:probDim])

                                dispVecMag2 = scipy.maximum(dispVecMag2, resLimit ** 2)

                                ampCost = ampSp / ampSp1
                                if ampCost < 1:
                                    ampCost **= -2

                                meanDisp1Track = trackMeanDispMag[indxMSMS[iPair, 0] - 1, 0]
                                if scipy.isnan(meanDisp1Track):
                                    meanDisp1Track = meanDispAllTrack

                                cost12 = dispVecMag2 * ampCost / (meanDisp1Track ** 2)

                                altCostMS[iPair] = cost12

                            else:

                                indx1MS[iPair] = indx1MS[prevAppearance]
                                indx2MS[iPair] = iStart + 1

                possiblePairs = scipy.nonzero(indx1MS != 0)
                indx1MS = indx1MS[possiblePairs] - 1
                indx2MS = indx2MS[possiblePairs] - 1
                costMS = costMS[possiblePairs]
                possibleSplits = scipy.nonzero(indxMSMS != 0)
                altCostMS = altCostMS[possibleSplits]
                indxMSMS = indxMSMS[possibleSplits] - 1
                del possiblePairs
                del possibleSplits

                indx1 = scipy.concatenate((indx1, indx1MS), axis=0)
                indx2 = scipy.concatenate((indx2, indx2MS), axis=0)
                cost = scipy.concatenate((cost, costMS), axis=0)
                altCostSplit = scipy.concatenate((altCostSplit, altCostMS), axis=0)
                indxSplit = scipy.concatenate((indxSplit, indxMSMS), axis=0)

    numEndSplit = numTracks + numSplit
    numStartMerge = numTracks + numMerge
    costMat = scipy.sparse.coo_matrix((cost, (indx1, indx2)), shape=(numEndSplit, numStartMerge))

    tmp = costMat != 0
    numPotAssignRow = sparse_sum_row(tmp)
    numPotAssignCol = sparse_sum_col(tmp)
    numPotAssignColAll = scipy.sum(numPotAssignCol)
    numPotAssignRowAll = scipy.sum(numPotAssignRow)
    numPartCol = len(numPotAssignCol) * 2
    extraCol = (numPotAssignColAll - numPartCol) / numPotAssignColAll
    numPartRow = len(numPotAssignRow) * 2
    extraRow = (numPotAssignRowAll - numPartRow) / numPotAssignRowAll
    prctile2use = scipy.minimum(100, 100 - scipy.mean(scipy.concatenate(([extraRow], [extraCol]))) * 100)

    costBD = 1.05 * mlPrctile.percentile(cost.flatten(), prctile2use)

    costMat = scipy.sparse.vstack((scipy.sparse.hstack(
        (costMat, scipy.sparse.spdiags(costBD * scipy.ones((numTracks + numSplit)), 0, numEndSplit, numEndSplit))),
                                   scipy.sparse.hstack((
                                       scipy.sparse.spdiags(costBD * scipy.ones((numTracks + numStartMerge)), 0,
                                                            numStartMerge, numStartMerge),
                                       scipy.sparse.coo_matrix((costBD * scipy.ones((len(indx1))), (indx2, indx1)),
                                                               shape=(numStartMerge, numEndSplit))))))

    nonlinkMarker = scipy.minimum(scipy.floor(scipy.amin(sparse_min_col(costMat))) - 5, -5)

    return CostMatRandomDirectedSwitchingMotionCloseGapsReturnValue(costMat, nonlinkMarker, indxMerge, numMerge,
                                                                    indxSplit, numSplit, errFlag)


def estimTrackTypeParamRDS(trackedFeatIndx, trackedFeatInfo, kalmanFilterInfo, lenForClassify, probDim):
    errFlag = False

    numTracksLink = trackedFeatIndx.shape[0]
    numFrames = trackedFeatIndx.shape[1]

    trackType = scipy.nan * scipy.ones((numTracksLink, 1))
    xyzVelS = scipy.zeros((numTracksLink, probDim))
    xyzVelE = scipy.zeros((numTracksLink, probDim))
    noiseStd = scipy.zeros((numTracksLink, 1))
    trackCentre = scipy.zeros((numTracksLink, probDim))
    trackMeanDispMag = scipy.nan * scipy.ones((numTracksLink, 1))

    trackSEL = getTrackSEL(trackedFeatInfo)
    trackStartTime = trackSEL[:, 0]
    trackEndTime = trackSEL[:, 1]
    trackLifeTime = trackSEL[:, 2]
    del trackSEL

    if probDim == 2:

        asymThresh = scipy.concatenate((scipy.array(
            [scipy.nan, scipy.nan, 5, 2.7, 2.1, 1.8, 1.7, 1.6, 1.5, 1.45, 1.45, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.35,
             1.35, 1.35]), 1.3 * scipy.ones((scipy.maximum((numFrames - 20), 0)))), axis=0)

    else:

        asymThresh = scipy.concatenate((scipy.array(
            [scipy.nan, scipy.nan, 2.9, 1.9, 1.5, 1.4, 1.3, 1.3, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2]),
                                        1.1 * scipy.ones(scipy.maximum((numFrames - 15), 0))), axis=0)

    for iTrack in range(numTracksLink):

        tmp = trackedFeatInfo[iTrack, :]

        if scipy.sparse.issparse(tmp):
            tmp = tmp.toarray()

        currentTrack = scipy.reshape(tmp, (-1, 8))
        currentTrack = currentTrack[:, :probDim]
        currentTrack = currentTrack[trackStartTime[iTrack] - 1:trackEndTime[iTrack], :]

        trackCentre[iTrack, :] = scipy.mean(currentTrack, axis=0)

        if currentTrack.shape[0] > 1:
            trackMeanDispMag[iTrack] = scipy.mean(scipy.sqrt(scipy.sum(scipy.diff(currentTrack, 1, 0) ** 2, axis=1)))

        overallType = scipy.nan

        if trackLifeTime[iTrack] >= lenForClassify:
            asymmetry = asymDeterm2D3D(currentTrack).asymParam

            overallType = asymmetry > asymThresh[trackLifeTime[iTrack] - 1]

        trackType[iTrack] = overallType

        if ~scipy.isnan(overallType):

            noiseStd[iTrack] = scipy.sqrt(scipy.absolute(kalmanFilterInfo[trackEndTime[iTrack] - 1].noiseVar[
                0, 0, trackedFeatIndx[iTrack, trackEndTime[iTrack] - 1] - 1]))

            if overallType:
                xyzVelS[iTrack, :] = kalmanFilterInfo[trackStartTime[iTrack] - 1].stateVec[
                                     trackedFeatIndx[iTrack, trackStartTime[iTrack] - 1] - 1, probDim:2 * probDim]
                xyzVelE[iTrack, :] = kalmanFilterInfo[trackEndTime[iTrack] - 1].stateVec[
                                     trackedFeatIndx[iTrack, trackEndTime[iTrack] - 1] - 1, probDim:2 * probDim]

        else:

            noiseStd[iTrack] = 1

    return EstimTrackTypeParamRDS(trackType, xyzVelS, xyzVelE, noiseStd, trackCentre, trackMeanDispMag, errFlag)


def getSearchRegionRDS(xyzVelS, xyzVelE, brownStd, trackType, undetBrownStd, timeWindow, brownStdMult, linStdMult,
                       timeReachConfB, timeReachConfL, minSearchRadius, maxSearchRadius, useLocalDensity,
                       closestDistScale, maxStdMult, nnDistLinkedFeat, nnWindow, trackStartTime, trackEndTime, probDim,
                       resLimit, brownScaling, linScaling, linearMotion):
    numTracks = brownStd.shape[0]

    longVecS = scipy.zeros((probDim, timeWindow, numTracks))
    longVecE = scipy.zeros((probDim, timeWindow, numTracks))
    longRedVecS = scipy.zeros((probDim, timeWindow, numTracks))
    longRedVecE = scipy.zeros((probDim, timeWindow, numTracks))
    shortVecS = scipy.zeros((probDim, timeWindow, numTracks))
    shortVecE = scipy.zeros((probDim, timeWindow, numTracks))
    longVecSMS = scipy.zeros((probDim, timeWindow, numTracks))
    longVecEMS = scipy.zeros((probDim, timeWindow, numTracks))
    longRedVecSMS = scipy.zeros((probDim, timeWindow, numTracks))
    longRedVecEMS = scipy.zeros((probDim, timeWindow, numTracks))
    shortVecSMS = scipy.zeros((probDim, timeWindow, numTracks))
    shortVecEMS = scipy.zeros((probDim, timeWindow, numTracks))

    if probDim == 3:
        shortVecS3D = scipy.zeros((probDim, timeWindow, numTracks))
        shortVecE3D = scipy.zeros((probDim, timeWindow, numTracks))
        shortVecS3DMS = scipy.zeros((probDim, timeWindow, numTracks))
        shortVecE3DMS = scipy.zeros((probDim, timeWindow, numTracks))

    else:
        shortVecS3D = scipy.array([])
        shortVecE3D = scipy.array([])
        shortVecS3DMS = scipy.array([])
        shortVecE3DMS = scipy.array([])

    sqrtDim = scipy.sqrt(probDim)

    timeScalingLin = scipy.concatenate((scipy.arange(1, timeReachConfL + 1) ** linScaling[0],
                                        (timeReachConfL ** linScaling[0]) * (
                                            scipy.arange(2, timeWindow - timeReachConfL + 2) ** linScaling[1])), axis=0)

    timeScalingBrown = scipy.concatenate((scipy.arange(1, timeReachConfB + 1) ** brownScaling[0],
                                          (timeReachConfB ** brownScaling[0]) * (
                                              scipy.arange(2, timeWindow - timeReachConfB + 2) ** brownScaling[1])),
                                         axis=0)

    maxSearchRadius *= timeScalingBrown

    minSearchRadiusMS = scipy.maximum(minSearchRadius, resLimit)
    maxSearchRadiusMS = scipy.maximum(maxSearchRadius, resLimit)

    trackStartTime = scipy.reshape(trackStartTime, (trackStartTime.shape[0], 1))
    trackEndTime = scipy.reshape(trackEndTime, (trackEndTime.shape[0], 1))

    windowLimS = scipy.amin(scipy.concatenate((trackStartTime + nnWindow, trackEndTime), axis=1), axis=1)
    windowLimE = scipy.amax(scipy.concatenate((trackEndTime - nnWindow, trackStartTime), axis=1), axis=1)
    nnDistTracksS = scipy.zeros(numTracks)
    nnDistTracksE = scipy.zeros(numTracks)
    for iTrack in range(numTracks):
        nnDistTracksS[iTrack] = scipy.nanmin(nnDistLinkedFeat[iTrack, trackStartTime[iTrack, 0] - 1:windowLimS[iTrack]])
        nnDistTracksE[iTrack] = scipy.nanmin(nnDistLinkedFeat[iTrack, windowLimE[iTrack] - 1:trackEndTime[iTrack, 0]])

    for iTrack in range(numTracks):

        # noinspection PyNoneFunctionAssignment,PyNoneFunctionAssignment,PyNoneFunctionAssignment,PyNoneFunctionAssignment
        if trackType[iTrack] == 1:

            velDriftS = scipy.mat(xyzVelS[iTrack, :]).transpose()
            velMagS = scipy.sqrt(velDriftS.transpose() * velDriftS)[0][0]
            directionMotionS = velDriftS / velMagS
            velDriftE = scipy.mat(xyzVelE[iTrack, :]).transpose()
            velMagE = scipy.sqrt(velDriftE.transpose() * velDriftE)[0][0]
            directionMotionE = velDriftE / velMagE

            if probDim == 2:
                perpendicularS = scipy.array([[-directionMotionS[1, 0], directionMotionS[0, 0]]]).transpose()
                perpendicularE = scipy.array([[-directionMotionE[1, 0], directionMotionE[0, 0]]]).transpose()
            else:
                perpendicularS = scipy.mat([-directionMotionS[1, 0], directionMotionS[0, 0], 0]).transpose()
                perpendicularS = perpendicularS / scipy.sqrt(perpendicularS.transpose() * perpendicularS)[0][0]
                perpendicular3DS = scipy.cross(directionMotionS.transpose(), perpendicularS.transpose()).transpose()
                perpendicularE = scipy.mat([-directionMotionE[1, 0], directionMotionE[0, 0], 0]).transpose()
                perpendicularE = perpendicularE / scipy.sqrt(perpendicularE.transpose() * perpendicularE)[0][0]
                perpendicular3DE = scipy.cross(directionMotionE.transpose(), perpendicularE.transpose()).transpose()

            dispDrift1FS = velMagS * timeScalingLin
            dispDrift1FE = velMagE * timeScalingLin

            dispBrown1 = brownStd[iTrack] * timeScalingBrown
            dispBrown1 = scipy.reshape(dispBrown1, (1, dispBrown1.shape[0]))

            brownStdMultModS = scipy.reshape(brownStdMult, (1, brownStdMult.shape[0]))
            brownStdMultModE = brownStdMultModS.copy()

            if useLocalDensity:
                ratioDist2Std = scipy.tile(scipy.array([[nnDistTracksS[iTrack] / closestDistScale]]),
                                           (1, timeWindow)) / dispBrown1
                ratioDist2Std[ratioDist2Std > maxStdMult] = maxStdMult

                brownStdMultModS = scipy.amax(scipy.concatenate((brownStdMultModS, ratioDist2Std), axis=0), axis=0)
                brownStdMultModS = scipy.reshape(brownStdMultModS, (1, brownStdMultModS.shape[0]))

                ratioDist2Std = scipy.tile(scipy.array([[nnDistTracksE[iTrack] / closestDistScale]]),
                                           (1, timeWindow)) / dispBrown1
                ratioDist2Std[ratioDist2Std > maxStdMult] = maxStdMult

                brownStdMultModE = scipy.amax(scipy.concatenate((brownStdMultModE, ratioDist2Std), axis=0), axis=0)
                brownStdMultModE = scipy.reshape(brownStdMultModE, (1, brownStdMultModE.shape[0]))

            longVec1FS = scipy.mat((scipy.tile((scipy.reshape(linStdMult, (1, linStdMult.shape[0])) * dispDrift1FS),
                                               (probDim, 1)) + scipy.tile(
                (scipy.array(brownStdMult.transpose() * dispBrown1 * sqrtDim)), (probDim, 1))) * scipy.tile(
                scipy.array(directionMotionS), (1, timeWindow)))
            longVecFSMag = scipy.sqrt(scipy.diag(longVec1FS.transpose() * longVec1FS)).transpose()
            longVecFSDir = longVec1FS / scipy.tile(scipy.reshape(longVecFSMag, (1, longVecFSMag.shape[0])),
                                                   (probDim, 1))

            longVec1FE = scipy.mat((scipy.tile((scipy.reshape(linStdMult, (1, linStdMult.shape[0])) * dispDrift1FE),
                                               (probDim, 1)) + scipy.tile(
                (scipy.array(brownStdMult.transpose() * dispBrown1 * sqrtDim)), (probDim, 1))) * scipy.tile(
                scipy.array(directionMotionE), (1, timeWindow)))
            longVecFEMag = scipy.sqrt(scipy.diag(longVec1FE.transpose() * longVec1FE)).transpose()
            longVecFEDir = longVec1FE / scipy.tile(scipy.reshape(longVecFEMag, (1, longVecFEMag.shape[0])),
                                                   (probDim, 1))

            if linearMotion == 1:
                longVec1BS = scipy.mat(scipy.tile((brownStdMultModS * dispBrown1 * sqrtDim), (probDim, 1)) * scipy.tile(
                    scipy.array(directionMotionS), (1, timeWindow)))
                longVecBSMag = scipy.sqrt(scipy.diag(longVec1BS.transpose() * longVec1BS)).transpose()
                longVecBSDir = longVec1BS / scipy.tile(scipy.reshape(longVecBSMag, (1, longVecBSMag.shape[0])),
                                                       (probDim, 1))

                longVec1BE = scipy.mat(scipy.tile((brownStdMultModE * dispBrown1 * sqrtDim), (probDim, 1)) * scipy.tile(
                    scipy.array(directionMotionE), (1, timeWindow)))
                longVecBEMag = scipy.sqrt(scipy.diag(longVec1BE.transpose() * longVec1BE)).transpose()
                longVecBEDir = longVec1BE / scipy.tile(scipy.reshape(longVecBEMag, (1, longVecBEMag.shape[0])),
                                                       (probDim, 1))

            shortVecS1 = scipy.mat(scipy.tile((brownStdMultModS * dispBrown1 * sqrtDim), (probDim, 1)) * scipy.tile(
                scipy.array(perpendicularS), (1, timeWindow)))
            shortVecSMag = scipy.sqrt(scipy.diag(shortVecS1.transpose() * shortVecS1)).transpose()
            shortVecSDir = shortVecS1 / scipy.tile(scipy.reshape(shortVecSMag, (1, shortVecSMag.shape[0])),
                                                   (probDim, 1))

            shortVecE1 = scipy.mat(scipy.tile((brownStdMultModE * dispBrown1 * sqrtDim), (probDim, 1)) * scipy.tile(
                scipy.array(perpendicularE), (1, timeWindow)))
            shortVecEMag = scipy.sqrt(scipy.diag(shortVecE1.transpose() * shortVecE1)).transpose()
            shortVecEDir = shortVecE1 / scipy.tile(scipy.reshape(shortVecEMag, (1, shortVecEMag.shape[0])),
                                                   (probDim, 1))

            longVecSMagTmp = scipy.amax(scipy.concatenate(
                (scipy.reshape(longVecFSMag, (1, longVecFSMag.shape[0])), scipy.tile(minSearchRadius, (1, timeWindow))),
                axis=0), axis=0)
            longVec1FS = scipy.tile(scipy.reshape(longVecSMagTmp, (1, longVecSMagTmp.shape[0])),
                                    (probDim, 1)) * scipy.array(longVecFSDir)
            longVecSMagTmp = scipy.amax(scipy.concatenate((scipy.reshape(longVecFSMag, (1, longVecFSMag.shape[0])),
                                                           scipy.tile(minSearchRadiusMS, (1, timeWindow))), axis=0),
                                        axis=0)
            longVec1MSFS = scipy.tile(scipy.reshape(longVecSMagTmp, (1, longVecSMagTmp.shape[0])),
                                      (probDim, 1)) * scipy.array(longVecFSDir)

            longVecEMagTmp = scipy.amax(scipy.concatenate(
                (scipy.reshape(longVecFEMag, (1, longVecFEMag.shape[0])), scipy.tile(minSearchRadius, (1, timeWindow))),
                axis=0), axis=0)
            longVec1FE = scipy.tile(scipy.reshape(longVecEMagTmp, (1, longVecEMagTmp.shape[0])),
                                    (probDim, 1)) * scipy.array(longVecFEDir)
            longVecEMagTmp = scipy.amax(scipy.concatenate((scipy.reshape(longVecFEMag, (1, longVecFEMag.shape[0])),
                                                           scipy.tile(minSearchRadiusMS, (1, timeWindow))), axis=0),
                                        axis=0)
            longVec1MSFE = scipy.tile(scipy.reshape(longVecEMagTmp, (1, longVecEMagTmp.shape[0])),
                                      (probDim, 1)) * scipy.array(longVecFEDir)

            if linearMotion == 1:
                longVecSMagTmp = scipy.amax(scipy.concatenate((scipy.reshape(longVecBSMag, (1, longVecBSMag.shape[0])),
                                                               scipy.tile(minSearchRadius, (1, timeWindow))), axis=0),
                                            axis=0)
                longVecSMagTmp = scipy.minimum(longVecSMagTmp, maxSearchRadius)
                longVec1BS = scipy.tile(scipy.reshape(longVecSMagTmp, (1, longVecSMagTmp.shape[0])),
                                        (probDim, 1)) * scipy.array(longVecBSDir)

                longVecSMagTmp = scipy.amax(scipy.concatenate((scipy.reshape(longVecBSMag, (1, longVecBSMag.shape[0])),
                                                               scipy.tile(minSearchRadiusMS, (1, timeWindow))), axis=0),
                                            axis=0)
                longVecSMagTmp = scipy.minimum(longVecSMagTmp, maxSearchRadiusMS)
                longVec1MSBS = scipy.tile(scipy.reshape(longVecSMagTmp, (1, longVecSMagTmp.shape[0])),
                                          (probDim, 1)) * scipy.array(longVecBSDir)

                longVecEMagTmp = scipy.amax(scipy.concatenate((scipy.reshape(longVecBEMag, (1, longVecBEMag.shape[0])),
                                                               scipy.tile(minSearchRadius, (1, timeWindow))), axis=0),
                                            axis=0)
                longVecEMagTmp = scipy.minimum(longVecEMagTmp, maxSearchRadius)
                longVec1BE = scipy.tile(scipy.reshape(longVecEMagTmp, (1, longVecEMagTmp.shape[0])),
                                        (probDim, 1)) * scipy.array(longVecBEDir)

                longVecEMagTmp = scipy.amax(scipy.concatenate((scipy.reshape(longVecBEMag, (1, longVecBEMag.shape[0])),
                                                               scipy.tile(minSearchRadiusMS, (1, timeWindow))), axis=0),
                                            axis=0)
                longVecEMagTmp = scipy.minimum(longVecEMagTmp, maxSearchRadiusMS)
                longVec1MSBE = scipy.tile(scipy.reshape(longVecEMagTmp, (1, longVecEMagTmp.shape[0])),
                                          (probDim, 1)) * scipy.array(longVecBEDir)

            shortVecSMagTmp = scipy.amax(scipy.concatenate(
                (scipy.reshape(shortVecSMag, (1, shortVecSMag.shape[0])), scipy.tile(minSearchRadius, (1, timeWindow))),
                axis=0), axis=0)
            shortVecSMagTmp = scipy.minimum(shortVecSMagTmp, maxSearchRadius)
            shortVecS1 = scipy.tile(scipy.reshape(shortVecSMagTmp, (1, shortVecSMagTmp.shape[0])),
                                    (probDim, 1)) * scipy.array(shortVecSDir)

            shortVecSMagTmpMS = scipy.amax(scipy.concatenate((scipy.reshape(shortVecSMag, (1, shortVecSMag.shape[0])),
                                                              scipy.tile(minSearchRadiusMS, (1, timeWindow))), axis=0),
                                           axis=0)
            shortVecSMagTmpMS = scipy.minimum(shortVecSMagTmpMS, maxSearchRadiusMS)
            shortVecS1MS = scipy.tile(scipy.reshape(shortVecSMagTmpMS, (1, shortVecSMagTmpMS.shape[0])),
                                      (probDim, 1)) * scipy.array(shortVecSDir)

            shortVecEMagTmp = scipy.amax(scipy.concatenate(
                (scipy.reshape(shortVecEMag, (1, shortVecEMag.shape[0])), scipy.tile(minSearchRadius, (1, timeWindow))),
                axis=0), axis=0)
            shortVecEMagTmp = scipy.minimum(shortVecEMagTmp, maxSearchRadius)
            shortVecE1 = scipy.tile(scipy.reshape(shortVecEMagTmp, (1, shortVecEMagTmp.shape[0])),
                                    (probDim, 1)) * scipy.array(shortVecEDir)

            shortVecEMagTmpMS = scipy.amax(scipy.concatenate((scipy.reshape(shortVecEMag, (1, shortVecEMag.shape[0])),
                                                              scipy.tile(minSearchRadiusMS, (1, timeWindow))), axis=0),
                                           axis=0)
            shortVecEMagTmpMS = scipy.minimum(shortVecEMagTmpMS, maxSearchRadiusMS)
            shortVecE1MS = scipy.tile(scipy.reshape(shortVecEMagTmpMS, (1, shortVecEMagTmpMS.shape[0])),
                                      (probDim, 1)) * scipy.array(shortVecEDir)

            longVecS[:, :, iTrack] = longVec1FS
            longVecE[:, :, iTrack] = longVec1FE
            shortVecS[:, :, iTrack] = shortVecS1
            shortVecE[:, :, iTrack] = shortVecE1
            longVecSMS[:, :, iTrack] = longVec1MSFS
            longVecEMS[:, :, iTrack] = longVec1MSFE
            shortVecSMS[:, :, iTrack] = shortVecS1MS
            shortVecEMS[:, :, iTrack] = shortVecE1MS

            if linearMotion == 1:
                longRedVecS[:, :, iTrack] = longVec1BS
                longRedVecE[:, :, iTrack] = longVec1BE
                longRedVecSMS[:, :, iTrack] = longVec1MSBS
                longRedVecEMS[:, :, iTrack] = longVec1MSBE
            elif linearMotion == 2:
                longRedVecS[:, :, iTrack] = longVec1FS
                longRedVecE[:, :, iTrack] = longVec1FE
                longRedVecSMS[:, :, iTrack] = longVec1MSFS
                longRedVecEMS[:, :, iTrack] = longVec1MSFE

            if probDim == 3:
                shortVecS13D = scipy.tile(scipy.reshape(shortVecSMagTmp, (1, shortVecSMagTmp.shape[0])),
                                          (probDim, 1)) * scipy.tile(scipy.array(perpendicular3DS), (1, timeWindow))
                shortVecE13D = scipy.tile(scipy.reshape(shortVecEMagTmp, (1, shortVecEMagTmp.shape[0])),
                                          (probDim, 1)) * scipy.tile(scipy.array(perpendicular3DE), (1, timeWindow))
                shortVecS3D[:, :, iTrack] = shortVecS13D
                shortVecE3D[:, :, iTrack] = shortVecE13D

                shortVecS13DMS = scipy.tile(scipy.reshape(shortVecSMagTmpMS, (1, shortVecSMagTmpMS.shape[0])),
                                            (probDim, 1)) * scipy.tile(scipy.array(perpendicular3DS), (1, timeWindow))
                shortVecE13DMS = scipy.tile(scipy.reshape(shortVecEMagTmpMS, (1, shortVecEMagTmpMS.shape[0])),
                                            (probDim, 1)) * scipy.tile(scipy.array(perpendicular3DE), (1, timeWindow))
                shortVecS3DMS[:, :, iTrack] = shortVecS13DMS
                shortVecE3DMS[:, :, iTrack] = shortVecE13DMS


        elif trackType[iTrack] == 0:

            if probDim == 2:
                directionMotion = scipy.array([[1, 0]]).transpose()
                perpendicular = scipy.array([[0, 1]]).transpose()
            else:
                directionMotion = scipy.array([[1, 0, 0]]).transpose()
                perpendicular = scipy.array([[0, 1, 0]]).transpose()
                perpendicular3D = scipy.array([[0, 0, 1]]).transpose()

            dispBrown1 = brownStd[iTrack] * timeScalingBrown

            brownStdMultModS = scipy.reshape(brownStdMult, (1, brownStdMult.shape[0]))
            brownStdMultModE = brownStdMultModS.copy()

            if useLocalDensity:
                ratioDist2Std = scipy.tile(nnDistTracksS[iTrack] / closestDistScale, (1, timeWindow)) / dispBrown1
                ratioDist2Std[ratioDist2Std > maxStdMult] = maxStdMult

                brownStdMultModS = scipy.amax(scipy.concatenate((brownStdMultModS, ratioDist2Std), axis=0), axis=0)

                ratioDist2Std = scipy.tile(nnDistTracksE[iTrack] / closestDistScale, (1, timeWindow)) / dispBrown1
                ratioDist2Std[ratioDist2Std > maxStdMult] = maxStdMult

                brownStdMultModE = scipy.amax(scipy.concatenate((brownStdMultModE, ratioDist2Std), axis=0), axis=0)

            longVecS1 = scipy.mat(
                scipy.tile((scipy.reshape(brownStdMultModS, (1, brownStdMultModS.shape[0])) * dispBrown1 * sqrtDim),
                           (probDim, 1)) * scipy.tile(directionMotion, (1, timeWindow)))

            longVecE1 = scipy.mat(
                scipy.tile((scipy.reshape(brownStdMultModE, (1, brownStdMultModE.shape[0])) * dispBrown1 * sqrtDim),
                           (probDim, 1)) * scipy.tile(directionMotion, (1, timeWindow)))

            shortVecS1 = scipy.mat(
                scipy.tile((scipy.reshape(brownStdMultModS, (1, brownStdMultModS.shape[0])) * dispBrown1 * sqrtDim),
                           (probDim, 1)) * scipy.tile(perpendicular, (1, timeWindow)))

            shortVecE1 = scipy.mat(
                scipy.tile((scipy.reshape(brownStdMultModE, (1, brownStdMultModE.shape[0])) * dispBrown1 * sqrtDim),
                           (probDim, 1)) * scipy.tile(perpendicular, (1, timeWindow)))

            vecMag = scipy.sqrt(scipy.diag(longVecS1.transpose() * longVecS1)).transpose()
            vecMag = scipy.reshape(vecMag, (1, vecMag.shape[0]))
            longVecDir = longVecS1 / scipy.tile(vecMag, (probDim, 1))
            shortVecDir = shortVecS1 / scipy.tile(vecMag, (probDim, 1))

            vecMagTmp = scipy.amax(scipy.concatenate((vecMag, scipy.tile(minSearchRadius, (1, timeWindow))), axis=0),
                                   axis=0)
            vecMagTmp = scipy.minimum(vecMagTmp, maxSearchRadius)
            vecMagTmp = scipy.reshape(vecMagTmp, (1, vecMagTmp.shape[0]))

            vecMagTmpMS = scipy.amax(
                scipy.concatenate((vecMag, scipy.tile(minSearchRadiusMS, (1, timeWindow))), axis=0), axis=0)
            vecMagTmpMS = scipy.minimum(vecMagTmpMS, maxSearchRadiusMS)
            vecMagTmpMS = scipy.reshape(vecMagTmpMS, (1, vecMagTmpMS.shape[0]))

            longVecS1 = scipy.tile(vecMagTmp, (probDim, 1)) * scipy.array(longVecDir)
            shortVecS1 = scipy.tile(vecMagTmp, (probDim, 1)) * scipy.array(shortVecDir)

            longVecS1MS = scipy.tile(vecMagTmpMS, (probDim, 1)) * scipy.array(longVecDir)
            shortVecS1MS = scipy.tile(vecMagTmpMS, (probDim, 1)) * scipy.array(shortVecDir)

            if probDim == 3:
                shortVecS13D = scipy.tile(vecMagTmp, (probDim, 1)) * scipy.tile(perpendicular3D, (1, timeWindow))
                shortVecS3D[:, :, iTrack] = shortVecS13D

                shortVecS13DMS = scipy.tile(vecMagTmpMS, (probDim, 1)) * scipy.tile(perpendicular3D, (1, timeWindow))
                shortVecS3DMS[:, :, iTrack] = shortVecS13DMS

            vecMag = scipy.sqrt(scipy.diag(longVecE1.transpose() * longVecE1)).transpose()
            vecMag = scipy.reshape(vecMag, (1, vecMag.shape[0]))
            longVecDir = longVecE1 / scipy.tile(vecMag, (probDim, 1))
            shortVecDir = shortVecE1 / scipy.tile(vecMag, (probDim, 1))

            vecMagTmp = scipy.amax(scipy.concatenate((vecMag, scipy.tile(minSearchRadius, (1, timeWindow))), axis=0),
                                   axis=0)
            vecMagTmp = scipy.minimum(vecMagTmp, maxSearchRadius)
            vecMagTmp = scipy.reshape(vecMagTmp, (1, vecMagTmp.shape[0]))

            vecMagTmpMS = scipy.amax(
                scipy.concatenate((vecMag, scipy.tile(minSearchRadiusMS, (1, timeWindow))), axis=0), axis=0)
            vecMagTmpMS = scipy.minimum(vecMagTmpMS, maxSearchRadiusMS)
            vecMagTmpMS = scipy.reshape(vecMagTmpMS, (1, vecMagTmpMS.shape[0]))

            longVecE1 = scipy.tile(vecMagTmp, (probDim, 1)) * scipy.array(longVecDir)
            shortVecE1 = scipy.tile(vecMagTmp, (probDim, 1)) * scipy.array(shortVecDir)

            longVecE1MS = scipy.tile(vecMagTmpMS, (probDim, 1)) * scipy.array(longVecDir)
            shortVecE1MS = scipy.tile(vecMagTmpMS, (probDim, 1)) * scipy.array(shortVecDir)

            if probDim == 3:
                shortVecE13D = scipy.tile(vecMagTmp, (probDim, 1)) * scipy.tile(perpendicular3D, (1, timeWindow))
                shortVecE3D[:, :, iTrack] = shortVecE13D

                shortVecE13DMS = scipy.tile(vecMagTmpMS, (probDim, 1)) * scipy.tile(perpendicular3D, (1, timeWindow))
                shortVecE3DMS[:, :, iTrack] = shortVecE13DMS

            longVecS[:, :, iTrack] = longVecS1
            longVecE[:, :, iTrack] = longVecE1
            longRedVecS[:, :, iTrack] = longVecS1
            longRedVecE[:, :, iTrack] = longVecE1
            shortVecS[:, :, iTrack] = shortVecS1
            shortVecE[:, :, iTrack] = shortVecE1
            longVecSMS[:, :, iTrack] = longVecS1MS
            longVecEMS[:, :, iTrack] = longVecE1MS
            longRedVecSMS[:, :, iTrack] = longVecS1MS
            longRedVecEMS[:, :, iTrack] = longVecE1MS
            shortVecSMS[:, :, iTrack] = shortVecS1MS
            shortVecEMS[:, :, iTrack] = shortVecE1MS

        else:

            if probDim == 2:
                directionMotion = scipy.array([[1, 0]]).transpose()
                perpendicular = scipy.array([[0, 1]]).transpose()
            else:
                directionMotion = scipy.array([[1, 0, 0]]).transpose()
                perpendicular = scipy.array([[0, 1, 0]]).transpose()
                perpendicular3D = scipy.array([[0, 0, 1]]).transpose()

            if brownStd[iTrack] == 1:
                dispBrown1 = undetBrownStd * timeScalingBrown
            else:
                dispBrown1 = brownStd[iTrack] * timeScalingBrown

            brownStdMultModS = scipy.reshape(brownStdMult, (1, brownStdMult.shape[0]))
            brownStdMultModE = brownStdMultModS.copy()

            if useLocalDensity:
                ratioDist2Std = scipy.tile(nnDistTracksS[iTrack] / closestDistScale, (1, timeWindow)) / dispBrown1
                ratioDist2Std[ratioDist2Std > maxStdMult] = maxStdMult

                brownStdMultModS = scipy.amax(scipy.concatenate((brownStdMultModS, ratioDist2Std), axis=0), axis=0)
                brownStdMultModS = scipy.reshape(brownStdMultModS, (1, brownStdMultModS.shape[0]))

                ratioDist2Std = scipy.tile(nnDistTracksE[iTrack] / closestDistScale, (1, timeWindow)) / dispBrown1
                ratioDist2Std[ratioDist2Std > maxStdMult] = maxStdMult

                brownStdMultModE = scipy.amax(scipy.concatenate((brownStdMultModE, ratioDist2Std), axis=0), axis=0)
                brownStdMultModE = scipy.reshape(brownStdMultModE, (1, brownStdMultModE.shape[0]))

            longVecS1 = scipy.mat(
                scipy.tile((brownStdMultModS * dispBrown1 * sqrtDim), (probDim, 1)) * scipy.tile(directionMotion,
                                                                                                 (1, timeWindow)))

            longVecE1 = scipy.mat(
                scipy.tile((brownStdMultModE * dispBrown1 * sqrtDim), (probDim, 1)) * scipy.tile(directionMotion,
                                                                                                 (1, timeWindow)))

            shortVecS1 = scipy.mat(
                scipy.tile((brownStdMultModS * dispBrown1 * sqrtDim), (probDim, 1)) * scipy.tile(perpendicular,
                                                                                                 (1, timeWindow)))

            shortVecE1 = scipy.mat(
                scipy.tile((brownStdMultModE * dispBrown1 * sqrtDim), (probDim, 1)) * scipy.tile(perpendicular,
                                                                                                 (1, timeWindow)))

            vecMag = scipy.sqrt(scipy.diag(longVecS1.transpose() * longVecS1)).transpose()
            vecMag = scipy.reshape(vecMag, (1, vecMag.shape[0]))
            longVecDir = longVecS1 / scipy.tile(vecMag, (probDim, 1))
            shortVecDir = shortVecS1 / scipy.tile(vecMag, (probDim, 1))

            vecMagTmp = scipy.amax(scipy.concatenate((vecMag, scipy.tile(minSearchRadius, (1, timeWindow))), axis=0),
                                   axis=0)
            vecMagTmp = scipy.minimum(vecMagTmp, maxSearchRadius)
            vecMagTmp = scipy.reshape(vecMagTmp, (1, vecMagTmp.shape[0]))

            vecMagTmpMS = scipy.amax(scipy.concatenate((vecMag, scipy.tile(minSearchRadius, (1, timeWindow))), axis=0),
                                     axis=0)
            vecMagTmpMS = scipy.minimum(vecMagTmpMS, maxSearchRadiusMS)
            vecMagTmpMS = scipy.reshape(vecMagTmpMS, (1, vecMagTmpMS.shape[0]))

            longVecS1 = scipy.tile(vecMagTmp, (probDim, 1)) * scipy.array(longVecDir)
            shortVecS1 = scipy.tile(vecMagTmp, (probDim, 1)) * scipy.array(shortVecDir)

            longVecS1MS = scipy.tile(vecMagTmpMS, (probDim, 1)) * scipy.array(longVecDir)
            shortVecS1MS = scipy.tile(vecMagTmpMS, (probDim, 1)) * scipy.array(shortVecDir)

            if probDim == 3:
                shortVecS13D = scipy.tile(vecMagTmp, (probDim, 1)) * scipy.tile(perpendicular3D, (1, timeWindow))
                shortVecS3D[:, :, iTrack] = shortVecS13D

                shortVecS13DMS = scipy.tile(vecMagTmpMS, (probDim, 1)) * scipy.tile(perpendicular3D, (1, timeWindow))
                shortVecS3DMS[:, :, iTrack] = shortVecS13DMS

            vecMag = scipy.sqrt(scipy.diag(longVecE1.transpose() * longVecE1)).transpose()
            vecMag = scipy.reshape(vecMag, (1, vecMag.shape[0]))
            longVecDir = longVecE1 / scipy.tile(vecMag, (probDim, 1))
            shortVecDir = shortVecE1 / scipy.tile(vecMag, (probDim, 1))

            vecMagTmp = scipy.amax(scipy.concatenate((vecMag, scipy.tile(minSearchRadius, (1, timeWindow))), axis=0),
                                   axis=0)
            vecMagTmp = scipy.minimum(vecMagTmp, maxSearchRadius)
            vecMagTmp = scipy.reshape(vecMagTmp, (1, vecMagTmp.shape[0]))

            vecMagTmpMS = scipy.amax(
                scipy.concatenate((vecMag, scipy.tile(minSearchRadiusMS, (1, timeWindow))), axis=0), axis=0)
            vecMagTmpMS = scipy.minimum(vecMagTmpMS, maxSearchRadiusMS)
            vecMagTmpMS = scipy.reshape(vecMagTmpMS, (1, vecMagTmpMS.shape[0]))

            longVecE1 = scipy.tile(vecMagTmp, (probDim, 1)) * scipy.array(longVecDir)
            shortVecE1 = scipy.tile(vecMagTmp, (probDim, 1)) * scipy.array(shortVecDir)

            longVecE1MS = scipy.tile(vecMagTmpMS, (probDim, 1)) * scipy.array(longVecDir)
            shortVecE1MS = scipy.tile(vecMagTmpMS, (probDim, 1)) * scipy.array(shortVecDir)

            if probDim == 3:
                shortVecE13D = scipy.tile(vecMagTmp, (probDim, 1)) * scipy.tile(perpendicular3D, (1, timeWindow))
                shortVecE3D[:, :, iTrack] = shortVecE13D

                shortVecE13DMS = scipy.tile(vecMagTmpMS, (probDim, 1)) * scipy.tile(perpendicular3D, (1, timeWindow))
                shortVecE3DMS[:, :, iTrack] = shortVecE13DMS

            longVecS[:, :, iTrack] = longVecS1
            longVecE[:, :, iTrack] = longVecE1
            longRedVecS[:, :, iTrack] = longVecS1
            longRedVecE[:, :, iTrack] = longVecE1
            shortVecS[:, :, iTrack] = shortVecS1
            shortVecE[:, :, iTrack] = shortVecE1
            longVecSMS[:, :, iTrack] = longVecS1MS
            longVecEMS[:, :, iTrack] = longVecE1MS
            longRedVecSMS[:, :, iTrack] = longVecS1MS
            longRedVecEMS[:, :, iTrack] = longVecE1MS
            shortVecSMS[:, :, iTrack] = shortVecS1MS
            shortVecEMS[:, :, iTrack] = shortVecE1MS

    return GetSearchRegionRDS(longVecS, longVecE, shortVecS, shortVecE, shortVecS3D, shortVecE3D, longVecSMS,
                              longVecEMS, shortVecSMS, shortVecEMS, shortVecS3DMS, shortVecE3DMS, longRedVecS,
                              longRedVecE, longRedVecSMS, longRedVecEMS)
