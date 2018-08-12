from copy import deepcopy

import scipy
from matplotlib import pylab as lab
from matplotlib.widgets import Slider

from Utilities.trackConstruction.constructTrackMatFromCoords import constructTrackMatFromCoords
from Utrack.utilityClasses import TracksFinal

__author__ = 'edwardharry'


def visuPoints2D(coords):
    def getLimits(i):
        tmp = scipy.concatenate([coords[iFrame][i] for iFrame in range(len(coords))], axis=0)

        return [scipy.nanmin(tmp[:, 0]), scipy.nanmax(tmp[:, 0])], [scipy.nanmin(tmp[:, 1]), scipy.nanmax(tmp[:, 1])]

    lim1X, lim1Y = getLimits(0)
    lim2X, lim2Y = getLimits(1)

    lab.subplot(121)
    lab.subplot(122)
    lab.subplots_adjust(bottom=0.25)

    ax1 = lab.subplot(121)
    l1, = lab.plot(coords[0][0][:, 0], coords[0][0][:, 1], 'r*')
    ax1.set_xlim(lim1X)
    ax1.set_ylim(lim1Y)

    ax2 = lab.subplot(122)
    l2, = lab.plot(coords[0][1][:, 0], coords[0][1][:, 1], 'g*')
    ax2.set_xlim(lim2X)
    ax2.set_ylim(lim2Y)

    sTime = Slider(lab.axes([0.25, 0.1, 0.65, 0.03], axisbg='lightgoldenrodyellow'), 'Time', 0, len(coords), valinit=0,
                   valfmt='%0.0f')

    def update(val):
        time = int(sTime.val)

        if time < 0 or time >= len(coords):
            return

        l1.set_xdata(coords[time][0][:, 0])
        l1.set_ydata(coords[time][0][:, 1])

        l2.set_xdata(coords[time][1][:, 0])
        l2.set_ydata(coords[time][1][:, 1])

        lab.draw()

    sTime.on_changed(update)

    lab.show()


def visuTracks2D(trackedFeatureInfo, image=None, series=None):
    if image is not None:
        import javabridge as jv, bioformats as bf
        jv.start_vm(class_path=bf.JARS, max_heap_size='8G')
        rdr = bf.ImageReader(image, perform_init=True)
    else:
        rdr = None

    def ReadIm():
        retVal = None
        global globalTime
        s = series if series is not None else 0
        if rdr is not None:
            retVal = rdr.read(z=0, t=globalTime, series=s)
        return retVal

    extendedColours = lab.cm.Set3(scipy.linspace(0, 1, 32))
    isMat = True

    if type(trackedFeatureInfo) is list:
        numTracks = len(trackedFeatureInfo)

        if type(trackedFeatureInfo[0]) is TracksFinal:
            tmp = scipy.concatenate([trackedFeatureInfo[iFrame].seqOfEvents for iFrame in range(numTracks)], axis=0)
            numTimePoints = int(scipy.amax(tmp[:, 0]))
            del tmp
            isMat = False

        else:
            trackedFeatureInfo = constructTrackMatFromCoords(trackedFeatureInfo)

            if trackedFeatureInfo is None:
                return

            if type(trackedFeatureInfo) is tuple:
                visuTracks2D(trackedFeatureInfo[0])
                trackedFeatureInfo = trackedFeatureInfo[1]

            numTracks, numTimePoints = trackedFeatureInfo.shape
            numTimePoints /= 8

    else:
        numTracks, numTimePoints = trackedFeatureInfo.shape
        numTimePoints /= 8

    if not isMat:
        inputStructure = deepcopy(trackedFeatureInfo)
        numSegments = scipy.zeros(numTracks).astype(scipy.int64)

        for i in range(numTracks):
            numSegments[i] = inputStructure[i].tracksCoordAmpCG.shape[0]

        if scipy.amax(numSegments) == 1:
            mergeSplit = False
            trackStartRow = scipy.arange(numTracks).astype(scipy.int64)
            trackedFeatureInfo = scipy.nan * scipy.ones((numTracks, numTimePoints * 8))

            for i in range(numTracks):
                startTime = int(inputStructure[i].seqOfEvents[0, 0])
                endTime = int(inputStructure[i].seqOfEvents[-1, 0])
                trackedFeatureInfo[i, (8 * (startTime - 1)):(8 * endTime)] = inputStructure[i].tracksCoordAmpCG

        else:
            mergeSplit = True
            trackStartRow = scipy.zeros(numTracks).astype(scipy.int64)

            for iTrack in range(1, numTracks):
                trackStartRow[iTrack] = trackStartRow[iTrack - 1] + numSegments[iTrack - 1]

            trackedFeatureInfo = scipy.nan * scipy.ones((trackStartRow[-1] + numSegments[-1], 8 * numTimePoints))

            for i in range(numTracks):
                startTime = int(inputStructure[i].seqOfEvents[0, 0])
                endTime = int(inputStructure[i].seqOfEvents[-1, 0])
                trackedFeatureInfo[trackStartRow[i]:(trackStartRow[i] + numSegments[i]),
                (8 * (startTime - 1)):(8 * endTime)] = inputStructure[i].tracksCoordAmpCG

    else:
        inputStructure = None
        mergeSplit = False
        numSegments = scipy.ones(numTracks).astype(scipy.int64)
        trackStartRow = scipy.arange(numTracks).astype(scipy.int64)

    tracksX = trackedFeatureInfo[:, :-1:8].transpose()
    tracksY = trackedFeatureInfo[:, 1:-1:8].transpose()

    maxX = scipy.nanmax(tracksX.flatten())
    minX = scipy.nanmin(tracksX.flatten())
    maxY = scipy.nanmax(tracksY.flatten())
    minY = scipy.nanmin(tracksY.flatten())

    fig = lab.figure()
    ax = lab.subplot(111)
    lab.subplots_adjust(bottom=0.25)
    ax.set_xlim([minX, maxX])
    ax.set_ylim([minY, maxY])

    global globalTime
    globalTime = 0

    def plotTracks():
        lab.sca(ax)
        xLim = ax.get_xlim()
        yLim = ax.get_ylim()
        lab.cla()
        ax.set_autoscalex_on(False)
        ax.set_autoscaley_on(False)
        ax.set_xlim(xLim)
        ax.set_ylim(yLim)

        def Normalise(im):
            min = scipy.amin(scipy.amin(im, axis=0), axis=0)
            max = scipy.amax(scipy.amax(im, axis=0), axis=0)
            im = (im - min[scipy.newaxis, scipy.newaxis, :]) / ((max - min)[scipy.newaxis, scipy.newaxis, :])
            im = 1 - im
            return im

        im = ReadIm()
        if im is not None:
            im = Normalise(im)
            lab.imshow((2 ** 16) * scipy.ones(shape=im.shape[:2]), cmap=lab.cm.Greys, alpha=1, interpolation=None)
            lab.imshow(im[:, :, 0], cmap=lab.cm.Greens, alpha=.25, interpolation=None)
            lab.imshow(im[:, :, 1], cmap=lab.cm.Reds, alpha=.25, interpolation=None)
            lab.imshow(im[:, :, 2], cmap=lab.cm.Blues, alpha=.25, interpolation=None)
            lab.imshow(im[:, :, 3], cmap=lab.cm.Purples, alpha=.25, interpolation=None)

        global globalTime
        timeRange = scipy.array([scipy.maximum(0, globalTime - 10), globalTime])
        fig.suptitle("Frame = " + str(globalTime + 1) + " / " + str(int(numTimePoints)))

        tracksXP = tracksX[timeRange[0]:(timeRange[1] + 1), :]
        tracksYP = tracksY[timeRange[0]:(timeRange[1] + 1), :]

        for i in range(trackStartRow[-1] + numSegments[-1]):
            obsAvail = scipy.nonzero(~scipy.isnan(tracksXP[:, i]))[0]
            lab.plot(tracksXP[obsAvail, i], tracksYP[obsAvail, i], 'k:', linewidth=3)
            lab.plot(tracksXP[:, i], tracksYP[:, i], color=extendedColours[scipy.mod(i, 32), :], linewidth=3)
            lab.plot(tracksXP[-1, i], tracksYP[-1, i], color=extendedColours[scipy.mod(i, 32), :], marker='+',
                     markersize=10, markeredgewidth=5, linestyle='None')

        if mergeSplit:

            for iTrack in range(numTracks):
                seqOfEvents = inputStructure[iTrack].seqOfEvents
                indxSplit = scipy.nonzero(
                    scipy.logical_and(scipy.logical_and(seqOfEvents[:, 1] == 1, ~scipy.isnan(seqOfEvents[:, 3])),
                                      scipy.logical_and(seqOfEvents[:, 0] > timeRange[0] + 1,
                                                        seqOfEvents[:, 0] <= timeRange[1] + 1)))[0]
                indxMerge = scipy.nonzero(
                    scipy.logical_and(scipy.logical_and(seqOfEvents[:, 1] == 2, ~scipy.isnan(seqOfEvents[:, 3])),
                                      scipy.logical_and(seqOfEvents[:, 0] > timeRange[0] + 1,
                                                        seqOfEvents[:, 0] <= timeRange[1] + 1)))[0]

                for iSplit in indxSplit:
                    timeSplit = int(seqOfEvents[iSplit, 0])
                    rowS = trackStartRow[iTrack] + int(seqOfEvents[iSplit, 2]) - 1
                    rowSp = trackStartRow[iTrack] + int(seqOfEvents[iSplit, 3]) - 1
                    lab.plot([tracksX[timeSplit - 1, rowS], tracksX[timeSplit - 2, rowSp]],
                             [tracksY[timeSplit - 1, rowS], tracksY[timeSplit - 2, rowSp]], 'k-.', linewidth=3)

                for iMerge in indxMerge:
                    timeSplit = int(seqOfEvents[iMerge, 0])
                    rowE = trackStartRow[iTrack] + int(seqOfEvents[iSplit, 2]) - 1
                    rowM = trackStartRow[iTrack] + int(seqOfEvents[iSplit, 3]) - 1
                    lab.plot([tracksX[timeSplit - 2, rowE], tracksX[timeSplit - 1, rowM]],
                             [tracksY[timeSplit - 2, rowE], tracksY[timeSplit - 1, rowM]], 'k--', linewidth=3)

                indxStart = scipy.nonzero(
                    scipy.logical_and(scipy.logical_and(seqOfEvents[:, 1] == 1, scipy.isnan(seqOfEvents[:, 3])),
                                      scipy.logical_and(seqOfEvents[:, 0] >= timeRange[0] + 1,
                                                        seqOfEvents[:, 0] <= timeRange[1] + 1)))[0]
                indxEnd = scipy.nonzero(
                    scipy.logical_and(scipy.logical_and(seqOfEvents[:, 1] == 2, scipy.isnan(seqOfEvents[:, 3])),
                                      scipy.logical_and(seqOfEvents[:, 0] >= timeRange[0] + 1,
                                                        seqOfEvents[:, 0] <= timeRange[1] + 1)))[0]

                startInfo = scipy.zeros((len(indxStart), 3))

                for i in range(len(indxStart)):
                    iStart = indxStart[i]
                    timeStart = int(seqOfEvents[iStart, 0])
                    rowS = trackStartRow[iTrack] + int(seqOfEvents[iStart, 2]) - 1
                    startInfo[i, :] = scipy.array(
                        [tracksX[timeStart - 1, rowS], tracksY[timeStart - 1, rowS], timeStart - 1])

                endInfo = scipy.zeros((len(indxEnd), 3))

                for i in range(len(indxEnd)):
                    iEnd = indxEnd[i]
                    timeEnd = int(seqOfEvents[iEnd, 0])
                    rowE = trackStartRow[iTrack] + int(seqOfEvents[iEnd, 2]) - 1
                    endInfo[i, :] = scipy.array(
                        [tracksX[timeEnd - 1, rowE], tracksY[timeEnd - 1, rowE], timeEnd - 1])

                if startInfo.shape[0] > 0:
                    lab.plot(startInfo[:, 0], startInfo[:, 1], color='k', marker='o', linestyle='None')

                if endInfo.shape[0] > 0:
                    lab.plot(endInfo[:, 0], endInfo[:, 1], color='k', marker='s', linestyle='None')

        else:

            startInfo = scipy.zeros((numTracks, 3))
            endInfo = scipy.zeros((numTracks, 3))

            for i in range(numTracks):
                timePoint = scipy.nonzero(~scipy.isnan(tracksX[:, i]))[0]
                startInfo[i, :] = scipy.array([tracksX[timePoint[0], i], tracksY[timePoint[0], i], timePoint[0]])
                endInfo[i, :] = scipy.array([tracksX[timePoint[-1], i], tracksY[timePoint[-1], i], timePoint[-1]])

            indx = scipy.nonzero(scipy.logical_and(startInfo[:, 2] >= timeRange[0], startInfo[:, 2] <= timeRange[1]))[0]
            lab.plot(startInfo[indx, 0], startInfo[indx, 1], color='k', marker='o', linestyle='None')
            indx = scipy.nonzero(scipy.logical_and(endInfo[:, 2] >= timeRange[0], endInfo[:, 2] <= timeRange[1]))[0]
            lab.plot(endInfo[indx, 0], endInfo[indx, 1], color='k', marker='s', linestyle='None')

    plotTracks()

    sTime = Slider(lab.axes([0.25, 0.1, 0.65, 0.03], axisbg='lightgoldenrodyellow'), 'Time', 1, numTimePoints,
                   valinit=0,
                   valfmt='%0.0f')

    def update(val):
        time = int(sTime.val) - 1

        if time < 0 or time >= numTimePoints:
            return

        global globalTime
        globalTime = time
        plotTracks()
        lab.draw()

    global pressProcessing
    pressProcessing = False

    def press(event):
        global pressProcessing
        if pressProcessing:
            return

        global globalTime
        time = globalTime
        if event.key == 'right':
            time += 1
        elif event.key == 'left':
            time -= 1

        if time >= 0 and time < numTimePoints and time != globalTime:
            pressProcessing = True
            globalTime = time
            plotTracks()
            lab.draw()
            pressProcessing = False

    sTime.on_changed(update)

    fig.canvas.mpl_connect('key_press_event', press)
    lab.show()


def MovieTracks2D(trackedFeatureInfo, image=None, series=None):
    if image is not None:
        import javabridge as jv, bioformats as bf
        jv.start_vm(class_path=bf.JARS, max_heap_size='8G')
        rdr = bf.ImageReader(image, perform_init=True)
    else:
        rdr = None

    def ReadIm():
        retVal = None
        global globalTime
        s = series if series is not None else 0
        if rdr is not None:
            retVal = rdr.read(z=0, t=globalTime, series=s)
        return retVal[::-1, :, :]

    extendedColours = lab.cm.Set3(scipy.linspace(0, 1, 32))
    isMat = True

    if type(trackedFeatureInfo) is list:
        numTracks = len(trackedFeatureInfo)

        if type(trackedFeatureInfo[0]) is TracksFinal:
            tmp = scipy.concatenate([trackedFeatureInfo[iFrame].seqOfEvents for iFrame in range(numTracks)], axis=0)
            numTimePoints = int(scipy.amax(tmp[:, 0]))
            del tmp
            isMat = False

        else:
            trackedFeatureInfo = constructTrackMatFromCoords(trackedFeatureInfo)

            if trackedFeatureInfo is None:
                return

            if type(trackedFeatureInfo) is tuple:
                visuTracks2D(trackedFeatureInfo[0])
                trackedFeatureInfo = trackedFeatureInfo[1]

            numTracks, numTimePoints = trackedFeatureInfo.shape
            numTimePoints /= 8

    else:
        numTracks, numTimePoints = trackedFeatureInfo.shape
        numTimePoints /= 8

    if not isMat:
        inputStructure = deepcopy(trackedFeatureInfo)
        numSegments = scipy.zeros(numTracks).astype(scipy.int64)

        for i in range(numTracks):
            numSegments[i] = inputStructure[i].tracksCoordAmpCG.shape[0]

        if scipy.amax(numSegments) == 1:
            mergeSplit = False
            trackStartRow = scipy.arange(numTracks).astype(scipy.int64)
            trackedFeatureInfo = scipy.nan * scipy.ones((numTracks, numTimePoints * 8))

            for i in range(numTracks):
                startTime = int(inputStructure[i].seqOfEvents[0, 0])
                endTime = int(inputStructure[i].seqOfEvents[-1, 0])
                trackedFeatureInfo[i, (8 * (startTime - 1)):(8 * endTime)] = inputStructure[i].tracksCoordAmpCG

        else:
            mergeSplit = True
            trackStartRow = scipy.zeros(numTracks).astype(scipy.int64)

            for iTrack in range(1, numTracks):
                trackStartRow[iTrack] = trackStartRow[iTrack - 1] + numSegments[iTrack - 1]

            trackedFeatureInfo = scipy.nan * scipy.ones((trackStartRow[-1] + numSegments[-1], 8 * numTimePoints))

            for i in range(numTracks):
                startTime = int(inputStructure[i].seqOfEvents[0, 0])
                endTime = int(inputStructure[i].seqOfEvents[-1, 0])
                trackedFeatureInfo[trackStartRow[i]:(trackStartRow[i] + numSegments[i]),
                (8 * (startTime - 1)):(8 * endTime)] = inputStructure[i].tracksCoordAmpCG

    else:
        inputStructure = None
        mergeSplit = False
        numSegments = scipy.ones(numTracks).astype(scipy.int64)
        trackStartRow = scipy.arange(numTracks).astype(scipy.int64)

    tracksX = trackedFeatureInfo[:, :-1:8].transpose()
    tracksY = trackedFeatureInfo[:, 1:-1:8].transpose()

    if rdr is not None:
        sizeY = getattr(rdr.rdr, "getSizeY")()
        tracksY = sizeY - tracksY

    maxX = scipy.nanmax(tracksX.flatten())
    minX = scipy.nanmin(tracksX.flatten())
    maxY = scipy.nanmax(tracksY.flatten())
    minY = scipy.nanmin(tracksY.flatten())

    fig = lab.figure()
    ax = lab.subplot(111)
    # lab.subplots_adjust(bottom=0.25)
    ax.set_xlim([minX, maxX])
    ax.set_ylim([minY, maxY])

    global globalTime
    globalTime = 0

    def plotTracks():
        lab.sca(ax)
        xLim = ax.get_xlim()
        yLim = ax.get_ylim()
        lab.cla()
        ax.set_autoscalex_on(False)
        ax.set_autoscaley_on(False)
        ax.set_xlim(xLim)
        ax.set_ylim(yLim)

        def Normalise(im):
            min = scipy.amin(scipy.amin(im, axis=0), axis=0)
            max = scipy.amax(scipy.amax(im, axis=0), axis=0)
            im = (im - min[scipy.newaxis, scipy.newaxis, :]) / ((max - min)[scipy.newaxis, scipy.newaxis, :])
            im = 1 - im
            return im

        im = ReadIm()
        if im is not None:
            im = Normalise(im)
            lab.imshow((2 ** 16) * scipy.ones(shape=im.shape[:2]), cmap=lab.cm.Greys, alpha=1, interpolation=None)
            lab.imshow(im[:, :, 0], cmap=lab.cm.Greens, alpha=.25, interpolation=None)
            lab.imshow(im[:, :, 1], cmap=lab.cm.Reds, alpha=.25, interpolation=None)
            lab.imshow(im[:, :, 2], cmap=lab.cm.Blues, alpha=.25, interpolation=None)
            lab.imshow(im[:, :, 3], cmap=lab.cm.Purples, alpha=.25, interpolation=None)

        global globalTime
        timeRange = scipy.array([scipy.maximum(0, globalTime - 10), globalTime])
        fig.suptitle("Frame = " + str(globalTime + 1) + " / " + str(int(numTimePoints)))

        tracksXP = tracksX[timeRange[0]:(timeRange[1] + 1), :]
        tracksYP = tracksY[timeRange[0]:(timeRange[1] + 1), :]

        for i in range(trackStartRow[-1] + numSegments[-1]):
            obsAvail = scipy.nonzero(~scipy.isnan(tracksXP[:, i]))[0]
            lab.plot(tracksXP[obsAvail, i], tracksYP[obsAvail, i], 'k:', linewidth=2)
            lab.plot(tracksXP[:, i], tracksYP[:, i], color=extendedColours[scipy.mod(i, 32), :], linewidth=2)
            lab.plot(tracksXP[-1, i], tracksYP[-1, i], color=extendedColours[scipy.mod(i, 32), :], marker='+',
                     markersize=3, markeredgewidth=1, linestyle='None')

        if mergeSplit:

            for iTrack in range(numTracks):
                seqOfEvents = inputStructure[iTrack].seqOfEvents
                indxSplit = scipy.nonzero(
                    scipy.logical_and(scipy.logical_and(seqOfEvents[:, 1] == 1, ~scipy.isnan(seqOfEvents[:, 3])),
                                      scipy.logical_and(seqOfEvents[:, 0] > timeRange[0] + 1,
                                                        seqOfEvents[:, 0] <= timeRange[1] + 1)))[0]
                indxMerge = scipy.nonzero(
                    scipy.logical_and(scipy.logical_and(seqOfEvents[:, 1] == 2, ~scipy.isnan(seqOfEvents[:, 3])),
                                      scipy.logical_and(seqOfEvents[:, 0] > timeRange[0] + 1,
                                                        seqOfEvents[:, 0] <= timeRange[1] + 1)))[0]

                for iSplit in indxSplit:
                    timeSplit = int(seqOfEvents[iSplit, 0])
                    rowS = trackStartRow[iTrack] + int(seqOfEvents[iSplit, 2]) - 1
                    rowSp = trackStartRow[iTrack] + int(seqOfEvents[iSplit, 3]) - 1
                    lab.plot([tracksX[timeSplit - 1, rowS], tracksX[timeSplit - 2, rowSp]],
                             [tracksY[timeSplit - 1, rowS], tracksY[timeSplit - 2, rowSp]], 'k-.', linewidth=2)

                for iMerge in indxMerge:
                    timeSplit = int(seqOfEvents[iMerge, 0])
                    rowE = trackStartRow[iTrack] + int(seqOfEvents[iSplit, 2]) - 1
                    rowM = trackStartRow[iTrack] + int(seqOfEvents[iSplit, 3]) - 1
                    lab.plot([tracksX[timeSplit - 2, rowE], tracksX[timeSplit - 1, rowM]],
                             [tracksY[timeSplit - 2, rowE], tracksY[timeSplit - 1, rowM]], 'k--', linewidth=2)

                indxStart = scipy.nonzero(
                    scipy.logical_and(scipy.logical_and(seqOfEvents[:, 1] == 1, scipy.isnan(seqOfEvents[:, 3])),
                                      scipy.logical_and(seqOfEvents[:, 0] >= timeRange[0] + 1,
                                                        seqOfEvents[:, 0] <= timeRange[1] + 1)))[0]
                indxEnd = scipy.nonzero(
                    scipy.logical_and(scipy.logical_and(seqOfEvents[:, 1] == 2, scipy.isnan(seqOfEvents[:, 3])),
                                      scipy.logical_and(seqOfEvents[:, 0] >= timeRange[0] + 1,
                                                        seqOfEvents[:, 0] <= timeRange[1] + 1)))[0]

                startInfo = scipy.zeros((len(indxStart), 3))

                for i in range(len(indxStart)):
                    iStart = indxStart[i]
                    timeStart = int(seqOfEvents[iStart, 0])
                    rowS = trackStartRow[iTrack] + int(seqOfEvents[iStart, 2]) - 1
                    startInfo[i, :] = scipy.array(
                        [tracksX[timeStart - 1, rowS], tracksY[timeStart - 1, rowS], timeStart - 1])

                endInfo = scipy.zeros((len(indxEnd), 3))

                for i in range(len(indxEnd)):
                    iEnd = indxEnd[i]
                    timeEnd = int(seqOfEvents[iEnd, 0])
                    rowE = trackStartRow[iTrack] + int(seqOfEvents[iEnd, 2]) - 1
                    endInfo[i, :] = scipy.array(
                        [tracksX[timeEnd - 1, rowE], tracksY[timeEnd - 1, rowE], timeEnd - 1])

                if startInfo.shape[0] > 0:
                    lab.plot(startInfo[:, 0], startInfo[:, 1], color='k', marker='o', linestyle='None', markersize=3,
                             markeredgewidth=1)

                if endInfo.shape[0] > 0:
                    lab.plot(endInfo[:, 0], endInfo[:, 1], color='k', marker='s', linestyle='None', markersize=3,
                             markeredgewidth=1)

        else:

            startInfo = scipy.zeros((numTracks, 3))
            endInfo = scipy.zeros((numTracks, 3))

            for i in range(numTracks):
                timePoint = scipy.nonzero(~scipy.isnan(tracksX[:, i]))[0]
                startInfo[i, :] = scipy.array([tracksX[timePoint[0], i], tracksY[timePoint[0], i], timePoint[0]])
                endInfo[i, :] = scipy.array([tracksX[timePoint[-1], i], tracksY[timePoint[-1], i], timePoint[-1]])

            indx = scipy.nonzero(scipy.logical_and(startInfo[:, 2] >= timeRange[0], startInfo[:, 2] <= timeRange[1]))[0]
            lab.plot(startInfo[indx, 0], startInfo[indx, 1], color='k', marker='o', linestyle='None')
            indx = scipy.nonzero(scipy.logical_and(endInfo[:, 2] >= timeRange[0], endInfo[:, 2] <= timeRange[1]))[0]
            lab.plot(endInfo[indx, 0], endInfo[indx, 1], color='k', marker='s', linestyle='None')

    # plotTracks()

    # sTime = Slider(lab.axes([0.25, 0.1, 0.65, 0.03], axisbg='lightgoldenrodyellow'), 'Time', 1, numTimePoints,
    #                valinit=0,
    #                valfmt='%0.0f')

    # def update(val):
    #     time = int(sTime.val) - 1
    #
    #     if time < 0 or time >= numTimePoints:
    #         return
    #
    #     global globalTime
    #     globalTime = time
    #     plotTracks()
    #     lab.draw()

    # global pressProcessing
    # pressProcessing = False
    # def press(event):
    #     global pressProcessing
    #     if pressProcessing:
    #         return
    #
    #     global globalTime
    #     time = globalTime
    #     if event.key == 'right':
    #         time += 1
    #     elif event.key == 'left':
    #         time -= 1
    #
    #     if time >= 0 and time < numTimePoints and time != globalTime:
    #         pressProcessing = True
    #         globalTime = time
    #         plotTracks()
    #         lab.draw()
    #         pressProcessing = False

    # sTime.on_changed(update)

    # fig.canvas.mpl_connect('key_press_event', press)
    # lab.show()

    import matplotlib.animation as manimation

    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='Tracking results of ' + image, artist='Matplotlib',
                    comment='')
    writer = FFMpegWriter(fps=10, metadata=metadata)

    with writer.saving(fig, "writer_test.mp4", 500):
        for i in range(numTimePoints):
            print("writing frame " + str(i) + "...")
            globalTime = i
            plotTracks()
            writer.grab_frame()


def visuTracks3D(trackedFeatureInfo):
    from mpl_toolkits.mplot3d import Axes3D

    Axes3D.zorder
    extendedColours = lab.cm.Set3(scipy.linspace(0, 1, 32))
    isMat = True

    if type(trackedFeatureInfo) is list:
        numTracks = len(trackedFeatureInfo)

        if type(trackedFeatureInfo[0]) is TracksFinal:
            tmp = scipy.concatenate([trackedFeatureInfo[iFrame].seqOfEvents for iFrame in range(numTracks)], axis=0)
            numTimePoints = int(scipy.amax(tmp[:, 0]))
            del tmp
            isMat = False

        else:
            trackedFeatureInfo = constructTrackMatFromCoords(trackedFeatureInfo)

            if trackedFeatureInfo is None:
                return

            if type(trackedFeatureInfo) is tuple:
                visuTracks3D(trackedFeatureInfo[0])
                trackedFeatureInfo = trackedFeatureInfo[1]

            numTracks, numTimePoints = trackedFeatureInfo.shape
            numTimePoints /= 8

    else:
        numTracks, numTimePoints = trackedFeatureInfo.shape
        numTimePoints /= 8

    if not isMat:
        inputStructure = deepcopy(trackedFeatureInfo)
        numSegments = scipy.zeros(numTracks).astype(scipy.int64)

        for i in range(numTracks):
            numSegments[i] = inputStructure[i].tracksCoordAmpCG.shape[0]

        if scipy.amax(numSegments) == 1:
            mergeSplit = False
            trackStartRow = scipy.arange(numTracks).astype(scipy.int64)
            trackedFeatureInfo = scipy.nan * scipy.ones((numTracks, numTimePoints * 8))

            for i in range(numTracks):
                startTime = int(inputStructure[i].seqOfEvents[0, 0])
                endTime = int(inputStructure[i].seqOfEvents[-1, 0])
                trackedFeatureInfo[i, (8 * (startTime - 1)):(8 * endTime)] = inputStructure[i].tracksCoordAmpCG

        else:
            mergeSplit = True
            trackStartRow = scipy.zeros(numTracks).astype(scipy.int64)

            for iTrack in range(1, numTracks):
                trackStartRow[iTrack] = trackStartRow[iTrack - 1] + numSegments[iTrack - 1]

            trackedFeatureInfo = scipy.nan * scipy.ones((trackStartRow[-1] + numSegments[-1], 8 * numTimePoints))

            for i in range(numTracks):
                startTime = int(inputStructure[i].seqOfEvents[0, 0])
                endTime = int(inputStructure[i].seqOfEvents[-1, 0])
                trackedFeatureInfo[trackStartRow[i]:(trackStartRow[i] + numSegments[i]),
                (8 * startTime):(8 * endTime)] = inputStructure[i].tracksCoordAmpCG

    else:
        inputStructure = None
        mergeSplit = False
        numSegments = scipy.ones(numTracks).astype(scipy.int64)
        trackStartRow = scipy.arange(numTracks).astype(scipy.int64)

    tracksX = trackedFeatureInfo[:, :-1:8].transpose()
    tracksY = trackedFeatureInfo[:, 1:-1:8].transpose()
    tracksZ = trackedFeatureInfo[:, 2:-1:8].transpose()

    maxX = scipy.nanmax(tracksX.flatten())
    minX = scipy.nanmin(tracksX.flatten())
    maxY = scipy.nanmax(tracksY.flatten())
    minY = scipy.nanmin(tracksY.flatten())
    maxZ = scipy.nanmax(tracksZ.flatten())
    minZ = scipy.nanmin(tracksZ.flatten())

    fig = lab.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    lab.subplots_adjust(bottom=0.25)
    ax.set_xlim([minX, maxX])
    ax.set_ylim([minY, maxY])
    ax.set_zlim([minZ, maxZ])

    global globalTime
    globalTime = 0

    def plotTracks():
        lab.sca(ax)
        xLim = ax.get_xlim()
        yLim = ax.get_ylim()
        zLim = ax.get_zlim()
        azim = ax.azim
        elev = ax.elev
        lab.cla()
        ax.set_autoscalex_on(False)
        ax.set_autoscaley_on(False)
        ax.set_xlim(xLim)
        ax.set_ylim(yLim)
        ax.set_zlim(zLim)
        ax.azim = azim
        ax.elev = elev
        global globalTime
        timeRange = scipy.array([scipy.maximum(0, globalTime - 10), globalTime])
        fig.suptitle("Frame = " + str(globalTime + 1) + " / " + str(int(numTimePoints)))

        tracksXP = tracksX[timeRange[0]:(timeRange[1] + 1), :]
        tracksYP = tracksY[timeRange[0]:(timeRange[1] + 1), :]
        tracksZP = tracksZ[timeRange[0]:(timeRange[1] + 1), :]

        for i in range(trackStartRow[-1] + numSegments[-1]):
            obsAvail = scipy.nonzero(~scipy.isnan(tracksXP[:, i]))[0]
            ax.plot(tracksXP[obsAvail, i], tracksYP[obsAvail, i], 'k:', zs=tracksZP[obsAvail, i], zdir='z', linewidth=3)
            ax.plot(tracksXP[:, i], tracksYP[:, i], zs=tracksZP[:, i], zdir='z',
                    color=extendedColours[scipy.mod(i, 32), :], linewidth=3)
            ax.plot(tracksXP[-1, i], tracksYP[-1, i], zs=[tracksZP[-1, i]], zdir='z',
                    color=extendedColours[scipy.mod(i, 32), :], marker='+',
                    markersize=10, markeredgewidth=5, linestyle='None')

        if mergeSplit:

            for iTrack in range(numTracks):
                seqOfEvents = inputStructure[iTrack].seqOfEvents
                indxSplit = scipy.nonzero(
                    scipy.logical_and(scipy.logical_and(seqOfEvents[:, 1] == 1, ~scipy.isnan(seqOfEvents[:, 3])),
                                      scipy.logical_and(seqOfEvents[:, 0] > timeRange[0] + 1,
                                                        seqOfEvents[:, 0] <= timeRange[1] + 1)))[0]
                indxMerge = scipy.nonzero(
                    scipy.logical_and(scipy.logical_and(seqOfEvents[:, 1] == 2, ~scipy.isnan(seqOfEvents[:, 3])),
                                      scipy.logical_and(seqOfEvents[:, 0] > timeRange[0] + 1,
                                                        seqOfEvents[:, 0] <= timeRange[1] + 1)))[0]

                for iSplit in indxSplit:
                    timeSplit = int(seqOfEvents[iSplit, 0])
                    rowS = trackStartRow[iTrack] + int(seqOfEvents[iSplit, 2])
                    rowSp = trackStartRow[iTrack] + int(seqOfEvents[iSplit, 3])
                    ax.plot([tracksX[timeSplit - 1, rowS], tracksX[timeSplit - 2, rowSp]],
                            [tracksY[timeSplit - 1, rowS], tracksY[timeSplit - 2, rowSp]], 'k-.',
                            zs=[tracksZ[timeSplit - 1, rowS], tracksZ[timeSplit - 2, rowSp]], zdir='z', linewidth=3)

                for iMerge in indxMerge:
                    timeSplit = int(seqOfEvents[iMerge, 0])
                    rowE = trackStartRow[iTrack] + int(seqOfEvents[iSplit, 2])
                    rowM = trackStartRow[iTrack] + int(seqOfEvents[iSplit, 3])
                    ax.plot([tracksX[timeSplit - 2, rowE], tracksX[timeSplit - 1, rowM]],
                            [tracksY[timeSplit - 2, rowE], tracksY[timeSplit - 1, rowM]], 'k--',
                            zs=[tracksZ[timeSplit - 2, rowE], tracksZ[timeSplit - 1, rowM]], zdir='z', linewidth=3)

                indxStart = scipy.nonzero(
                    scipy.logical_and(scipy.logical_and(seqOfEvents[:, 1] == 1, scipy.isnan(seqOfEvents[:, 3])),
                                      scipy.logical_and(seqOfEvents[:, 0] >= timeRange[0] + 1,
                                                        seqOfEvents[:, 0] <= timeRange[1] + 1)))[0]
                indxEnd = scipy.nonzero(
                    scipy.logical_and(scipy.logical_and(seqOfEvents[:, 1] == 2, scipy.isnan(seqOfEvents[:, 3])),
                                      scipy.logical_and(seqOfEvents[:, 0] >= timeRange[0] + 1,
                                                        seqOfEvents[:, 0] <= timeRange[1] + 1)))[0]

                startInfo = scipy.zeros((len(indxStart), 4))

                for i in range(len(indxStart)):
                    iStart = indxStart[i]
                    timeStart = int(seqOfEvents[iStart, 0])
                    rowS = trackStartRow[iTrack] + int(seqOfEvents[iStart, 2])
                    startInfo[i, :] = scipy.array(
                        [tracksX[timeStart - 1, rowS], tracksY[timeStart - 1, rowS], timeStart - 1,
                         tracksZ[timeStart - 1, rowS]])

                endInfo = scipy.zeros((len(indxEnd), 4))

                for i in range(len(indxEnd)):
                    iEnd = indxEnd[i]
                    timeEnd = int(seqOfEvents[iEnd, 0])
                    rowE = trackStartRow[iTrack] + int(seqOfEvents[iEnd, 2])
                    endInfo[i, :] = scipy.array(
                        [tracksX[timeEnd - 1, rowE], tracksY[timeEnd - 1, rowE], timeEnd - 1,
                         tracksZ[timeEnd - 1, rowE]])

                if startInfo.shape[0] > 0:
                    ax.plot(startInfo[:, 0], startInfo[:, 1], zs=startInfo[:, 3], zdir='z', color='k', marker='o',
                            linestyle='None')

                if endInfo.shape[0] > 0:
                    ax.plot(endInfo[:, 0], endInfo[:, 1], zs=endInfo[:, 3], zdir='z', color='k', marker='s',
                            linestyle='None')

        else:

            startInfo = scipy.zeros((numTracks, 4))
            endInfo = scipy.zeros((numTracks, 4))

            for i in range(numTracks):
                timePoint = scipy.nonzero(~scipy.isnan(tracksX[:, i]))[0]
                startInfo[i, :] = scipy.array(
                    [tracksX[timePoint[0], i], tracksY[timePoint[0], i], timePoint[0], tracksZ[timePoint[0], i]])
                endInfo[i, :] = scipy.array(
                    [tracksX[timePoint[-1], i], tracksY[timePoint[-1], i], timePoint[-1], tracksZ[timePoint[-1], i]])

            indx = scipy.nonzero(scipy.logical_and(startInfo[:, 2] >= timeRange[0], startInfo[:, 2] <= timeRange[1]))[0]
            ax.plot(startInfo[indx, 0], startInfo[indx, 1], zs=startInfo[indx, 3], zdir='z', color='k', marker='o',
                    linestyle='None')
            indx = scipy.nonzero(scipy.logical_and(endInfo[:, 2] >= timeRange[0], endInfo[:, 2] <= timeRange[1]))[0]
            ax.plot(endInfo[indx, 0], endInfo[indx, 1], zs=endInfo[indx, 3], zdir='z', color='k', marker='s',
                    linestyle='None')

    plotTracks()

    sTime = Slider(lab.axes([0.25, 0.1, 0.65, 0.03], axisbg='lightgoldenrodyellow'), 'Time', 1, numTimePoints,
                   valinit=0,
                   valfmt='%0.0f')

    def update(val):
        time = int(sTime.val) - 1

        if time < 0 or time >= numTimePoints:
            return

        global globalTime
        globalTime = time
        plotTracks()
        lab.draw()

    sTime.on_changed(update)

    lab.show()
