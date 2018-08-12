import sys, os, numpy as np
#sys.path.append(os.path.dirname(__file__) + "../MovieGenerator")
#sys.path.append(os.path.dirname(__file__) + "../keras-frcnn")
#from MovieGenerator import GenerateMovie
from PIL import Image, ImageDraw
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix

GenImSize = 128

class positions_and_radii:
	def __init__(self, positions, radii):
		self.positions = positions
		self.radii = radii

def MakeCellImage(x, y, r, i):
    im = Image.new(mode='F', size=(GenImSize, GenImSize))
    draw = ImageDraw.Draw(im)
    draw.ellipse(xy=[x-r, y-r, x+r, y+r], fill='White')
    im = np.array(im).astype(np.float32)
    im *= (i / 255.0)
    #im += (np.random.randn(im.shape[0], im.shape[1]) * 0.1) + 0.2
    #im[im < 0] = 0
    #im[im > 1] = 1
    return im

def GenerateMovie(numFrames=256, numCells=3, InitVelSD=1, AngleSDRad=np.radians(5), VelScaleSD=0.1,
				  WriteMovieFile=False):
	global position
	global position_minusOne
	global radius
	global velocity

	movie = np.zeros(shape=(GenImSize, GenImSize, numFrames))

	radius = np.random.randint(size=numCells, low=-5, high=10) + 10
	intensity = (np.random.randn(numCells) * 0.1) + 0.5
	intensity = np.clip(intensity, a_min=0.0, a_max=1.0)

	position = np.zeros(shape=(numCells, 2))
	position_minusOne = np.zeros(shape=(numCells, 2))

	class positions_and_radii:
		def __init__(self, positions, radii):
			self.positions = positions
			self.radii = radii

	rois = []

	def overlap(p1, r1, p2, r2):
		diff = p1 - p2
		return ((diff[0] * diff[0]) + (diff[1] * diff[1])) < ((r1 + r2) ** 2)

	for c in range(numCells):
		goodPos = False
		i = 0
		while not goodPos and i < (numCells * numCells):
			i += 1
			goodPos = True
			position[c, :] = np.random.randint(low=radius[c], high=GenImSize - radius[c], size=2)
			for c2 in range(c):
				if overlap(position[c, :], radius[c], position[c2, :], radius[c2]):
					radius[c] = np.clip(radius[c] - 1, a_min=5, a_max=20)
					goodPos = False
					break

	def IntForces():
		return np.zeros(shape=(numCells, 2))

	velocity = (np.random.randn(numCells, 2) * InitVelSD) + IntForces()
	forces = np.zeros(shape=(numCells, 2))

	def MakeImage(t):
		for c in range(numCells):
			movie[:, :, t] += MakeCellImage(position[c, 0], position[c, 1], radius[c], intensity[c])
		movie[:, :, t] += ((np.random.randn(movie.shape[0], movie.shape[1]) * 0.1) + 0.2)
		movie[:, :, t] = np.clip(movie[:, :, t], a_min=0.0, a_max=1.0)
		rois.insert(t, positions_and_radii(position.copy(), radius.copy()))

	def UpdatePositions(candidate):
		global position
		global position_minusOne
		# global velocity

		position_minusOne = position.copy()
		done = False
		i = 0
		while not done and i < (numCells * numCells):
			i += 1
			done = True
			for c_i in range(numCells):
				if np.any(candidate[c_i, :] < radius[c]) or np.any(candidate[c_i, :] > (GenImSize - radius[c])):
					candidate[c_i, :] = position[c_i, :]
					# velocity[c_i, :] *= -1
					done = False
				for c_j in range(numCells):
					if c_i != c_j and overlap(candidate[c_i, :], radius[c_i], candidate[c_j, :], radius[c_j]):
						candidate[c_i, :] = position[c_i, :]
						# velocity[c_i, :] *= -1
						done = False
						break
		position = candidate

	def RandRotVel():
		global velocity

		theta = AngleSDRad * np.random.randn()
		c, s = np.cos(theta), np.sin(theta)
		R = np.matrix([[c, -s], [s, c]])
		vMag = np.linalg.norm(velocity, axis=1, keepdims=True)
		velocity /= vMag
		velocity = np.array((R * velocity.transpose()).transpose())
		vMag += (VelScaleSD * np.random.randn(numCells, 1))
		velocity *= vMag

	def UpdateVelocityAndForces():
		global velocity
		global forces

		oldVel = velocity.copy()
		RandRotVel()
		velocity += IntForces()
		forces = velocity - oldVel

	# Frame 0
	MakeImage(0)

	# Frame 1
	UpdatePositions(position + velocity)
	UpdateVelocityAndForces()
	MakeImage(1)

	# Frames 2-n
	for t in range(2, numFrames):
		UpdatePositions((2 * (forces + position)) - position_minusOne)
		UpdateVelocityAndForces()
		MakeImage(t)

	return movie, rois


def IterativeRobustMean(data_in, dim=None, k=None, fit=None):
    data = data_in.copy()
    oldOutliers = scipy.array([])
    loop = True
    while loop:
        finalMean, stdSample, inlierIdx, outlierIdx = robustMean(data, dim, k, fit)
        data[outlierIdx] = scipy.nan
        loop = scipy.array_equal(oldOutliers, outlierIdx) is not True
        oldOutliers = outlierIdx

    return data, finalMean, stdSample

def robustMean(data, dim=None, k=None, fit=None):
    if dim is None:

        if len(data.shape) > 1:
            dim = scipy.nonzero(scipy.array(data.shape) > 1)[0][0]
        else:
            dim = 0

    if k is None:
        k = 3

    if fit is None:
        fit = True

    if fit:

        if scipy.sum(scipy.array(data.shape) > 1) > 1:
            print("fitting is currently only supported for 1D data")
            return [], [], [], []

    if scipy.sum(scipy.isfinite(data.flatten())) < 4:
        finalMean = scipy.nanmean(data, axis=dim)
        stdSample = scipy.nan * scipy.ones(finalMean.shape)
        inlierIdx = scipy.nonzero(scipy.isfinite(data.flatten()))[0]
        outlierIdx = scipy.array([]).astype(scipy.int64)
        return finalMean, stdSample, inlierIdx, outlierIdx

    magicNumber = 1.4826 ** 2
    dataSize = scipy.array(data.shape)
    reducedDataSize = dataSize.copy()
    reducedDataSize[dim] = 1
    blowUpDataSize = (dataSize / reducedDataSize).astype(scipy.int32)
    realDimensions = len(scipy.nonzero(dataSize > 1)[0])

    if fit:
        medianData = fmin(lambda x: scipy.nanmedian((data - x) ** 2), scipy.nanmedian(data), disp=False)

    else:
        medianData = scipy.nanmedian(data, axis=dim)

    res2 = (data - scipy.tile(medianData, blowUpDataSize)) ** 2
    medRes2 = scipy.maximum(scipy.nanmedian(res2, axis=dim), scipy.spacing(1))
    testValue = res2 / scipy.tile(magicNumber * medRes2, blowUpDataSize)

    if realDimensions == 1:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            inlierIdx = scipy.nonzero(testValue <= (k ** 2))[0]
            outlierIdx = scipy.nonzero(testValue > (k ** 2))[0]
        nInlier = len(inlierIdx)

        if nInlier > 4:
            stdSample = scipy.sqrt(scipy.sum(res2[inlierIdx]) / (nInlier - 4))

        else:
            stdSample = scipy.nan

        finalMean = scipy.mean(data[inlierIdx])

    else:
        inlierIdx = scipy.nonzero(testValue <= (k ** 2))
        outlierIdx = scipy.nonzero(testValue > (k ** 2))
        res2[outlierIdx] = scipy.nan
        nInlier = scipy.sum(~scipy.isnan(res2), axis=dim)
        goodIdx = scipy.sum(scipy.isfinite(res2), axis=dim) > 4
        stdSample = scipy.nan * scipy.ones(goodIdx.shape)
        stdSample[goodIdx] = scipy.sqrt(scipy.nansum(res2[goodIdx], axis=dim) / (nInlier[goodIdx] - 4))
        data[outlierIdx] = scipy.nan
        finalMean = scipy.nanmean(data, axis=dim)

    return finalMean, stdSample, inlierIdx, outlierIdx

from keras.models import load_model

np.random.seed(400582830)

sys.path.append(os.path.dirname(__file__) + "/../keras-frcnn")
for p in sys.path:
	print(p)

from frcnn_detect import Detect

movie, _ = GenerateMovie(numFrames=128, numCells=6, InitVelSD=1, VelScaleSD=1, AngleSDRad=np.radians(20),
						 WriteMovieFile=False)
rois = Detect(movie)

minProbForCorrectAssignment = 0.01
stdDevCutoffForTrackAssignment = 3


def GetCentreDisp(rois):
	# disp = np.nan * np.ones(len(rois)-1)
	# for roi_curr, roi_next, v in zip(rois, rois[1:], np.nditer(disp, op_flags=['writeonly'])):
	# 	vec = (roi_next.positions - roi_curr.positions)[0, :]
	# 	v[...] = np.linalg.norm(vec)
	# return disp

	disp = []
	first = None
	for i, roi in enumerate(rois):
		if roi.radii[0] >= 0.0:
			first = i
			break

	for i in range(first + 1, len(rois)):
		roi = rois[i]
		if roi.radii[0] >= 0.0:
			vec = (roi.positions - rois[first].positions)[0, :]
			disp.append(np.linalg.norm(vec) / (i - first))
			first = i

	return np.array(disp)


nFrames = 11
row, col = 64, 64
roiExpansionFactor = 1.5
movieRow, movieCol = movie.shape[0:2]

movie = np.pad(movie, ((0, 0), (0, 0), (nFrames - 1, 0)), 'reflect', reflect_type='even')
rois = np.pad(np.array(rois), ((nFrames - 1, 0)), 'reflect', reflect_type='even').tolist()


class track_data(object):
	def __init__(self, initROIs):
		self.rois = [positions_and_radii(initROIs.positions.copy(), initROIs.radii.copy())]
		n = initROIs.positions.shape[0]
		self.timepoints = [0 for i in range(n)]
		self.trackIndexes = [i for i in range(n)]
		self.roiIndexes_plusone = [i for i in range(1, n + 1)]

	def GetAvail(self, timepoint):
		m = coo_matrix((self.roiIndexes_plusone, (self.trackIndexes, self.timepoints)))
		m = csc_matrix(m)
		idx = m[:, max(0, timepoint - nFrames + 2):(timepoint + 1)].toarray()
		idx = np.where(np.any(idx > 0, axis=1))[0]
		return idx

	def GetROIs(self, trackIDs=None, timepoints=None):
		if trackIDs is None:
			trackIDs = np.arange(max(self.trackIndexes) + 1)

		if timepoints is None:
			timepoints = (0, max(self.timepoints))

		nTracks = len(trackIDs)
		nTimepoints = timepoints[1] - timepoints[0] + 1
		result = [
			positions_and_radii(-1 * np.ones((nTracks, 2), dtype=np.float), -1 * np.ones((nTracks), dtype=np.float)) for
			i in range(nTimepoints)]

		m = coo_matrix((self.roiIndexes_plusone, (self.trackIndexes, self.timepoints)))
		m = csr_matrix(m)
		m = m[trackIDs, :]
		m = csc_matrix(m)
		m = m[:, timepoints[0]:(timepoints[1] + 1)]
		m = m.toarray()

		for iTrack in range(nTracks):
			for iTimePoint in range(nTimepoints):
				if m[iTrack, iTimePoint] > 0:
					result[iTimePoint].positions[iTrack, :] = self.rois[iTimePoint + timepoints[0]].positions[
															  m[iTrack, iTimePoint] - 1, :]
					result[iTimePoint].radii[iTrack] = self.rois[iTimePoint + timepoints[0]].radii[
						m[iTrack, iTimePoint] - 1]

		return result

	def AddDetections(self, trackIDs, newRois, timepoint):
		if len(newRois) > 0:
			if len(self.rois) <= timepoint:
				offset = 0
				self.rois.insert(timepoint, newRois[0])
			else:
				offset = self.rois[timepoint].positions.shape[0]
				self.rois[timepoint].positions = np.concatenate((self.rois[timepoint].positions, newRois[0].positions),
																axis=0)
				self.rois[timepoint].radii = np.concatenate((self.rois[timepoint].radii, newRois[0].radii), axis=0)

			for track in trackIDs:
				offset += 1
				self.timepoints.append(timepoint)
				self.trackIndexes.append(track)
				self.roiIndexes_plusone.append(offset)

	def AddNewTracks(self, newRois, timepoint):
		if len(newRois) > 0:
			t = max(self.trackIndexes)
			newTracks = [i + t for i in range(1, newRois[0].positions.shape[0] + 1)]
			self.AddDetections(newTracks, newRois, timepoint)


tracks = track_data(rois[0])


def NumCells(timepoint):
	return rois[timepoint].positions.shape[0]


def GetCellROIs(cellIDx, timepoint):
	if type(cellIDx) is np.ndarray:
		if len(cellIDx) == 0:
			return []
		else:
			return [positions_and_radii(rois[timepoint].positions[cellIDx, :], rois[timepoint].radii[cellIDx])]
	else:
		return [positions_and_radii(rois[timepoint].positions[cellIDx:(cellIDx + 1), :],
									rois[timepoint].radii[cellIDx:(cellIDx + 1)])]


class rectangle(object):
	def __init__(self, topLeft=np.zeros((2, 1), dtype=np.float), bottomRight=np.zeros((2, 1), dtype=np.float)):
		self.topLeft = topLeft
		self.bottomRight = bottomRight


def RecFromROI(centre, radius):
	r = rectangle(centre - radius, centre + radius)
	return r


def AddRec(r1, r2):
	r = rectangle(np.minimum(r1.topLeft, r2.topLeft), np.maximum(r1.bottomRight, r2.bottomRight))
	return r


def RangeFromRec(r):
	xmin, ymin = np.maximum(np.zeros((2)),
							np.minimum(np.array([movieRow - 1, movieCol - 1]), np.floor(r.topLeft))).astype(np.uint32)
	xmax, ymax = np.maximum(np.zeros((2)),
							np.minimum(np.array([movieRow - 1, movieCol - 1]), np.ceil(r.bottomRight))).astype(
		np.uint32)
	return (xmin, xmax), (ymin, ymax)


class roi_extractor(object):
	def __init__(self, tStart, ROIs, cellIndexes):
		self.ROIs = ROIs
		self.cellIndexes = cellIndexes
		self.tStart = tStart
		self.currentIndex = 0

		r = rectangle(topLeft=np.array([movieRow, movieCol]), bottomRight=np.zeros((2)))
		for i, index in enumerate(cellIndexes):
			if ROIs[i].radii[index] >= 0.0:
				r = AddRec(r, RecFromROI(ROIs[i].positions[index, :], roiExpansionFactor * ROIs[i].radii[index]))
		self.rec = r

	def extractNextROI(self):
		(total_xmin, total_xmax), (total_ymin, total_ymax) = RangeFromRec(self.rec)
		tmpFrame = np.zeros((total_xmax - total_xmin, total_ymax - total_ymin))

		if self.ROIs[self.currentIndex].radii[self.cellIndexes[self.currentIndex]] >= 0:
			frameRec = RecFromROI(self.ROIs[self.currentIndex].positions[self.cellIndexes[self.currentIndex], :],
								  roiExpansionFactor * self.ROIs[self.currentIndex].radii[
									  self.cellIndexes[self.currentIndex]])

			(xmin, xmax), (ymin, ymax) = RangeFromRec(frameRec)

			tmpFrame[xmin - total_xmin:xmax - total_xmin, ymin - total_ymin:ymax - total_ymin] = movie[xmin:xmax,
																								 ymin:ymax,
																								 self.tStart + self.currentIndex]

		tmpFrame = Image.fromarray(tmpFrame)
		tmpFrame = tmpFrame.resize((row, col), Image.ANTIALIAS)
		tmpFrame = np.array(tmpFrame)
		m, s = np.mean(tmpFrame.flatten()), np.std(tmpFrame.flatten())
		if s == 0.0:
			s = 1.0
		tmpFrame = (tmpFrame - m) / s

		self.currentIndex += 1

		return tmpFrame


nFrames_movie = len(rois)
model = load_model('TrackingModel.h5')

probs_all = []

for currFrame in range(1, nFrames_movie):
	prevFrame = currFrame - 1
	availTracks = tracks.GetAvail(prevFrame)
	nTracksAvail = availTracks.shape[0]
	nCells = NumCells(currFrame)
	costMat = np.inf * np.ones((nTracksAvail, nCells + 1), dtype=np.float)

	probs = -1 * np.ones((nTracksAvail, nCells, 3), dtype=np.float)

	for iTrack in range(nTracksAvail):
		for iCell in range(nCells):

			startFrame_subTrack = max(0, currFrame - nFrames + 1)

			trackRois = tracks.GetROIs([availTracks[iTrack]], (0, prevFrame))
			cellRoi = GetCellROIs((iCell), currFrame)

			disp = GetCentreDisp(trackRois + cellRoi)
			disp, _, _ = IterativeRobustMean(disp, k=stdDevCutoffForTrackAssignment)

			if np.isnan(disp[-1]) is not True:

				trackRois = tracks.GetROIs([availTracks[iTrack]], (startFrame_subTrack, prevFrame))
				inputMovie = np.zeros((1, nFrames, row, col, 1), dtype=np.float)
				roiExtractor = roi_extractor(startFrame_subTrack, trackRois + cellRoi,
											 [0 for j in range(currFrame - startFrame_subTrack + 1)])

				offset = nFrames - (currFrame - startFrame_subTrack + 1)
				for kFrame in range(startFrame_subTrack, currFrame + 1):
					inputMovie[0, offset, :, :, 0] = roiExtractor.extractNextROI()
					offset += 1

				p = model.predict(inputMovie)[0, :]
				p /= np.linalg.norm(p)
				if p[0] > minProbForCorrectAssignment:
					costMat[iTrack, iCell] = p[1]

				probs[iTrack, iCell, 0] = availTracks[iTrack]
				probs[iTrack, iCell, 1] = p[0]
				probs[iTrack, iCell, 2] = p[1]

	maxCost = max(costMat[:, :nCells].flatten())
	if np.isinf(maxCost):
		maxCost = 1.0
	costMat[:, -1] = 1.1 * maxCost
	costMat[np.isinf(costMat)] = 1.2 * maxCost

	track_ind, cell_ind = linear_sum_assignment(costMat)
	track_ind = track_ind[cell_ind < nCells]
	cell_ind = cell_ind[cell_ind < nCells]

	tracks.AddDetections(availTracks[track_ind], GetCellROIs(cell_ind, currFrame), currFrame)
	tracks.AddNewTracks(GetCellROIs(np.setxor1d(cell_ind, np.arange(nCells), assume_unique=True), currFrame), currFrame)

	probs_all.append(probs)

	sys.stdout.write("Tracking: {0:.2f}%".format(100.0 * currFrame / (nFrames_movie - 1)) + '\r')
	sys.stdout.flush()

WriteMovie(data=movie, roiData=tracks.GetROIs(), name='trackingTest_wDetector.mp4')

print("Tracking: {0:.2f}%".format(100.0 * currFrame / (nFrames_movie - 1)))

