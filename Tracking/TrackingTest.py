import sys, os, numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../keras-frcnn")
from MovieGenerator.MovieGenerator import GenerateMovie, WriteMovie, positions_and_radii
from PIL import Image
from keras.models import load_model
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix
from scipy.spatial import distance

# def Predict_TrackerOnly():
# 	movie, rois = GenerateMovie(numCells=5, InitVelSD=1, VelScaleSD=1, AngleSDRad=np.radians(20), WriteMovieFile=False)
	
# 	for roi in rois:
# 		n = roi.positions.shape[0]
# 		p = np.random.permutation(np.arange(n))
# 		roi.positions = roi.positions[p, :]
# 		roi.radii = roi.radii[p]

# 	rois_rand = [positions_and_radii(roi.positions.copy(), roi.radii.copy()) for roi in rois]

# 	nFrames = 11
# 	row, col = 64, 64
# 	roiExpansionFactor = 1.5
# 	movieRow, movieCol = movie.shape[0:2]

# 	class rectangle(object):
# 		def __init__(self, topLeft=np.zeros((2, 1), dtype=np.float), bottomRight=np.zeros((2, 1), dtype=np.float)):
# 			self.topLeft = topLeft
# 			self.bottomRight = bottomRight

# 	def RecFromROI(centre, radius):
# 		r = rectangle(centre - radius, centre + radius)
# 		return r

# 	def AddRec(r1, r2):
# 		r = rectangle(np.minimum(r1.topLeft, r2.topLeft), np.maximum(r1.bottomRight, r2.bottomRight))
# 		return r

# 	def RangeFromRec(r):
# 		xmin, ymin = np.maximum(np.zeros((2)), np.minimum(np.array([movieRow-1, movieCol-1]), np.floor(r.topLeft))).astype(np.uint32)
# 		xmax, ymax = np.maximum(np.zeros((2)), np.minimum(np.array([movieRow-1, movieCol-1]), np.ceil(r.bottomRight))).astype(np.uint32)
# 		return (xmin, xmax), (ymin, ymax)

# 	class roi_extractor(object):
# 		def __init__(self, tStart, ROIs, cellIndexes):
# 			self.ROIs = ROIs
# 			self.cellIndexes = cellIndexes
# 			self.tStart = tStart
# 			self.currentIndex = 0

# 			r = rectangle(topLeft=np.array([movieRow, movieCol]), bottomRight=np.zeros((2)))
# 			for i, index in enumerate(cellIndexes):
# 				r = AddRec(r, RecFromROI(ROIs[i].positions[index, :], roiExpansionFactor*ROIs[i].radii[index]))
# 			self.rec = r

# 		def extractNextROI(self):
# 		 	(total_xmin, total_xmax), (total_ymin, total_ymax) = RangeFromRec(self.rec)
# 		 	tmpFrame = np.zeros((total_xmax-total_xmin, total_ymax-total_ymin))

# 		 	frameRec = RecFromROI(self.ROIs[self.currentIndex].positions[self.cellIndexes[self.currentIndex], :], roiExpansionFactor*self.ROIs[self.currentIndex].radii[self.cellIndexes[self.currentIndex]])
		 	
# 		 	(xmin, xmax), (ymin, ymax) = RangeFromRec(frameRec)

# 		 	tmpFrame[xmin-total_xmin:xmax-total_xmin, ymin-total_ymin:ymax-total_ymin] = movie[xmin:xmax, ymin:ymax, self.tStart + self.currentIndex]

# 		 	tmpFrame = Image.fromarray(tmpFrame)
# 		 	tmpFrame = tmpFrame.resize((row, col), Image.ANTIALIAS)
# 		 	tmpFrame = np.array(tmpFrame)
# 		 	m, s = np.mean(tmpFrame.flatten()), np.std(tmpFrame.flatten())
# 		 	tmpFrame = (tmpFrame - m) / s

# 		 	self.currentIndex += 1

# 		 	return tmpFrame

# 	nFrames_movie = len(rois)
# 	model = load_model('TrackingModel.h5')
	
# 	nCells = rois[0].radii.shape[0]
# 	for iFrame in range(1, nFrames_movie):
# 		costMat = np.zeros((nCells, nCells), dtype=np.float)
# 		for iCell in range(nCells):
# 			for jCell in range(nCells):
# 				inputMovie = np.zeros((1, nFrames, row, col, 1), dtype=np.float)
# 				jFrame = max(0, iFrame - nFrames + 1)

# 				roiExtractor = roi_extractor(jFrame, rois[jFrame:(iFrame+1)], [iCell for j in range(iFrame - jFrame)] + [jCell])

# 				offset = nFrames - (iFrame - jFrame + 1)
# 				for kFrame in range(jFrame, iFrame + 1):
# 					inputMovie[0, offset, :, :, 0] = roiExtractor.extractNextROI()
# 					offset += 1

# 				p = model.predict(inputMovie)[0, :]
# 				norm = np.linalg.norm(p)

# 				costMat[iCell, jCell] = p[1] / norm

# 		row_ind, col_ind = linear_sum_assignment(costMat)

# 		rois[iFrame].positions = rois[iFrame].positions[col_ind, :]
# 		rois[iFrame].radii = rois[iFrame].radii[col_ind]

# 	WriteMovie(data=movie, roiData=rois_rand, name='trackingTest_randomised.mp4')
# 	WriteMovie(data=movie, roiData=rois, name='trackingTest_tracked.mp4')

# def Predict_TrackerOnly_WithDisapearances():
# 	movie, rois = GenerateMovie(numFrames=128, numCells=4, InitVelSD=1, VelScaleSD=1, AngleSDRad=np.radians(20), WriteMovieFile=False)

# 	probPerCellPerFrameToDis = 0.3

# 	#nDis = 0
# 	for i, roi in enumerate(rois):
# 		n = roi.positions.shape[0]
# 		p = np.random.permutation(np.arange(n))
# 		#if np.random.rand() < 0.1:
# 		#if i == 1 or i == 5:
# 			#nDis += 1
# 		r = np.random.rand()
# 		toDel = max(np.where(r <  probPerCellPerFrameToDis**np.arange(n+1))[0])

# 		p = p[:(n-toDel)]

# 		roi.positions = roi.positions[p, :]
# 		roi.radii = roi.radii[p]

# 	rois_rand = [positions_and_radii(roi.positions.copy(), roi.radii.copy()) for roi in rois]

# 	#print("Fraction of frames with missing detections: {0:.2f}%".format(100.0 * nDis / len(rois)))

# 	nFrames = 11
# 	row, col = 64, 64
# 	roiExpansionFactor = 1.5
# 	movieRow, movieCol = movie.shape[0:2]

# 	class track_data(object):
# 		def __init__(self, initROIs):
# 			self.rois = [positions_and_radii(initROIs.positions.copy(), initROIs.radii.copy())]
# 			n = initROIs.positions.shape[0]
# 			self.timepoints = [0 for i in range(n)]
# 			self.trackIndexes = [i for i in range(n)]
# 			self.roiIndexes_plusone = [i for i in range(1, n + 1)]

# 		def GetAvail(self, timepoint):
# 			m = coo_matrix((self.roiIndexes_plusone, (self.trackIndexes, self.timepoints)))
# 			m = csc_matrix(m)
# 			idx = m[:, max(0, timepoint-nFrames+2):(timepoint+1)].toarray()
# 			idx = np.where(np.any(idx > 0, axis=1))[0]
# 			return idx

# 		def GetROIs(self, trackIDs=None, timepoints=None):
# 			if trackIDs is None:
# 				trackIDs = np.arange(max(self.trackIndexes) + 1)

# 			if timepoints is None:
# 				timepoints = (0, max(self.timepoints))

# 			nTracks = len(trackIDs)
# 			nTimepoints = timepoints[1] - timepoints[0] + 1
# 			result = [positions_and_radii(-1 * np.ones((nTracks, 2), dtype=np.float), -1 * np.ones((nTracks), dtype=np.float)) for i in range(nTimepoints)]

# 			m = coo_matrix((self.roiIndexes_plusone, (self.trackIndexes, self.timepoints)))
# 			m = csr_matrix(m)
# 			m = m[trackIDs, :]
# 			m = csc_matrix(m)
# 			m = m[:, timepoints[0]:(timepoints[1] + 1)]
# 			m = m.toarray()

# 			for iTrack in range(nTracks):
# 				for iTimePoint in range(nTimepoints):
# 					if m[iTrack, iTimePoint] > 0:
# 						result[iTimePoint].positions[iTrack, :] = self.rois[iTimePoint + timepoints[0]].positions[
# 																  m[iTrack, iTimePoint] - 1, :]
# 						result[iTimePoint].radii[iTrack] = self.rois[iTimePoint + timepoints[0]].radii[
# 							m[iTrack, iTimePoint] - 1]

# 			return result

# 		def AddDetections(self, trackIDs, newRois, timepoint):
# 			if len(newRois) > 0:
# 				if len(self.rois) <= timepoint:
# 					offset = 0
# 					self.rois.insert(timepoint, newRois[0])
# 				else:
# 					offset = self.rois[timepoint].positions.shape[0]
# 					self.rois[timepoint].positions = np.concatenate((self.rois[timepoint].positions, newRois[0].positions),
# 																	axis=0)
# 					self.rois[timepoint].radii = np.concatenate((self.rois[timepoint].radii, newRois[0].radii), axis=0)

# 				for track in trackIDs:
# 					offset += 1
# 					self.timepoints.append(timepoint)
# 					self.trackIndexes.append(track)
# 					self.roiIndexes_plusone.append(offset)

# 		def AddNewTracks(self, newRois, timepoint):
# 			if len(newRois) > 0:
# 				t = max(self.trackIndexes)
# 				newTracks = [i + t for i in range(1, newRois[0].positions.shape[0] + 1)]
# 				self.AddDetections(newTracks, newRois, timepoint)

# 	tracks = track_data(rois[0])

# 	def NumCells(timepoint):
# 		return rois[timepoint].positions.shape[0]

# 	def GetCellROIs(cellIDx, timepoint):
# 		if type(cellIDx) is np.ndarray:
# 			if len(cellIDx) == 0:
# 				return []
# 			else:
# 				return [positions_and_radii(rois[timepoint].positions[cellIDx, :], rois[timepoint].radii[cellIDx])]
# 		else:
# 			return [positions_and_radii(rois[timepoint].positions[cellIDx:(cellIDx+1), :], rois[timepoint].radii[cellIDx:(cellIDx+1)])]

# 	class rectangle(object):
# 		def __init__(self, topLeft=np.zeros((2, 1), dtype=np.float), bottomRight=np.zeros((2, 1), dtype=np.float)):
# 			self.topLeft = topLeft
# 			self.bottomRight = bottomRight

# 	def RecFromROI(centre, radius):
# 		r = rectangle(centre - radius, centre + radius)
# 		return r

# 	def AddRec(r1, r2):
# 		r = rectangle(np.minimum(r1.topLeft, r2.topLeft), np.maximum(r1.bottomRight, r2.bottomRight))
# 		return r

# 	def RangeFromRec(r):
# 		xmin, ymin = np.maximum(np.zeros((2)),
# 								np.minimum(np.array([movieRow - 1, movieCol - 1]), np.floor(r.topLeft))).astype(np.uint32)
# 		xmax, ymax = np.maximum(np.zeros((2)),
# 								np.minimum(np.array([movieRow - 1, movieCol - 1]), np.ceil(r.bottomRight))).astype(np.uint32)
# 		return (xmin, xmax), (ymin, ymax)

# 	class roi_extractor(object):
# 		def __init__(self, tStart, ROIs, cellIndexes):
# 			self.ROIs = ROIs
# 			self.cellIndexes = cellIndexes
# 			self.tStart = tStart
# 			self.currentIndex = 0

# 			r = rectangle(topLeft=np.array([movieRow, movieCol]), bottomRight=np.zeros((2)))
# 			for i, index in enumerate(cellIndexes):
# 				if ROIs[i].radii[index] >= 0.0:
# 					r = AddRec(r, RecFromROI(ROIs[i].positions[index, :], roiExpansionFactor * ROIs[i].radii[index]))
# 			self.rec = r

# 		def extractNextROI(self):
# 			(total_xmin, total_xmax), (total_ymin, total_ymax) = RangeFromRec(self.rec)
# 			tmpFrame = np.zeros((total_xmax - total_xmin, total_ymax - total_ymin))

# 			if self.ROIs[self.currentIndex].radii[self.cellIndexes[self.currentIndex]] >= 0:

# 				frameRec = RecFromROI(self.ROIs[self.currentIndex].positions[self.cellIndexes[self.currentIndex], :], roiExpansionFactor * self.ROIs[self.currentIndex].radii[self.cellIndexes[self.currentIndex]])

# 				(xmin, xmax), (ymin, ymax) = RangeFromRec(frameRec)

# 				tmpFrame[xmin - total_xmin:xmax - total_xmin, ymin - total_ymin:ymax - total_ymin] = movie[xmin:xmax, ymin:ymax, self.tStart + self.currentIndex]

# 			tmpFrame = Image.fromarray(tmpFrame)
# 			tmpFrame = tmpFrame.resize((row, col), Image.ANTIALIAS)
# 			tmpFrame = np.array(tmpFrame)
# 			m, s = np.mean(tmpFrame.flatten()), np.std(tmpFrame.flatten())
# 			if s == 0.0:
# 				s = 1.0
# 			tmpFrame = (tmpFrame - m) / s

# 			self.currentIndex += 1

# 			return tmpFrame

# 	nFrames_movie = len(rois)
# 	model = load_model('TrackingModel.h5')

# 	probs_all = []

# 	for currFrame in range(1, nFrames_movie):
# 		prevFrame = currFrame - 1
# 		availTracks = tracks.GetAvail(prevFrame)
# 		nTracksAvail = availTracks.shape[0]
# 		nCells = NumCells(currFrame)
# 		costMat = np.inf * np.ones((nTracksAvail, nCells + 1), dtype=np.float)

# 		probs = np.zeros((nTracksAvail, nCells, 3), dtype=np.float)

# 		for iTrack in range(nTracksAvail):
# 			for iCell in range(nCells):

# 				startFrame_subTrack = max(0, currFrame - nFrames + 1)

# 				inputMovie = np.zeros((1, nFrames, row, col, 1), dtype=np.float)
# 				trackRois = tracks.GetROIs([availTracks[iTrack]], (startFrame_subTrack, prevFrame))
# 				cellRoi = GetCellROIs((iCell), currFrame)
# 				roiExtractor = roi_extractor(startFrame_subTrack, trackRois + cellRoi, [0 for j in range(currFrame - startFrame_subTrack + 1)])

# 				offset = nFrames - (currFrame - startFrame_subTrack + 1)
# 				for kFrame in range(startFrame_subTrack, currFrame + 1):
# 					inputMovie[0, offset, :, :, 0] = roiExtractor.extractNextROI()
# 					offset += 1

# 				p = model.predict(inputMovie)[0, :]
# 				p /= np.linalg.norm(p)
# 				if p[0] > 0.01:
# 					costMat[iTrack, iCell] = p[1]

# 				probs[iTrack, iCell, 0] = availTracks[iTrack]
# 				probs[iTrack, iCell, 1] = p[0]
# 				probs[iTrack, iCell, 2] = p[1]

# 		maxCost = max(costMat[:, :nCells].flatten())
# 		if np.isinf(maxCost):
# 			maxCost = 1.0
# 		costMat[:, -1] = 1.1 * maxCost
# 		costMat[np.isinf(costMat)] = 1.2 * maxCost

# 		track_ind, cell_ind = linear_sum_assignment(costMat)
# 		track_ind = track_ind[cell_ind < nCells]
# 		cell_ind = cell_ind[cell_ind < nCells]

# 		tracks.AddDetections(availTracks[track_ind], GetCellROIs(cell_ind, currFrame), currFrame)
# 		tracks.AddNewTracks(GetCellROIs(np.setxor1d(cell_ind, np.arange(nCells), assume_unique=True), currFrame), currFrame)

# 		probs_all.append(probs)

# 		sys.stdout.write("Tracking: {0:.2f}%".format(100.0 * currFrame / (nFrames_movie-1)) + '\r')
# 		sys.stdout.flush()

# 	#WriteMovie(data=movie, roiData=rois_rand, name='trackingTest_randomised.mp4')
# 	#WriteMovie(data=movie, roiData=rois, name='trackingTest_tracked.mp4')
# 	WriteMovie(data=movie, roiData=tracks.GetROIs(), name='trackingTest_wDisappearances.mp4')

# 	return probs_all

def Predict(movie=None):
	np.random.seed(400582833)

	from frcnn_detect import Detect

	if movie is None:
		movie, _ = GenerateMovie(GenImSize=(256, 256), numFrames=16, numCells=64, InitVelSD=2, VelScaleSD=1, AngleSDRad=np.radians(20),
		radius_mean=7, radius_sd=1, radius_min=6, radius_max=10, WriteMovieFile=True)
	
	rois = Detect(movie)

	minProbForCorrectAssignment = 0.01
	stdDevCutoffForTrackAssignment = 10
	maxDisplacementPerFrame = 20
	minSearchRadius = 8
	brownianStdMult = 3.5
	closestDistScale = 2
	maxStdMult = 100

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

		for i in range(first+1, len(rois)):
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

	movie = np.pad(movie, ((0, 0), (0, 0), (nFrames-1, 0)), 'reflect', reflect_type='even')
	rois = np.pad(np.array(rois), ((nFrames-1, 0)), 'reflect', reflect_type='even').tolist()

	class KalmanFilterInfo(object):
	    def __init__(self, stateVec, stateCov, noiseVar, stateNoise, scheme):
	        self.stateVec = stateVec
	        self.stateCov = stateCov
	        self.noiseVar = noiseVar
	        self.stateNoise = stateNoise
	        self.scheme = scheme

	    def __len__(self):
	        return 1

	class KalmanFilterInfoFrame(object):
	    def __init__(self, stateVec, stateCov, obsVec):
	        self.stateVec = stateVec
	        self.stateCov = stateCov
	        self.obsVec = obsVec

	    def __len__(self):
	        return 1

	class track_data(object):
		def __init__(self, initROIs):
			self.rois = [positions_and_radii(initROIs.positions.copy(), initROIs.radii.copy())]
			n = initROIs.positions.shape[0]
			self.timepoints = [0 for i in range(n)]
			self.trackIndexes = [i for i in range(n)]
			self.roiIndexes_plusone = [i for i in range(1, n + 1)]

			##### Kalman #####
			noiseVarInit = ((maxDisplacementPerFrame / 2 / brownianStdMult) ** 2) / 2
			stateVec = np.concatenate((self.rois[0].positions, np.zeros((n, 2))), axis=1)
			stateCov = np.zeros((4, 4, n))
			noiseVar = np.zeros((4, 4, n))
			
			for iFeature in range(n):
				posVar = self.rois[0].radii[iFeature:(iFeature+1)]
				stateCov[:, :, iFeature] = np.diag(np.concatenate((posVar, posVar, 4 * np.ones(2)), axis=0))
				noiseVar[:, :, iFeature] = np.diag(noiseVarInit * np.ones((4)))

			self.kalmanFilters = [
			KalmanFilterInfo(
				stateVec, 
				stateCov, 
				noiseVar, 
				np.zeros((n, 4)), 
				np.zeros((n, 2))
				)]
			##### Kalman #####

		def GetAvail(self, timepoint):
			m = coo_matrix((self.roiIndexes_plusone, (self.trackIndexes, self.timepoints)))
			m = csc_matrix(m)
			idx = m[:, max(0, timepoint-nFrames+2):(timepoint+1)].toarray()
			idx = np.where(np.any(idx > 0, axis=1))[0]
			return idx

		def GetROIs(self, trackIDs=None, timepoints=None, removeEmptyTracks=False):
			if trackIDs is None:
				trackIDs = np.arange(max(self.trackIndexes) + 1)

			if timepoints is None:
				timepoints = (0, max(self.timepoints))

			nTracks = len(trackIDs)
			nTimepoints = timepoints[1] - timepoints[0] + 1

			m = coo_matrix((self.roiIndexes_plusone, (self.trackIndexes, self.timepoints)))
			m = csr_matrix(m)
			m = m[trackIDs, :]
			m = csc_matrix(m)
			m = m[:, timepoints[0]:(timepoints[1] + 1)]
			m = m.toarray()

			if removeEmptyTracks:
				m = np.delete(m, np.where(np.all(m==0, axis=1))[0], axis=0)
				nTracks = m.shape[0]

			result = [positions_and_radii(-1 * np.ones((nTracks, 2), dtype=np.float), -1 * np.ones((nTracks), dtype=np.float)) for i in range(nTimepoints)]

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

		def GetLastDetection(self, iTrack, timepoint):
			m = coo_matrix((self.roiIndexes_plusone, (self.trackIndexes, self.timepoints)))
			m = csr_matrix(m)
			m = m[iTrack, :]
			m = csc_matrix(m)
			m = m[:, :(timepoint+1)]
			m = m.toarray()[0, ::-1]

			result_index = -1
			result_timepoint = timepoint
			for i in range(m.shape[0]):
				result_index = m[i] - 1
				if m[i] > 0:
					break
				result_timepoint -= 1

			return result_index, result_timepoint



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
			return [positions_and_radii(rois[timepoint].positions[cellIDx:(cellIDx+1), :], rois[timepoint].radii[cellIDx:(cellIDx+1)])]

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
								np.minimum(np.array([movieCol - 1, movieRow - 1]), np.floor(r.topLeft))).astype(np.uint32)
		xmax, ymax = np.maximum(np.zeros((2)),
								np.minimum(np.array([movieCol - 1, movieRow - 1]), np.ceil(r.bottomRight))).astype(np.uint32)
		return (xmin, xmax), (ymin, ymax)

	class roi_extractor(object):
		def __init__(self, tStart, ROIs, cellIndexes):
			self.ROIs = ROIs
			self.cellIndexes = cellIndexes
			self.tStart = tStart
			self.currentIndex = 0

			r = rectangle(topLeft=np.array([movieCol, movieRow]), bottomRight=np.zeros((2)))
			for i, index in enumerate(cellIndexes):
				if ROIs[i].radii[index] >= 0.0:
					r = AddRec(r, RecFromROI(ROIs[i].positions[index, :], roiExpansionFactor * ROIs[i].radii[index]))
			self.rec = r

		def extractNextROI(self):
			(total_xmin, total_xmax), (total_ymin, total_ymax) = RangeFromRec(self.rec)
			tmpFrame = np.zeros((total_ymax - total_ymin, total_xmax - total_xmin))

			if self.ROIs[self.currentIndex].radii[self.cellIndexes[self.currentIndex]] >= 0:

				frameRec = RecFromROI(self.ROIs[self.currentIndex].positions[self.cellIndexes[self.currentIndex], :], roiExpansionFactor * self.ROIs[self.currentIndex].radii[self.cellIndexes[self.currentIndex]])

				(xmin, xmax), (ymin, ymax) = RangeFromRec(frameRec)

				tmpFrame[ymin - total_ymin:ymax - total_ymin, xmin - total_xmin:xmax - total_xmin] = movie[ymin:ymax, xmin:xmax, self.tStart + self.currentIndex]

			tmpFrame = Image.fromarray(tmpFrame)
			tmpFrame = tmpFrame.resize((row, col), Image.ANTIALIAS)
			tmpFrame = np.array(tmpFrame)
			# m, s = np.mean(tmpFrame.flatten()), np.std(tmpFrame.flatten())
			# if s == 0.0:
			# 	s = 1.0
			# tmpFrame = (tmpFrame - m) / s

			self.currentIndex += 1

			return tmpFrame

	nFrames_movie = len(rois)
	model = load_model('TrackingModel_V3.h5')

	#from Utilities.Keras.MultiGPUModel import to_multi_gpu
	#model = to_multi_gpu(model)

	def GetNNDists(roi):
		n = roi.radii.shape[0]
		if n == 0:
			nnDist = np.zeros((0, 1))
		elif n == 1:
			nnDist = np.array([[999999999]])
		else:
			p = roi.positions.copy()
			p[p < 0] = np.nan
			nnDist = distance.squareform(distance.pdist(p))
			nnDist = np.sort(nnDist, axis=1)
			nnDist = np.reshape(nnDist[:, 1], (nnDist.shape[0], 1))

		return nnDist

	##### Start Kalman Init #####
	# numFeatures = np.array([roi.radii.shape[0] for roi in rois])
	# vecSize = 4
	# kalmanFilterInfo = [KalmanFilterInfo(np.zeros((numFeatures[iFrame], vecSize)), 
	# 	np.zeros((vecSize, vecSize, numFeatures[iFrame])), 
	# 	np.zeros((vecSize, vecSize, numFeatures[iFrame])), 
	# 	np.zeros((numFeatures[iFrame], vecSize)), 
	# 	np.zeros((numFeatures[iFrame], 2))) 
	# for iFrame in range(nFrames_movie)]

	# noiseVarInit = ((maxDisplacementPerFrame / 2 / brownStdMult) ** 2) / 2
	# velocityInit = np.zeros((numFeatures[0], 2))
	# stateVec = np.concatenate((rois[0].positions, velocityInit), axis=1)
	# stateCov = np.zeros((4, 4, numFeatures[0]))
	# noiseVar = np.zeros((4, 4, numFeatures[0]))
	
	# for iFeature in range(numFeatures[0]):
	# 	posVar = rois[0].radii[iFeature:(iFeature+1)]
	# 	stateCov[:, :, iFeature] = np.diag(np.concatenate((posVar, posVar, 4 * scipy.ones(2)), axis=0))
	# 	noiseVar[:, :, iFeature] = np.diag(noiseVarInit * scipy.ones((4)))

	# trackedFeatureIndx = np.arange(numFeatures[0]).astype(scipy.int32)
 #    trackedFeatureIndx = np.reshape(trackedFeatureIndx, (trackedFeatureIndx.shape[0], 1))

 #    prevCost = np.nan * np.ones((numFeatures[0], 1))
 #    prevCostStruct = PrevCost(prevCost, np.array([np.amax(prevCost.flatten())]), np.array([]))
 #    featLifetime = np.ones((numFeatures[0], 1))

    #nnDistFeatures = GetNNDists(tracks.rois[0])
	observationMat = np.concatenate((np.eye(2), np.zeros((2, 2))), axis=1)
    #####  End  Kalman Init #####
	
	probs_all = []

	for currFrame in range(1, nFrames_movie):
		prevFrame = currFrame - 1
		
		startFrame_subTrack = max(0, currFrame - nFrames + 1)

		availTracks = tracks.GetAvail(prevFrame)
		nTracksAvail = availTracks.shape[0]
		nCells = NumCells(currFrame)
		costMat = -1 * np.ones((nTracksAvail, nCells), dtype=np.float)

		probs = -1 * np.ones((nTracksAvail, nCells, 3), dtype=np.float)

		##### Start Kalman #####
		#tmpNN = max(1, nnDistFeatures.shape[1] - nFrames)
		#nnDistTracks = np.nanmin(nnDistFeatures[:, tmpNN - 1:], axis=1)
		#nnDistTracks = np.reshape(nnDistTracks, (nnDistTracks.shape[0], 1))
		kalmanCurrFrame = []
		propagationSchemes = np.ones((nTracksAvail, nCells), dtype=np.int32)
		#####  End  Kalman #####

		for iTrack in range(nTracksAvail):
			
			##### Start Kalman ##### 
			iFeat, iTime = tracks.GetLastDetection(availTracks[iTrack], prevFrame)
			kalmanPrevFrame = tracks.kalmanFilters[iTime]
			dt = currFrame - iTime
			numSchemes = (2 * dt) + 1
			kalmanTrack = KalmanFilterInfoFrame(
				np.zeros((4, numSchemes)),
				np.zeros((4, 4, numSchemes)),
				np.zeros((2, numSchemes)))

			transMat = np.zeros((4, 4, numSchemes))
			for i, dt_i in enumerate(range(-dt, dt+1)):
				transMat[:, :, i] = np.eye(4) + np.diag(dt_i * np.ones(2), 2)

			stateOld = np.mat(kalmanPrevFrame.stateVec[iFeat, :]).transpose()
			stateCovOld = np.mat(kalmanPrevFrame.stateCov[:, :, iFeat])
			noiseVar = np.mat(np.absolute(kalmanPrevFrame.noiseVar[:, :, iFeat]))

			for iScheme in range(numSchemes):
				stateVec = transMat[:, :, iScheme] * stateOld
				stateCov = transMat[:, :, iScheme] * stateCovOld * transMat[:, :, iScheme].transpose() + noiseVar
				obsVec = observationMat * stateVec
				kalmanTrack.stateVec[:, iScheme] = stateVec.transpose()
				kalmanTrack.stateCov[:, :, iScheme] = stateCov
				kalmanTrack.obsVec[:, iScheme] = obsVec.transpose()
			
			propagatedPos = kalmanTrack.obsVec.transpose()
			kalmanCurrFrame.insert(iTrack, kalmanTrack)

			trackRois = tracks.GetROIs(availTracks, (startFrame_subTrack, prevFrame))
			nnDis = np.concatenate([GetNNDists(roi)[iTrack, :] for roi in trackRois], axis=0)
			if np.all(np.isnan(nnDis)):
				nnDist = np.nan
			else:
				nnDis = np.nanmin(nnDis)

			notFirstAppearance = kalmanPrevFrame.noiseVar[0, 0, iFeat] >= 0
			kalmanStd = max(np.sqrt(2 * np.absolute(kalmanPrevFrame.noiseVar[0, 0, iFeat])), 1e-10)
			ratioDist2Std = nnDis / kalmanStd / closestDistScale
			if not np.isnan(ratioDist2Std) and ratioDist2Std > maxStdMult:
				ratioDist2Std = maxStdMult
			stdMult = np.nanmax([ratioDist2Std, brownianStdMult])
			searchRadius = stdMult * kalmanStd
			if notFirstAppearance and searchRadius > maxDisplacementPerFrame:
				searchRadius = maxDisplacementPerFrame
			if notFirstAppearance and searchRadius < minSearchRadius:
				searchRadius = minSearchRadius

			for iCell in range(nCells):

				cellRoi = GetCellROIs((iCell), currFrame)

				dis = distance.cdist(cellRoi[0].positions, propagatedPos)
				propagationSchemes[iTrack, iCell] = np.argmin(dis)
				dis = np.amin(dis)

				#####  End  Kalman #####

				#disp = GetCentreDisp(trackRois + cellRoi)
				# if disp[-1] > 1e-6:
				# 	disp = disp[disp > 1e-6]
				# 	disp, indx = np.unique(disp, return_inverse=True)
				# 	disp = disp[np.concatenate((np.setxor1d(indx[-1:], np.arange(len(disp))), indx[-1:]))]
				# 	disp, _, _ = IterativeRobustMean(disp, k=stdDevCutoffForTrackAssignment)

				# if not np.isnan(disp[-1]):
				#if disp[-1] < maxDisplacementPerFrame:
				if dis < searchRadius:

					trackRois_ = tracks.GetROIs([availTracks[iTrack]], (startFrame_subTrack, prevFrame))
					inputMovie = np.zeros((1, nFrames, row, col, 1), dtype=np.float)
					roiExtractor = roi_extractor(startFrame_subTrack, trackRois_ + cellRoi, [0 for j in range(currFrame - startFrame_subTrack + 1)])

					offset = nFrames - (currFrame - startFrame_subTrack + 1)
					for kFrame in range(startFrame_subTrack, currFrame + 1):
						inputMovie[0, offset, :, :, 0] = roiExtractor.extractNextROI()
						offset += 1

					m, s = np.mean(inputMovie.flatten()), np.std(inputMovie.flatten())
					if s == 0.0:
						s = 1.0
					inputMovie = (inputMovie - m) / s

					p = model.predict(inputMovie)[0, :]
					p /= np.linalg.norm(p)
					#if p[0] > minProbForCorrectAssignment:
					costMat[iTrack, iCell] = p[1]

					probs[iTrack, iCell, 0] = availTracks[iTrack]
					probs[iTrack, iCell, 1] = p[0]
					probs[iTrack, iCell, 2] = p[1]

		maxCost = max(costMat.flatten())
		if maxCost < 0:
			maxCost = 1.0

		deathBlock = -1 * np.ones((nTracksAvail, nTracksAvail), dtype=np.float)
		np.fill_diagonal(deathBlock, 1.1 * maxCost)
		costMat = np.concatenate((costMat, deathBlock), axis=1)
		costMat[costMat < 0] = 1.2 * maxCost

		track_ind, cell_ind = linear_sum_assignment(costMat)
		track_ind = track_ind[cell_ind < nCells]
		cell_ind = cell_ind[cell_ind < nCells]

		##### Kalman Update #####
		n = len(track_ind)
		kalmanNew = KalmanFilterInfo(
				np.zeros((nCells, 4)), 
				np.zeros((4, 4, nCells)), 
				np.zeros((4, 4, nCells)), 
				np.zeros((nCells, 4)), 
				np.zeros((nCells, 2))
				)
		for trackId, cellId, idx in zip(track_ind, cell_ind, range(n)):
			iFeat, iTime = tracks.GetLastDetection(availTracks[trackId], prevFrame)
			
			cellRoi = GetCellROIs(cellId, currFrame)[0]
			obsVar = cellRoi.radii[[0, 0]]
			obs = cellRoi.positions

			kalmanTrack = kalmanCurrFrame[trackId]
			iScheme = propagationSchemes[trackId, cellId]
			stateVecOld = np.mat(kalmanTrack.stateVec[:, iScheme]).transpose()
			stateCovOld = np.mat(kalmanTrack.stateCov[:, :, iScheme])
			obsVecOld = np.mat(kalmanTrack.obsVec[:, iScheme]).transpose()

			kalmanGain = np.mat(np.linalg.solve((
				observationMat * stateCovOld * observationMat.transpose() + 
				np.diag(np.spacing(1) + obsVar)).transpose(), 
			(stateCovOld * observationMat.transpose()).transpose())).transpose()

			stateNoise = kalmanGain * (obs.transpose() - obsVecOld)
			tracks.kalmanFilters[iTime].stateNoise[iFeat, :] = np.array(stateNoise.transpose())[0, :]
			stateVec = stateVecOld + stateNoise
			stateCov = stateCovOld - (kalmanGain * observationMat * stateCovOld)

			tmpTrackRoi = tracks.GetROIs([availTracks[trackId]], (0, prevFrame))
			stateNoiseAll = []
			for i in range(currFrame):
				if tmpTrackRoi[i].radii[0] >= 0:
					tmpFeat, _ = tracks.GetLastDetection(availTracks[trackId], i)
					stateNoiseAll.append(tracks.kalmanFilters[i].stateNoise[tmpFeat, :].tolist())
			stateNoiseAll = np.array(stateNoiseAll)

			noiseVar = np.zeros((4))
			noiseVar[:2] = np.var(stateNoiseAll[:, :2].flatten(), ddof=1)
			noiseVar[2:] = np.var(stateNoiseAll[:, 2:].flatten(), ddof=1)

			kalmanNew.stateVec[idx, :] = np.array(stateVec.transpose())[0, :]
			kalmanNew.stateCov[:, :, idx] = np.array(stateCov)
			kalmanNew.noiseVar[:, :, idx] = np.diag(noiseVar)

		##### Kalman Update #####

		tracks.AddDetections(availTracks[track_ind], GetCellROIs(cell_ind, currFrame), currFrame)
		newROIs = GetCellROIs(np.setxor1d(cell_ind, np.arange(nCells), assume_unique=True), currFrame)
		tracks.AddNewTracks(newROIs, currFrame)

		if len(newROIs) > 0:
			tmpTracks = track_data(newROIs[0])
			kalmanNew.stateVec[n:, :] = tmpTracks.kalmanFilters[0].stateVec
			kalmanNew.stateCov[:, :, n:] = tmpTracks.kalmanFilters[0].stateCov
			kalmanNew.noiseVar[:, :, n:] = tmpTracks.kalmanFilters[0].noiseVar

		tracks.kalmanFilters.insert(currFrame, kalmanNew)

		probs_all.append(probs)

		sys.stdout.write("Tracking: {0:.2f}%".format(100.0 * currFrame / (nFrames_movie-1)) + '\r')
		sys.stdout.flush()

	# WriteMovie(data=movie[:, :, nFrames-1:], roiData=tracks.GetROIs(timepoints=[nFrames-1, nFrames_movie-1], removeEmptyTracks=True), name='trackingTest_wDetector.mp4')

	print("Tracking: {0:.2f}%".format(100.0 * currFrame / (nFrames_movie-1)))

	return tracks.GetROIs(timepoints=[nFrames-1, nFrames_movie-1], removeEmptyTracks=True), probs_all

if __name__ == "__main__":
	#Predict_TrackerOnly()
	Predict()
