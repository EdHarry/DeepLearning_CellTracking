import sys, os, numpy as np
#sys.path.append(os.path.dirname(__file__) + "../MovieGenerator")
#sys.path.append(os.path.dirname(__file__) + "../keras-frcnn")
#from MovieGenerator import GenerateMovie
from PIL import Image, ImageDraw

GenImSize = 128

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


movie, rois = GenerateMovie()

nSamples = 3000
nFrames = 16
row, col = 64, 64

movieRow, movieCol = movie.shape[0:2]

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

def WidthHeightFromRec(r):
	return r.bottomRight - r.topLeft

class roi_extractor(object):
	def __init__(self, tStart, ROIs, cellIndexes):
		self.ROIs = ROIs
		self.cellIndexes = cellIndexes
		self.tStart = tStart
		self.currentIndex = 0

		r = rectangle(topLeft=np.array([movieRow, movieCol]), bottomRight=np.zeros((2)))
		for i, index in enumerate(cellIndexes):
			r = AddRec(r, RecFromROI(ROIs[i].positions[index, :], ROIs[i].radii[index]))
		self.rec = r

	def extractNextROI(self):
		tmpFrame = np.zeros(np.ceil(WidthHeightFromRec(self.rec)).astype(np.uint32))
		frameRec = RecFromROI(self.ROIs[self.currentIndex].positions[self.cellIndexes[self.currentIndex], :], self.ROIs[self.currentIndex].radii[self.cellIndexes[self.currentIndex]])

		xmin, ymin = np.maximum(np.zeros((2)), np.minimum(np.array([movieRow-1, movieCol-1]), np.floor(frameRec.topLeft))).astype(np.uint32)
		xmax, ymax = np.maximum(np.zeros((2)), np.minimum(np.array([movieRow-1, movieCol-1]), np.floor(frameRec.bottomRight))).astype(np.uint32)

		#tmpFrame = np.zeros((xmax-xmin+1, ymax-ymin+1))

		offset_x, offset_y = np.maximum(np.zeros((2)), np.minimum(np.array([movieRow-1, movieCol-1]), np.floor(self.rec.topLeft))).astype(np.uint32)

		try:
			tmpFrame[xmin-offset_x:xmax-offset_x, ymin-offset_y:ymax-offset_y] = movie[xmin:xmax, ymin:ymax, self.tStart]
		except:
			bh = 1

		tmpFrame = Image.fromarray(tmpFrame)
		tmpFrame = tmpFrame.resize((row, col), Image.ANTIALIAS)
		tmpFrame = np.array(tmpFrame)
		m, s = np.mean(tmpFrame.flatten()), np.std(tmpFrame.flatten())
		tmpFrame = (tmpFrame - m) / s

		self.currentIndex += 1

		return tmpFrame

nFrames_movie = len(rois)
nSamples_per_SubTrack = nSamples // ((nFrames-1) * 2)
nCells = rois[0].radii.shape[0]

inputMovies = np.zeros((nSamples, nFrames, row, col, 1), dtype=np.float)
labels = np.zeros((nSamples, 1))

iSample = 0
for lenSubTrack in range(2, nFrames+1):
	for i in range(nSamples_per_SubTrack):
		subTrack_start = np.random.randint(low=0, high=nFrames_movie-lenSubTrack+1)
		
		iCell_correct = np.random.randint(low=0, high=nCells)
		iCell_incorrect = (iCell_correct + 1) % nCells

		roiExtractor_correct = roi_extractor(subTrack_start, rois[subTrack_start:(subTrack_start+lenSubTrack)], [iCell_correct for j in range(lenSubTrack)])
		roiExtractor_incorrect = roi_extractor(subTrack_start, rois[subTrack_start:(subTrack_start+lenSubTrack)], [iCell_correct for j in range(lenSubTrack-1)] + [iCell_incorrect])

		for frame in range(lenSubTrack-1):
			frame_inputMovie = frame + nFrames - lenSubTrack

			inputMovies[iSample, frame_inputMovie, :, :, 0] = roiExtractor_correct.extractNextROI()
			inputMovies[iSample+1, frame_inputMovie, :, :, 0] = roiExtractor_incorrect.extractNextROI()

		frame_inputMovie = nFrames - 1

		inputMovies[iSample, frame_inputMovie, :, :, 0] = roiExtractor_correct.extractNextROI()
		inputMovies[iSample+1, frame_inputMovie, :, :, 0] = roiExtractor_incorrect.extractNextROI()

		labels[iSample+1] = 1

		iSample += 2