import sys, os, numpy as np
#sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../keras-frcnn")
#from keras_frcnn.data_augment import MakeCellImage, GenImSize
from PIL import Image, ImageDraw
from .MovieWriter import WriteMovie

class positions_and_radii:
	def __init__(self, positions, radii):
		self.positions = positions
		self.radii = radii

def GenerateMovie(GenImSize=(128, 128), numFrames=256, numCells=16, InitVelSD=2, AngleSDRad=np.radians(20), VelScaleSD=1, 
	border=10, radius_mean=7, radius_sd=1, radius_min=6, radius_max=10, intensity_sd=0.1, intensity_mean=0.5, WriteMovieFile=False):

	def MakeCellImage(x, y, r, i):
	    im = Image.new(mode='F', size=GenImSize)
	    draw = ImageDraw.Draw(im)
	    draw.ellipse(xy=[x-r, y-r, x+r, y+r], fill='White')
	    im = np.array(im).astype(np.float32)
	    im *= (i / 255.0)
	    return im

	global position
	global position_minusOne
	global radius
	global velocity

	movie = np.zeros(shape=(GenImSize[0], GenImSize[1], numFrames))

	#radius = np.random.randint(size=numCells, low=-5, high=10) + 10
	radius = (np.random.randn(numCells) * radius_sd) + radius_mean
	radius = np.clip(radius, a_min=radius_min, a_max=radius_max)

	intensity = (np.random.randn(numCells) * intensity_sd) + intensity_mean
	intensity = np.clip(intensity, a_min=0.0, a_max=1.0)

	position = np.zeros(shape=(numCells, 2))
	position_minusOne = np.zeros(shape=(numCells, 2))

	# class positions_and_radii:
	# 	def __init__(self, positions, radii):
	# 		self.positions = positions
	# 		self.radii = radii

	rois = []

	def SqDis(p1, p2):
		diff = p1 - p2
		return (diff[0] * diff[0]) + (diff[1] * diff[1])

	def overlap(p1, r1, p2, r2):
		return SqDis(p1, p2) < ((r1 + r2)**2)

	for c in range(numCells):
		goodPos = False
		i = 0
		while not goodPos and i < (numCells * numCells):
			i += 1
			goodPos = True
			#position[c, :] = np.random.randint(low=radius[c]+border, high=GenImSize-radius[c]-border, size=2)
			
			position[c, 0] = (np.random.rand() * (GenImSize[0] - (2*(border+radius[c])))) + border + radius[c]
			position[c, 1] = (np.random.rand() * (GenImSize[1] - (2*(border+radius[c])))) + border + radius[c]

			for c2 in range(c):
				if overlap(position[c, :], radius[c], position[c2, :], radius[c2]):
					radius[c] = np.clip(radius[c] - 1, a_min=5, a_max=20)
					goodPos = False
					break

	from scipy.spatial import KDTree

	nBins = 8
	flowForceFactor = 100
	def FlowForces():
		force = np.zeros(shape=(numCells, 2))

		hist, xEdge, yEdge =  np.histogram2d(position[:, 0], position[:, 1], normed=True, bins=nBins)
		grad = np.gradient(hist)

		for c in range(position.shape[0]):
			idx = np.where(position[c, 0] > xEdge)[0]
			xBin = max(idx if idx.shape[0] > 0 else [0])
			idx = np.where(position[c, 1] > yEdge)[0]
			yBin = max(idx if idx.shape[0] > 0 else [0])

			force[c, 0] = -flowForceFactor * grad[0][xBin, yBin]
			force[c, 1] = -flowForceFactor * grad[1][xBin, yBin]

		return force


	ljfFactor = 3
	def IntForces():
		#force = np.zeros(shape=(numCells, 2))
		force = FlowForces()

		maxD = max(radius) * 3
		tree = KDTree(position)
		pairs = tree.query_pairs(maxD)

		for pair in pairs:
			i, j = pair
			sigma = (radius[i] + radius[j]) * 1.001
			v = position[j, :] - position[i, :]
			r = np.linalg.norm(v)
			v_norm = v / r
			sor = sigma / r
			ljf = ljfFactor * ((sor**6) - (sor**12))
			force[i, :] += (v_norm * ljf)
			force[j, :] += (-v_norm * ljf)

		return force

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
		#global velocity

		position_minusOne = position.copy()
		outOfRange = []
		for c in range(numCells):
			x_oor = (candidate[c, 0] < (radius[c] + border)) or (candidate[c, 0] > (GenImSize[0]-radius[c]-border))
			y_oor = (candidate[c, 1] < (radius[c] + border)) or (candidate[c, 1] > (GenImSize[1]-radius[c]-border))

			if x_oor or y_oor:
				newP = (2 * position[c, :]) - candidate[c, :]
				candidate[c, :] = np.array([newP[0] if x_oor else candidate[c, 0], newP[1] if y_oor else candidate[c, 1]])
				#velocity[c, :] *= np.array([-1 if x_oor else 1, -1 if y_oor else 1])
				outOfRange.append(c)

		outOfRange = np.array(outOfRange)
		maxD = np.sum(radius[radius.argsort()[::-1][:2]])
		tree = KDTree(candidate)
		pairs = tree.query_pairs(maxD)

		for pair in pairs:
			if overlap(candidate[pair[0], :], radius[pair[0]], candidate[pair[1], :], radius[pair[1]]):
				
				oldDis = SqDis(position[pair[0], :], position[pair[1], :])
				candDis = SqDis(candidate[pair[0], :], candidate[pair[1], :])

				if candDis < oldDis:
					oor_0 = np.any(pair[0] == outOfRange)
					oor_1 = np.any(pair[1] == outOfRange)

					if oor_0 and oor_1:
						continue
					elif oor_0:
						toMove = pair[1]
						other = pair[0]
					else:
						toMove = pair[0]
						other = pair[1]

					newP = (2 * position[toMove, :]) - candidate[toMove, :]
					newDis = SqDis(newP, candidate[other, :])

					if newDis > candDis:
						candidate[toMove, :] = newP
					elif oor_0 or oor_1:
						continue
					else:
						toMove = other
						candidate[toMove, :] = (2 * position[toMove, :]) - candidate[toMove, :]

					outOfRange = outOfRange.tolist()
					outOfRange.append(toMove)
					outOfRange = np.array(outOfRange)

		position = candidate

	# def UpdatePositions(candidate):
	# 	global position
	# 	global position_minusOne
	# 	global velocity

	# 	position_minusOne = position.copy()
	# 	done = False
	# 	i = 0
	# 	while not done and i < (numCells * numCells):
	# 		i += 1
	# 		done = True
	# 		for c_i in range(numCells):
	# 			if np.any(candidate[c_i, :] < (radius[c] + border)) or candidate[c_i, 0] > (GenImSize[0]-radius[c]-border) or candidate[c_i, 1] > (GenImSize[1]-radius[c]-border):
	# 				#candidate[c_i, :] = position[c_i, :]
	# 				candidate[c_i, :] = (2 * position[c_i, :]) - candidate[c_i, :]
	# 				velocity[c_i, :] *= -1
	# 				done = False
	# 			for c_j in range(numCells):
	# 				if c_i != c_j and overlap(candidate[c_i, :], radius[c_i], candidate[c_j, :], radius[c_j]):
	# 					#candidate[c_i, :] = position[c_i, :]
	# 					#velocity[c_i, :] = (np.random.randn(1, 2) * InitVelSD) + IntForces()[c_i, :]
	# 					#candidate[c_i, :] = position[c_i, :] + velocity[c_i, :]
	# 					candidate[c_i, :] = (2 * position[c_i, :]) - candidate[c_i, :]
	# 					velocity[c_i, :] *= -1
	# 					done = False
	# 					break
	# 	position = candidate

	velDamp = 0.7
	def RandRotVel():
		global velocity

		theta = AngleSDRad * np.random.randn(numCells)
		R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
		vMag = np.linalg.norm(velocity, axis=1, keepdims=True)
		velocity /= vMag
		#velocity = np.array((R * velocity.transpose()).transpose())
		velocity = np.matmul(R.transpose((2, 0, 1)), velocity.reshape((-1, 2, 1)))[:, :, 0]
		vMag += (VelScaleSD * np.random.randn(numCells, 1))
		velocity *= (vMag * velDamp)

	def UpdateVelocityAndForces():
		global velocity
		global forces

		oldVel = velocity.copy()
		RandRotVel()
		velocity += IntForces()
		forces = velocity - oldVel

	print("---\n---")
	sys.stdout.write("Simulating: {0:.2f}%".format(0) + '\r')
	sys.stdout.flush()

	# Frame 0
	MakeImage(0)
	sys.stdout.write("Simulating: {0:.2f}%".format(100.0 * 1 / numFrames) + '\r')
	sys.stdout.flush()

	# Frame 1
	UpdatePositions(position + velocity)
	UpdateVelocityAndForces()
	MakeImage(1)
	sys.stdout.write("Simulating: {0:.2f}%".format(100.0 * 2 / numFrames) + '\r')
	sys.stdout.flush()

	# Frames 2-n
	for t in range(2, numFrames):
		UpdatePositions((2 * (forces + position)) - position_minusOne)
		UpdateVelocityAndForces()
		MakeImage(t)
		sys.stdout.write("Simulating: {0:.2f}%".format(100.0 * (t + 1) / numFrames) + '\r')
		sys.stdout.flush()
	
	print("Simulating: {0:.2f}%".format(100.0))
	print("---\n---")

	if WriteMovieFile:
		WriteMovie(data=movie, roiData=rois)

	return movie, rois

if __name__ == "__main__":
	np.random.seed(449545456)
	GenerateMovie(GenImSize=(256, 256), numFrames=256, numCells=64, InitVelSD=2, VelScaleSD=1, AngleSDRad=np.radians(10),
		radius_mean=7, radius_sd=1, radius_min=6, radius_max=10, WriteMovieFile=True)