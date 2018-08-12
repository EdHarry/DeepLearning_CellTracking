import sys, os, numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../keras-frcnn")
from MovieGenerator.MovieGenerator import GenerateMovie, WriteMovie
from PIL import Image

movie, rois = GenerateMovie(numFrames=16, numCells=16, InitVelSD=2, VelScaleSD=1, AngleSDRad=np.radians(20), 
		radius_mean=7, radius_sd=1, radius_min=6, radius_max=10, WriteMovieFile=False)

nSamples = 1000
nFrames = 11
nMovies = 10

assert((nSamples % ((nFrames-1) * 2 * nMovies)) == 0)

row, col = 64, 64
roiExpansionFactor = 1.5

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

def RangeFromRec(r):
	xmin, ymin = np.maximum(np.zeros((2)), np.minimum(np.array([movieCol-1, movieRow-1]), np.floor(r.topLeft))).astype(np.uint32)
	xmax, ymax = np.maximum(np.zeros((2)), np.minimum(np.array([movieCol-1, movieRow-1]), np.ceil(r.bottomRight))).astype(np.uint32)
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
		#m, s = np.mean(tmpFrame.flatten()), np.std(tmpFrame.flatten())
		#if s == 0.0:
		#	s = 1.0
		#tmpFrame = (tmpFrame - m) / s

		self.currentIndex += 1

		return tmpFrame

nFrames_movie = len(rois)
nSamples_per_SubTrack = nSamples // ((nFrames-1) * 2 * nMovies)
nCells = rois[0].radii.shape[0]

def CreateInputMovies():
	inputMovies = np.zeros((nSamples, nFrames, row, col, 1), dtype=np.float)
	iSample = 0
	for iMovie in range(nMovies):
		# if iMovie > 0:
		movie, rois = GenerateMovie(numFrames=16, numCells=16, InitVelSD=2, VelScaleSD=1, AngleSDRad=np.radians(20), 
			radius_mean=7, radius_sd=1, radius_min=6, radius_max=10, WriteMovieFile=False)

		for lenSubTrack in range(2, nFrames+1):
			for i in range(nSamples_per_SubTrack):
				subTrack_start = np.random.randint(low=0, high=nFrames_movie-lenSubTrack+1)
				
				iCell_correct = np.random.randint(low=0, high=nCells)
				iCell_incorrect = (iCell_correct + 1) % nCells

				roiExtractor_correct = roi_extractor(subTrack_start, rois[subTrack_start:(subTrack_start+lenSubTrack)], [iCell_correct for j in range(lenSubTrack)])
				roiExtractor_incorrect = roi_extractor(subTrack_start, rois[subTrack_start:(subTrack_start+lenSubTrack)], [iCell_correct for j in range(lenSubTrack-1)] + [iCell_incorrect])

				for frame in range(lenSubTrack):
					frame_inputMovie = frame + nFrames - lenSubTrack

					inputMovies[iSample, frame_inputMovie, :, :, 0] = roiExtractor_correct.extractNextROI()
					inputMovies[iSample+1, frame_inputMovie, :, :, 0] = roiExtractor_incorrect.extractNextROI()

					if frame > 0 and frame < (lenSubTrack-1) and (i % 2) == 0:
						if np.random.rand() < 0.3:
							inputMovies[iSample, frame_inputMovie, :, :, 0] = 0
						if np.random.rand() < 0.3:
							inputMovies[iSample+1, frame_inputMovie, :, :, 0] = 0

				m, s = np.mean(inputMovies[iSample, :, :, :, 0].flatten()), np.std(inputMovies[iSample, :, :, :, 0].flatten())
				inputMovies[iSample, :, :, :, 0] = (inputMovies[iSample, :, :, :, 0] - m) / s
				m, s = np.mean(inputMovies[iSample+1, :, :, :, 0].flatten()), np.std(inputMovies[iSample+1, :, :, :, 0].flatten())
				inputMovies[iSample+1, :, :, :, 0] = (inputMovies[iSample+1, :, :, :, 0] - m) / s

				iSample += 2

				sys.stdout.write("Generating Data: {0:.2f}%".format(100.0 * iSample / nSamples) + '\r')
				sys.stdout.flush()

	return inputMovies

#WriteMovie(data=inputMovies[-2, :, :, :, 0].transpose((1, 2, 0)), fps=1, name='example_roi.mp4')

# import tensorflow as tf
# tf_config = tf.ConfigProto()
# tf_config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
# sess = tf.Session(config=tf_config)

# from keras import backend as K
# K.set_session(sess)

#import tensorflow as tf
#tf_config = tf.ConfigProto()
#tf_config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
#sess = tf.Session(config=tf_config)

# from keras import backend as keras_bkend
#keras_bkend.set_session(sess)

from keras.utils import to_categorical
labels = to_categorical(np.arange(nSamples) % 2, num_classes=2)

modelName = 'TrackingModel_V4.h5'

if os.path.exists(modelName):
	from keras.models import load_model
	seq = load_model(modelName)
else:
	# assert(False)

	from keras.models import Sequential
	from keras.layers import Dense, Dropout, Flatten
	from keras.layers.convolutional import Conv3D, MaxPooling3D
	from keras.layers.convolutional_recurrent import ConvLSTM2D
	from keras.layers.normalization import BatchNormalization

	seq = Sequential()
	seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
	                   input_shape=(nFrames, row, col, 1),
	                   padding='same', return_sequences=True))
	seq.add(BatchNormalization())

	seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
	                   padding='same', return_sequences=True))
	seq.add(BatchNormalization())

	seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
	                   padding='same', return_sequences=True))
	seq.add(BatchNormalization())

	seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
	                   padding='same', return_sequences=True))
	seq.add(BatchNormalization())

	seq.add(Conv3D(filters=1, kernel_size=(3, 3, 3),
	               activation='sigmoid',
	               padding='same', data_format='channels_last'))

	seq.add(MaxPooling3D(pool_size=(2, 2, 2)))
	seq.add(Dropout(0.25))

	seq.add(Flatten())
	seq.add(Dense(256, activation='relu'))
	seq.add(Dropout(0.5))
	seq.add(Dense(2, activation='softmax'))


from keras import metrics
# from keras.models import Model
# from keras.layers import Input, merge
# from keras.layers.core import Lambda

# def slice_batch(x, n_gpu, part):
#     sh = keras_bkend.shape(x)
#     L = sh[0] // n_gpu
#     if part == n_gpu - 1:
#         result = x[part*L:]
#     else:
#         result = x[part*L:(part+1)*L]
    
#     return result

# def to_multi_gpu(model, n_gpu=2):
#     with tf.device('/cpu:0'):
#         x = Input(model.input_shape[1:], name=model.input_names[0])
#     towers = []
#     for gpu in range(n_gpu):
#         with tf.device('/gpu:' + str(gpu)):
#             slice_gpu = Lambda(slice_batch, lambda shape: shape, arguments={'n_gpu':n_gpu, 'part':gpu})(x)
#             towers.append(model(slice_gpu))
#     with tf.device('/cpu:0'):
#         merged = merge(towers, mode='concat', concat_axis=0)

#     return Model(input=[x], output=merged)

# from Utilities.Keras.MultiGPUModel import to_multi_gpu
# seq = to_multi_gpu(seq)

seq.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=[metrics.binary_accuracy])

epochesPerInput = 25
nInputs = 40

for i in range(nInputs):
	print("---\n---")
	print("Iteration {}/{}".format(i+1, nInputs))
	print("---\n---")
	
	loop = True
	while loop:
		loop = False
		try:
			inputMovies = CreateInputMovies()
		except:
			print("---\n---")
			loop = True

	seq.fit(inputMovies, labels, batch_size=10, epochs=epochesPerInput, validation_split=0.05)
	seq.save(modelName)
	x = seq.evaluate(inputMovies, labels, batch_size=10, verbose=1)
	print("Loss: {}, Binary Accuracy: {}".format(x[0], x[1]))
