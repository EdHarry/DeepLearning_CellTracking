import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../..")
from Utilities.graphTracks.generateCojoinedTracks import generateCojoinedTracks
from Utilities.graphTracks.findTrackConflicts import findTrackConflicts
import numpy as np
from scipy.sparse import csc_matrix

def test():
	idx = np.array([[2, 0, 2, 3, 1],
       [2, 1, 1, 3, 0],
       [3, 3, 3, 3, 2],
       [1, 3, 0, 0, 1],
       [2, 0, 2, 3, 3],
       [2, 0, 0, 1, 1],
       [2, 1, 3, 0, 2],
       [3, 3, 3, 0, 0],
       [0, 0, 2, 0, 0],
       [0, 1, 2, 1, 1],
       [2, 3, 1, 0, 0],
       [0, 0, 2, 2, 1],
       [0, 0, 0, 2, 1],
       [3, 2, 3, 2, 0],
       [0, 0, 0, 3, 3],
       [0, 0, 1, 1, 3]])

	ret = generateCojoinedTracks(idx)
	idx = np.concatenate((idx, ret.extraTracks), axis=0)

	confs = findTrackConflicts(csc_matrix(idx))

	BreakHere = 1

if __name__ == "__main__":
	test()