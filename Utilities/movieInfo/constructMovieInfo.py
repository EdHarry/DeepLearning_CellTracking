import scipy

from Utrack.utilityClasses import MovieInfo

__author__ = 'edwardharry'


class MovieInfoReturnValue(object):
    def __init__(self, movieInfo, indexMap):
        self.movieInfo = movieInfo
        self.indexMap = indexMap


def constructMovieInfo(indexMat, coordMat, probDim):
    nFrames = indexMat.shape[1]

    if probDim == 2:
        movieInfo = [MovieInfo([], [], []) for i in range(nFrames)]
    else:
        movieInfo = [MovieInfo([], [], [], []) for i in range(nFrames)]

    indexMap = []

    for iFrame in range(nFrames):
        indexes = indexMat[:, iFrame]
        coords = coordMat[:, 8 * iFrame:(8 * iFrame) + 8]
        x = scipy.concatenate((coords[:, ::8], coords[:, 4::8]), axis=1)
        y = scipy.concatenate((coords[:, 1::8], coords[:, 5::8]), axis=1)
        z = scipy.concatenate((coords[:, 2::8], coords[:, 6::8]), axis=1)
        amp = scipy.concatenate((coords[:, 3::8], coords[:, 7::8]), axis=1)

        goodCoord = ~scipy.isnan(x[:, 0])
        x[scipy.isnan(x[:, 1]), 1] = 0
        y[scipy.isnan(y[:, 1]), 1] = 0
        z[scipy.isnan(z[:, 1]), 1] = 0
        amp[scipy.isnan(amp[:, 0]), :] = 0

        movieInfo[iFrame].xCoord = x[goodCoord, :]
        movieInfo[iFrame].yCoord = y[goodCoord, :]

        if probDim == 3:
            movieInfo[iFrame].zCoord = z[goodCoord, :]

        movieInfo[iFrame].amp = amp[goodCoord, :]
        indexMap.append(indexes[goodCoord])

    return MovieInfoReturnValue(movieInfo, indexMap)
