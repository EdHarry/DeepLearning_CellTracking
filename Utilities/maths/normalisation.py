import scipy

__author__ = 'edwardharry'


def normList(vectors, returnNormVectors=False):
    listOfNorms = scipy.sqrt(scipy.sum(vectors ** 2, axis=1))

    if not returnNormVectors:

        return listOfNorms

    else:
        normedVectors = vectors / listOfNorms[:, scipy.newaxis]
        normedVectors[scipy.logical_and(scipy.isnan(normedVectors[:, 0]), ~scipy.isnan(vectors[:, 0])), 0] = 0
        normedVectors[scipy.logical_and(scipy.isnan(normedVectors[:, 1]), ~scipy.isnan(vectors[:, 1])), 1] = 0
        normedVectors[scipy.logical_and(scipy.isnan(normedVectors[:, 2]), ~scipy.isnan(vectors[:, 2])), 2] = 0

        return listOfNorms, normedVectors
