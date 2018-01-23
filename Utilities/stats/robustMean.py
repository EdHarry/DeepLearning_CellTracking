import warnings

import scipy
from scipy.optimize import fmin

__author__ = 'edwardharry'


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
    blowUpDataSize = dataSize / reducedDataSize
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
