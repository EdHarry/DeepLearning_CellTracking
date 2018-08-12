import scipy
from Utrack.getTrackSEL import getTrackSEL as GetSEL

def FilterByLength(tracks, minLength):
    sel = GetSEL(tracks)
    select = sel[:, 2] >= minLength

    tracksReturn = scipy.array(tracks)[select].tolist()

    return tracksReturn