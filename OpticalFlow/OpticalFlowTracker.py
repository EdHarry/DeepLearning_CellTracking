import sys, os, numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../flownet2-tf")
import keras
# from crfasrnn_keras.ImageSegmentation import ImageSegmenter
# from src.flownet2.flownet2 import FlowNet2
# from src.net import Mode
from .OpticalFlowVariation import LocalFlowVar

checkpoint = os.path.dirname(os.path.abspath(__file__)) + '/../flownet2-tf/checkpoints/FlowNet2/flownet-2.ckpt-0'

UNKNOWN_FLOW_THRESH = 1e7
SMALLFLOW = 0.0
LARGEFLOW = 1e8
RHO_IMAGE_SMOOTHING = 1
TRACK_SPATIAL_STEP = 4
FLOW_VARIANCE_GRAD = 0.05
FLOW_VARIANCE_CONST = 2.5
MOTION_EDGE_GRAD = 0.05
MOTION_EDGE_CONST = 0.01

h = 1
w = 1
n = 1

class Track(object):
    def __init__(self, mox=0, moy=0, t=0):
        self.mox = mox
        self.moy = moy
        self.t = t
        self.mx = []
        self.my = []
        self.stopped = False

def recursiveSmoothX(matrix, sigma):
    global h
    global w
    global n

    vals1 = np.zeros(w)
    vals2 = np.zeros(w)
    alpha = 2.5 / (np.sqrt(np.pi) * sigma)
    exp = np.exp(-alpha)
    expSqr = exp * exp
    twoExp = 2.0 * exp
    k = (1.0 - exp) * (1.0 - exp) / (1.0 + (twoExp * alpha) - expSqr)
    preMinus = exp * (alpha - 1.0)
    prePlus = exp * (alpha + 1.0)
    for y in range(h):
        vals1[0] = (0.5 - (k * preMinus)) * matrix[y, 0]
        vals1[1] = (k * (matrix[y, 1] + (preMinus * matrix[y, 0]))) + ((twoExp - expSqr) * vals1[0])
        for x in range(2, w):
            vals1[x] = (k * (matrix[y, x] + (preMinus * matrix[y, x - 1]))) + (twoExp * vals1[x - 1]) - (expSqr * vals1[x - 2])
        vals2[w - 1] = (0.5 + (k * preMinus)) * matrix[y, w - 1]
        vals2[w - 2] = (k * ((prePlus - expSqr) * matrix[y, w - 1])) + ((twoExp - expSqr) * vals2[w - 1])
        for x in range(w - 3, -1, -1):
            vals2[x] = (k * ((prePlus * matrix[y, x + 1]) - (expSqr * matrix[y, x + 2]))) + (twoExp * vals2[x + 1]) - (expSqr * vals2[x + 2])
        for x in range(w):
            matrix[y, x] = vals1[x] + vals2[x]

def recursiveSmoothY(matrix, sigma):
    global h
    global w
    global n

    vals1 = np.zeros(h)
    vals2 = np.zeros(h)
    alpha = 2.5 / (np.sqrt(np.pi) * sigma)
    exp = np.exp(-alpha)
    expSqr = exp * exp
    twoExp = 2.0 * exp
    k = (1.0 - exp) * (1.0 - exp) / (1.0 + (twoExp * alpha) - expSqr)
    preMinus = exp * (alpha - 1.0)
    prePlus = exp * (alpha + 1.0)
    for x in range(w):
        vals1[0] = (0.5 - (k * preMinus)) * matrix[x, 0]
        vals1[1] = (k * (matrix[1, x] + (preMinus * matrix[0, x]))) + ((twoExp - expSqr) * vals1[0])
        for y in range(2, h):
            vals1[y] = (k * (matrix[y, x] + (preMinus * matrix[y - 1, x]))) + ((twoExp * vals1[y - 1]) - (expSqr * vals1[y - 2]))
        vals2[h - 1] = (0.5 + (k * preMinus)) * matrix[h - 1, x]
        vals2[h - 2] = (k * ((prePlus - expSqr) * matrix[h - 1, x])) + ((twoExp - expSqr) * vals2[h - 1])
        for y in range(h - 3, -1, -1):
            vals2[y] = (k * ((prePlus * matrix[y + 1, x]) - (expSqr * matrix[y + 2, x]))) + ((twoExp * vals2[y + 1]) - (expSqr * vals2[y + 2]))
        for y in range(h):
            matrix[y, x] = vals1[y] + vals2[y]

def recursiveSmooth(matrix, sigma):
    recursiveSmoothX(matrix, sigma)
    recursiveSmoothY(matrix, sigma)

# Code fragment from Pedro Felzenszwalb  ---------------
# http://people.cs.uchicago.edu/~pff/dt/
def dt(f, n):
    d = np.zeros(n)
    v = np.zeros(n).astype(np.int32)
    z = np.zeros(n + 1)
    k = 0
    v[0] = 0
    z[0] = -10e20
    z[1] = 10e20
    
    for q in range(1, n):
        s  = ((f[q] + (q * q)) - (f[v[k]] + (v[k] * v[k]))) / (2.0 * (q - v[k]))
        while s <= z[k]:
            k -= 1
            s  = ((f[q] + (q * q)) - (f[v[k]] + (v[k] * v[k]))) / (2.0 * (q - v[k]))
        k += 1
        v[k] = q
        z[k] = s
        z[k + 1] = 10e20

    k = 0
    for q in range(n):
        while z[k + 1] < q:
            k += 1
        tmp =  q - v[k]
        d[q] = (tmp * tmp) + f[v[k]]

    return d

def euclideanDistanceTransform(matrix):
    xSize = matrix.shape[1]
    ySize = matrix.shape[0]
    
    # Transform along columns
    for y in range(ySize):
        f = matrix[y, :]
        d = dt(f, ySize)
        matrix[y, :] = d

    # Transform along rows
    for x in range(xSize):
        f = matrix[:, x]
        d = dt(f, xSize)
        matrix[:, x] = d

def GenerateTracks(movie):
    global h
    global w
    global n

    flowVar = np.zeros(movie.shape)

    h, w, n = (int(s) for s in movie.shape)

    input_a = movie[..., 0].copy()
    recursiveSmooth(input_a, RHO_IMAGE_SMOOTHING)
    covered = np.zeros(movie.shape[:2])
    tracks = []

    y, x = np.meshgrid(range(4, h - 4, TRACK_SPATIAL_STEP), range(4, w - 4, TRACK_SPATIAL_STEP))
    stepSq = TRACK_SPATIAL_STEP * TRACK_SPATIAL_STEP

    fnet = FlowNet2(mode=Mode.TEST)
    
    for t in range(n - 1):
        input_b = movie[..., t + 1].copy()
        recursiveSmooth(input_b, RHO_IMAGE_SMOOTHING)

        covered[...] = 1e20
        if t > 0:
            for track in tracks:
                if track.stopped == False:
                    covered[int(track.my[-1]), int(track.mx[-1])] = 0
            euclideanDistanceTransform(covered)

        segmenter = ImageSegmenter()
        seg = segmenter.Segment(input_a)
        keras.backend.clear_session()

        select = np.logical_and(covered[y, x] > stepSq, seg[y, x] > 0)
        tracks += [Track(mox=ax, moy=ay, t=t) for ay, ax in zip(y[select], x[select])]
        
        pred_flow_fwd = fnet.test(checkpoint=checkpoint, input_a=input_a, input_b=input_b, out_path='./', save_flo=True)
        pred_flow_bk = fnet.test(checkpoint=checkpoint, input_a=input_b, input_b=input_a, out_path='./', save_flo=True)

        flowVar_fwd = LocalFlowVar(pred_flow_fwd)
        if t == 0:
            flowVar[..., 0] = flowVar_fwd
        else:
            flowVar[..., t] = 0.5 * (flowVar_fwd + flowVar_bk)
        flowVar_bk = LocalFlowVar(pred_flow_bk)
        if t == n - 2:
            flowVar[..., -1] = flowVar_bk

        goodTracks = np.zeros((h, w))
        dy, dx = np.gradient(pred_flow_fwd, axis=(0, 1))
        motionEdge = (dx * dx).sum(axis=2) + (dy * dy).sum(axis=2)
        
        ay, ax = np.meshgrid(range(h), range(w))

        bx = ax + pred_flow_fwd[ay, ax, 0]
        by = ay + pred_flow_fwd[ay, ax, 1]
        
        x1 = bx.astype('int')
        y1 = by.astype('int')
        x2 = x1 + 1
        y2 = y1 + 1

        select = np.logical_and(x1 >= 0, np.logical_and(x2 < w, np.logical_and(y1 >= 0, y2 < h)))
        ax = ax[select]
        ay = ay[select]
        bx = bx[select]
        by = by[select]
        x1 = x1[select]
        y1 = y1[select]
        x2 = x2[select]
        y2 = y2[select]

        alphaX = bx - x1
        alphaY = by - y1

        a = ((1 - alphaX) * pred_flow_bk[y1, x1, 0]) + (alphaX * pred_flow_bk[y1, x2, 0])
        b = ((1 - alphaX) * pred_flow_bk[y2, x1, 0]) + (alphaX * pred_flow_bk[y2, x2, 0])
        u = ((1 - alphaY) * a) + (alphaY * b)

        a = ((1 - alphaX) * pred_flow_bk[y1, x1, 1]) + (alphaX * pred_flow_bk[y1, x2, 1])
        b = ((1 - alphaX) * pred_flow_bk[y2, x1, 1]) + (alphaX * pred_flow_bk[y2, x2, 1])
        v = ((1 - alphaY) * a) + (alphaY * b)

        cx = bx + u
        cy = by + v
        u2 = pred_flow_fwd[ay, ax, 0]
        v2 = pred_flow_fwd[ay, ax, 1]

        select = np.logical_and((((cx - ax) * (cx - ax)) + ((cy - ay) * (cy - ay))) < ((FLOW_VARIANCE_GRAD * ((u2 * u2) + (v2 * v2) + (u * u) + (v * v))) + FLOW_VARIANCE_CONST), motionEdge[ay, ax] < ((MOTION_EDGE_GRAD * ((u2 * u2) + (v2 * v2))) + MOTION_EDGE_CONST))
        goodTracks[ay[select], ax[select]] = 1

        for track in tracks:
            if track.stopped:
                continue

            if track.t == t:
                ax = float(track.mox)
                ay = float(track.moy)
            else:
                ax = track.mx[-1]
                ay = track.my[-1]

            iax = int(round(ax))
            iay = int(round(ay))

            if goodTracks[iay, iax] < 1:
                track.stopped = True
            else:
                bx = ax + pred_flow_fwd[iay, iax, 0]
                by = ay + pred_flow_fwd[iay, iax, 1]
                ibx = round(bx)
                iby = round(by)

                if ibx < 0 or iby < 0 or ibx >= w or iby >= h:
                    track.stopped = True
                else:
                    track.mx.append(bx)
                    track.my.append(by)

        sys.stdout.write("Optical Flow Tracking: {0:.2f}%".format(100.0 * (t + 1) / n) + '\r')
        sys.stdout.flush()

        input_a = input_b

    print("Optical Flow Tracking: {0:.2f}%".format(100.0))

    return tracks, flowVar