import sys, os, numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
from MovieGenerator.MovieGenerator import GenerateMovie
import matplotlib.animation as animation
from pylab import *

from OpticalFlowTracker import GenerateTracks

np.random.seed(400582833)
dpi = 100

fig = plt.figure()
axes = fig.add_subplot(111)
axes.set_aspect('equal')
axes.get_xaxis().set_visible(False)
axes.get_yaxis().set_visible(False)

movie, _ = GenerateMovie(GenImSize=(128, 128), numFrames=64, numCells=16, InitVelSD=2, VelScaleSD=1, AngleSDRad=np.radians(20),
                radius_mean=7, radius_sd=1, radius_min=6, radius_max=10, WriteMovieFile=False)

tracks = GenerateTracks(movie)

h, w, n = (int(s) for s in movie.shape)

name = 'demo.mp4'
fps = 5
im = axes.imshow(movie[:, :, 0],cmap='gray',interpolation='nearest')
im.set_clim([0,1])
fig.set_size_inches([5, 5])

tight_layout()

def draw_flow_lines(flow, step=16):
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, fx, fy]).T.reshape(-1, 2, 2)
    return lines

def draw_tracks(tracks, t):
    lines = []

    if t > 0:
        x1 = []
        y1 = []
        x2 = []
        y2 = []

        for track in tracks:
            trackLen = track.t + len(track.mx)
            if trackLen >= t:
                trackT = t - track.t
                if trackT <= 0:
                    continue
                elif trackT == 1:
                    x1.append(float(track.mox))
                    y1.append(float(track.moy))
                else:
                    x1.append(track.mx[trackT - 2])
                    y1.append(track.my[trackT - 2])

                x2.append(track.mx[trackT - 1] - x1[-1])
                y2.append(track.my[trackT - 1] - y1[-1])

        if len(x1) > 0:
            lines = np.vstack([x1, y1, x2, y2]).T.reshape(-1, 2, 2)

    return lines


prevDrawList = []
boxProps_frame = dict(boxstyle='square', facecolor='grey', alpha=1)

def compute_color(u, v):
    """
    compute optical flow color map
    :param u: optical flow horizontal map
    :param v: optical flow vertical map
    :return: optical flow in color code
    """
    [h, w] = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0

    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)

    rad = np.sqrt(u**2+v**2)

    a = np.arctan2(-v, -u) / np.pi

    fk = (a+1) / 2 * (ncols - 1) + 1

    k0 = np.floor(fk).astype(int)

    k1 = k0 + 1
    k1[k1 == ncols+1] = 1
    f = fk - k0

    for i in range(0, np.size(colorwheel,1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0-1] / 255
        col1 = tmp[k1-1] / 255
        col = (1-f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1-rad[idx]*(1-col[idx])
        notidx = np.logical_not(idx)

        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col*(1-nanIdx)))

    return img


def make_color_wheel():
    """
    Generate color wheel according Middlebury color code
    :return: Color wheel
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3])

    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255*np.arange(0, RY) / RY))
    col += RY

    # YG
    colorwheel[col:col+YG, 0] = 255 - np.transpose(np.floor(255*np.arange(0, YG) / YG))
    colorwheel[col:col+YG, 1] = 255
    col += YG

    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.transpose(np.floor(255*np.arange(0, GC) / GC))
    col += GC

    # CB
    colorwheel[col:col+CB, 1] = 255 - np.transpose(np.floor(255*np.arange(0, CB) / CB))
    colorwheel[col:col+CB, 2] = 255
    col += CB

    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.transpose(np.floor(255*np.arange(0, BM) / BM))
    col += + BM

    # MR
    colorwheel[col:col+MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col+MR, 0] = 255

    return colorwheel

def flow_to_image(flow):
    """
    Convert flow into middlebury color code image
    :param flow: optical flow map
    :return: optical flow image in middlebury color
    """
    u = flow[:, :, 0]
    v = flow[:, :, 1]

    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.

    idxUnknow = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH)
    u[idxUnknow] = 0
    v[idxUnknow] = 0

    maxu = max(maxu, np.max(u))
    minu = min(minu, np.min(u))

    maxv = max(maxv, np.max(v))
    minv = min(minv, np.min(v))

    rad = np.sqrt(u ** 2 + v ** 2)
    maxrad = max(-1, np.max(rad))

    #print("max flow: %.4f\nflow range:\nu = %.3f .. %.3f\nv = %.3f .. %.3f" % (maxrad, minu,maxu, minv, maxv))

    u = u/(maxrad + np.finfo(float).eps)
    v = v/(maxrad + np.finfo(float).eps)

    img = compute_color(u, v)

    idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0

    return np.uint8(img)

def update_img(t):
    global prevDrawList

    frameLabel = axes.annotate(t, (10, 10), color='yellow', size='large', bbox=boxProps_frame)

    im.set_data(movie[:, :, t])

    track_lines = draw_tracks(tracks, t + 1)
    ax_lines = [axes.arrow(x1, y1, x2, y2, head_width=2, head_length=4, fc='green', ec='green') for (x1, y1), (x2, y2) in track_lines]

    for patch in prevDrawList:
        patch.remove()

    prevDrawList = ax_lines + [frameLabel]
    sys.stdout.write("Generating Movie File: {0:.2f}%".format(100.0 * (t + 1) / n) + '\r')
    sys.stdout.flush()

    return [im] + ax_lines + [frameLabel]



ani = animation.FuncAnimation(fig, update_img, frames=range(n-1), blit=True)
writer = animation.writers['ffmpeg'](fps=fps)
ani.save(name,writer=writer,dpi=dpi)
print("Generating Movie File: {0:.2f}%".format(100.0))