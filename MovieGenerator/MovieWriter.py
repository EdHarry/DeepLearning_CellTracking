import matplotlib.animation as animation, matplotlib.patches as patches
from matplotlib.pyplot import cm
from pylab import *

dpi = 100

prevDrawList = []

def WriteMovie(data, roiData=None, name='demo.mp4', fps=5):
	print("---\n---")

	n = int(data.shape[2])

	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.set_aspect('equal')
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)

	im = ax.imshow(data[:, :, 0],cmap='gray',interpolation='nearest')
	im.set_clim([0,1])
	fig.set_size_inches([5,5])

	tight_layout()
	boxProps = dict(boxstyle='round', facecolor='grey', alpha=0.9)
	boxProps_frame = dict(boxstyle='square', facecolor='grey', alpha=1)

	def PatchesFromROIs(rois):
		n = rois.radii.shape[0]
		color = iter(cm.rainbow(np.linspace(0,1, n)))
		patch_list = []
		label_list = []
		for i in range(n):
			p = rois.positions[i, :]
			r = rois.radii[i]
			c = next(color)
			if r >= 0.0:
				patch = patches.Rectangle((p[0] - r, p[1] - r), 2*r, 2*r, ec=c, fill=False, aa=True)
				patch_list.append(patch)

				textLabel = ax.annotate(i, (p[0], p[1]), color=c, size='large', bbox=boxProps)
				label_list.append(textLabel)
		
		return patch_list, label_list

	def update_img(t):
	    global prevDrawList

	    frameLabel = ax.annotate(t, (10, 10), color='yellow', size='large', bbox=boxProps_frame)

	    tmp = data[:, :, t]
	    im.set_data(tmp)
	    
	    if roiData is None:
	    	draw_list = [[],[]]
	    else:
	    	draw_list = PatchesFromROIs(roiData[t])

	    for patch in prevDrawList:
	    	patch.remove()

	    for patch in draw_list[0]:
	    	ax.add_patch(patch)

	    prevDrawList = draw_list[0] + draw_list[1] + [frameLabel]

	    sys.stdout.write("Generating Movie File: {0:.2f}%".format(100.0 * (t + 1) / n) + '\r')
	    sys.stdout.flush()

	    return [im] + draw_list[0] + draw_list[1] + [frameLabel]

	#legend(loc=0)
	ani = animation.FuncAnimation(fig, update_img, frames=range(n), blit=True)
	writer = animation.writers['ffmpeg'](fps=fps)

	ani.save(name,writer=writer,dpi=dpi)

	print("Generating Movie File: {0:.2f}%".format(100.0))
	print("---\n---")