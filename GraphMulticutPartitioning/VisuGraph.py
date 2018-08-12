import matplotlib.animation as animation
from pylab import *
import numpy as np

prevDrawList = []
dpi = 100
fps = 5
name = 'Graphed_Tracks.mp4'
figSize = [5, 5]

def VisuGraph(movie, tracks, graph, graphPart):
	
	fig = plt.figure()
	axes = fig.add_subplot(111)
	axes.set_aspect('equal')
	axes.get_xaxis().set_visible(False)
	axes.get_yaxis().set_visible(False)

	h, w, n = (int(s) for s in movie.shape)

	im = axes.imshow(movie[:, :, 0],cmap='gray',interpolation='nearest')
	im.set_clim([0,1])
	fig.set_size_inches(figSize)

	tight_layout()

	def draw_tracks(graph, tracks, t):
	    lines = []
	    graph_lines = []
	    graph_lines_zeros = []
	    graphEnds = []
	    graphIndexs = []

	    if t > 0:
	        x1 = []
	        y1 = []
	        x2 = []
	        y2 = []

	        graph_x1_tmp = []
	        graph_y1_tmp = []

	        graph_x1 = []
	        graph_y1 = []
	        graph_x2 = []
	        graph_y2 = []

	        graph_x1_zero = []
	        graph_y1_zero = []
	        graph_x2_zero = []
	        graph_y2_zero = []

	        for iTrack, track in enumerate(tracks):
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
	                graphIndexs.append(iTrack)

	                if iTrack in graph.nodes:
	                	graph_x1_tmp.append(x1[-1])
	                	graph_y1_tmp.append(y1[-1])

	                	tmp = []
	                	for edge in graph.nodes[iTrack].edges:
	                		if graphPart[edge] == 1:
		                		for node in edge.nodes:
		                			if node != graph.nodes[iTrack]:
		                				tmp.append(node.data * (-1 if edge.cost >= 0 else 1))
	                	graphEnds.append(tmp)
	                	
	        for i, ends in enumerate(graphEnds):
	        	for j in ends:
	        		sign = np.sign(j)
	        		j = np.abs(j)
	        		if j in graphIndexs:
		        		if sign >= 0:
			        		graph_x1.append(graph_x1_tmp[i])
			        		graph_y1.append(graph_y1_tmp[i])
			        		graph_x2.append(x1[graphIndexs.index(j)] - graph_x1[-1])
			        		graph_y2.append(y1[graphIndexs.index(j)] - graph_y1[-1])
			        	else:
			        		graph_x1_zero.append(graph_x1_tmp[i])
			        		graph_y1_zero.append(graph_y1_tmp[i])
			        		graph_x2_zero.append(x1[graphIndexs.index(j)] - graph_x1_zero[-1])
			        		graph_y2_zero.append(y1[graphIndexs.index(j)] - graph_y1_zero[-1])		        		

	        if len(x1) > 0:
	            lines = np.vstack([x1, y1, x2, y2]).T.reshape(-1, 2, 2)

	        if len(graph_x1) > 0:
	        	graph_lines = np.vstack([graph_x1, graph_y1, graph_x2, graph_y2]).T.reshape(-1, 2, 2)

	        if len(graph_x1_zero) > 0:
	        	graph_lines_zeros = np.vstack([graph_x1_zero, graph_y1_zero, graph_x2_zero, graph_y2_zero]).T.reshape(-1, 2, 2)

	    return lines, graph_lines, graph_lines_zeros

	boxProps_frame = dict(boxstyle='square', facecolor='grey', alpha=1)

	def update_img(t):
	    global prevDrawList

	    frameLabel = axes.annotate(t, (10, 10), color='yellow', size='large', bbox=boxProps_frame)

	    im.set_data(movie[:, :, t])

	    track_lines, graph_lines, graph_lines_zeros = draw_tracks(graph, tracks, t + 1)
	    ax_lines = [axes.arrow(x1, y1, x2, y2, head_width=2, head_length=4, fc='green', ec='green') for (x1, y1), (x2, y2) in track_lines]
	    ax_lines += [axes.arrow(x1, y1, x2, y2, head_width=0.2, head_length=0.4, fc='red', ec='red') for (x1, y1), (x2, y2) in graph_lines_zeros]
	    ax_lines += [axes.arrow(x1, y1, x2, y2, head_width=0.2, head_length=0.4, fc='blue', ec='blue') for (x1, y1), (x2, y2) in graph_lines]

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
