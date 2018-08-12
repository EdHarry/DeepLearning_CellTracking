import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
import numpy as np
from scipy.spatial import KDTree
from scipy.sparse import lil_matrix, coo_matrix, csr_matrix, csc_matrix, diags, vstack
from scipy.sparse.csgraph import connected_components
from scipy.ndimage import label
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment
import networkx as nx
from PIL import Image
from skimage import color
from Utilities.graphTracks.generateCojoinedTracks import generateCojoinedTracks
from Utilities.graphTracks.findTrackConflicts import findTrackConflicts
from Utilities.sparse.sparseIndexing import delete_row_csr

# from OpticalFlow.OpticalFlowTracker import GenerateTracks as OpFlowTrack
# from OpticalFlow.OpticalFlowTracker import Track

# import matplotlib.animation as animation
# from pylab import *

# TRACK_IM_SCALE = 256

# NUM_SEEDS = 5
# NUM_TRACKS_PER_SEED = 3
# SEED_SCATTER_SCALE = 0.05
# NUM_TIMEPOINTS = 128
# DISPLACEMENT_SCALE = 0.1

NUM_NN_GRAPH = 4
MIN_PAIRING_DIS = 4

### Constants from "Motion Trajectory Segmentation via Minimum Cost Multicuts" M Keuper etal (edited)
ALPHA_ZERO_BAR = 6
ALPHA_ZERO = 2
ALPHA_ONE = -2
ALPHA_TWO = -4
ALPHA_THREE = -0.02
###

### Constants from "Higher-Order Minimum Cost Lifted Multicuts for Motion Segmentation" M Keuper (edited)
THETA_ZERO = -1
THETA_ONE = 0.8
###

KLALG_MAX_ITER = 100
KLARG_HISTORY_MEMORY = 10

class Track(object):
    def __init__(self, mox=0, moy=0, t=0):
        self.mox = mox
        self.moy = moy
        self.t = t
        self.mx = []
        self.my = []
        self.stopped = False

class Node(object):
        def __init__(self):
                self.edges = []
                self.data = -1

class Edge(object):
        def __init__(self):
                self.nodes = []
                self.cost = 1

        def order(self):
                return len(self.nodes)

class Graph(object):
        def __init__(self):
                self.nodes = {}

        def AddNode(self, node):
                if node.data not in self.nodes:
                        self.nodes[node.data] = node

        def GetNode(self, data):
                return self.nodes[data]

        def AddEdge(self, datas, cost=1):
                edge = Edge()
                edge.cost = cost

                for data in datas:
                        node = self.GetNode(data)
                        edge.nodes.append(node)
                        node.edges.append(edge)

        def RemoveOrphanNodes(self):
            self.nodes = {k: v for k, v in self.nodes.items() if len(v.edges) > 0}

def GetTrackStartEndTimes(tracks):
    n = len(tracks)
    trackSET = np.zeros((n, 2))
    trackSET[:, 0] = [track.t for track in tracks]
    trackSET[:, 1] = [track.t + len(track.mx) for track in tracks]

    return trackSET

def DoTracksIntersect(set_1, set_2):
    return not (set_1[1] <= set_2[0] or set_2[1] <= set_1[0]) 

def GraphTracks(tracks):
        nTracks = len(tracks)
        trackSET = GetTrackStartEndTimes(tracks)

        meanPositions = np.array([[np.mean([track.mox] + track.mx) for track in tracks], [np.mean([track.moy] + track.my) for track in tracks]]).T

        tree = KDTree(meanPositions)
        nn_index = tree.query(meanPositions, k=(NUM_NN_GRAPH + 1))[1][:, 1:]
        pairs = tree.query_pairs(MIN_PAIRING_DIS)

        graph = Graph()
        graphMat = lil_matrix((nTracks, nTracks), dtype='int')

        for iTrack in range(nTracks):
                node_i = Node()
                node_i.data = iTrack
                graph.AddNode(node_i)

                for jTrack in nn_index[iTrack, :]:
                        if graphMat[iTrack, jTrack] == 0 and graphMat[jTrack, iTrack] == 0 and DoTracksIntersect(trackSET[iTrack, :], trackSET[jTrack, :]):
                                node_j = Node()
                                node_j.data = jTrack
                                graph.AddNode(node_j)
                                graph.AddEdge([iTrack, jTrack])

                                graphMat[iTrack, jTrack] = 1
                                graphMat[jTrack, iTrack] = 1

        for pair in pairs:
                iTrack = pair[0]
                jTrack = pair[1]

                node_i = Node()
                node_i.data = iTrack
                graph.AddNode(node_i)

                if graphMat[iTrack, jTrack] == 0 and graphMat[jTrack, iTrack] == 0 and DoTracksIntersect(trackSET[iTrack, :], trackSET[jTrack, :]):
                        node_j = Node()
                        node_j.data = jTrack
                        graph.AddNode(node_j)
                        graph.AddEdge([iTrack, jTrack])

                        graphMat[iTrack, jTrack] = 1
                        graphMat[jTrack, iTrack] = 1

        graph.RemoveOrphanNodes()

        return graph

def AddNewThirdOrderCosts(thirdOrderCostsToCalc, node_i, node_j):
    new = []
    for edge in node_i.edges + node_j.edges:
        for node_k in edge.nodes:
            if node_k != node_i and node_k != node_j:
                new.append([node_i.data, node_j.data, node_k.data])

    if len(new) > 0:
        new = np.sort(np.array(new))
        thirdOrderCostsToCalc = np.unique(np.concatenate((thirdOrderCostsToCalc, new), axis=0), axis=0)

    return thirdOrderCostsToCalc

def CalculatePairwiseCosts(tracks, graph, movie, flowVar):
        thirdOrderCostsToCalc = np.zeros((0, 3), dtype=np.uint32)
        i = 0
        n = len(graph.nodes)
        n2 = n * n
        print("{} Nodes...".format(n))

        for _, node_i in graph.nodes.items():
            tmp = len(node_i.edges)
            for edge in node_i.edges:
                if edge.cost > 0:
                    for node_j in edge.nodes:
                        if node_i != node_j:

                            track_i = np.array([[tracks[node_i.data].mox] + tracks[node_i.data].mx, [tracks[node_i.data].moy] + tracks[node_i.data].my]).T
                            track_j = np.array([[tracks[node_j.data].mox] + tracks[node_j.data].mx, [tracks[node_j.data].moy] + tracks[node_j.data].my]).T
                            trackSET = GetTrackStartEndTimes([tracks[node_i.data]] + [tracks[node_j.data]])

                            startT = int(max(trackSET[:, 0]))
                            endT = int(min(trackSET[:, 1]))
                            time_i = tracks[node_i.data].t
                            time_j = tracks[node_j.data].t

                            track_i = track_i[startT - time_i: endT - time_i + 1, :]
                            track_j = track_j[startT - time_j: endT - time_j + 1, :]

                            d_track_i = track_i[1:, :] - track_i[:-1, :]
                            d_track_j = track_j[1:, :] - track_j[:-1, :]
                            d_track = np.linalg.norm(d_track_i - d_track_j, axis=1)
                            
                            track_flowVar_i = flowVar[track_i[1:, 1].astype('int'), track_i[1:, 0].astype('int'), range(startT + 1, endT + 1)]
                            track_flowVar_j = flowVar[track_j[1:, 1].astype('int'), track_j[1:, 0].astype('int'), range(startT + 1, endT + 1)]

                            motion_dis = 2 * d_track / (track_flowVar_i + track_flowVar_j)
                            motion_dis = motion_dis.max()

                            spatial_dis = np.linalg.norm(track_i - track_j, axis=1).mean()

                            movie_i = movie[track_i[:, 1].astype('int'), track_i[:, 0].astype('int'), range(startT, endT + 1)].reshape((-1, 1))
                            movie_j = movie[track_j[:, 1].astype('int'), track_j[:, 0].astype('int'), range(startT, endT + 1)].reshape((-1, 1))

                            im = Image.fromarray((movie_i * 255).astype('uint8'))
                            im_rgb = Image.new('RGB', im.size)
                            im_rgb.paste(im)
                            lab_i = color.rgb2lab(np.array(im_rgb)).squeeze()
                            im = Image.fromarray((movie_j * 255).astype('uint8'))
                            im_rgb.paste(im)
                            lab_j = color.rgb2lab(np.array(im_rgb)).squeeze()

                            colour_dis = np.linalg.norm(lab_i - lab_j, axis=1).mean()

                            cost = -max(ALPHA_ZERO_BAR + (ALPHA_ONE * motion_dis) + (ALPHA_TWO * spatial_dis) + (ALPHA_THREE * colour_dis), ALPHA_ZERO + (ALPHA_ONE * motion_dis))

                            if cost < 0:
                                edge.cost = cost
                            else:
                                edge.cost = 0
                                thirdOrderCostsToCalc = AddNewThirdOrderCosts(thirdOrderCostsToCalc, node_i, node_j)

                            sys.stdout.write("Calculating Second Order Costs...: {0:.2f}%".format(100.0 * (i + 1) / n2) + '\r')
                            i += 1
                            sys.stdout.flush()
            i += (n - tmp)

        print("Calculating Second Order Costs...: {0:.2f}%".format(100.0))

        return thirdOrderCostsToCalc

def CalculateTranslationModel(track_i_0, track_j_0, track_i_1, track_j_1):
    alpha = np.arccos(np.clip(np.dot(track_i_1 - track_j_1, track_i_0 - track_j_0) / (np.linalg.norm(track_i_1 - track_j_1) * np.linalg.norm(track_i_0 - track_j_0)), a_min=-1, a_max=1))
    R = np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])

    scale = np.linalg.norm(track_i_1 - track_j_1) / np.linalg.norm(track_i_0 - track_j_0)

    trans = 0.5 * (track_i_1 + track_j_1 - (scale * np.matmul(R, track_i_0 + track_j_0)))

    return R, scale, trans

def ApplyTranslationModel(R, scale, trans, track_i_0, track_j_0, track_k_0, track_k_1, flowVar):
    d = (np.matmul(R, track_k_0) * scale) + trans - track_k_1
    normFac = ((0.5 * ((np.linalg.norm(track_i_0 - track_j_0) / np.linalg.norm(track_i_0 - track_k_0)) + (np.linalg.norm(track_i_0 - track_j_0) / np.linalg.norm(track_j_0 - track_k_0)))) ** 0.25) / flowVar[int(track_k_0[1]), int(track_k_0[0])]

    return np.linalg.norm(d) / normFac 

def CalculateThirdOrderCosts(tracks, graph, flowVar, thirdOrderCostsToCalc):
    trackSET = GetTrackStartEndTimes(tracks)
    n = len(thirdOrderCostsToCalc)
    for i, triplet in enumerate(thirdOrderCostsToCalc):
        if DoTracksIntersect(trackSET[triplet[0], :], trackSET[triplet[1], :]) and DoTracksIntersect(trackSET[triplet[1], :], trackSET[triplet[2], :]) and DoTracksIntersect(trackSET[triplet[0], :], trackSET[triplet[2], :]):
            subtrackSET = GetTrackStartEndTimes([tracks[triplet[0]]] + [tracks[triplet[1]]] + [tracks[triplet[2]]])

            track_i = np.array([[tracks[triplet[0]].mox] + tracks[triplet[0]].mx, [tracks[triplet[0]].moy] + tracks[triplet[0]].my]).T
            track_j = np.array([[tracks[triplet[1]].mox] + tracks[triplet[1]].mx, [tracks[triplet[1]].moy] + tracks[triplet[1]].my]).T
            track_k = np.array([[tracks[triplet[2]].mox] + tracks[triplet[2]].mx, [tracks[triplet[2]].moy] + tracks[triplet[2]].my]).T

            startT = int(max(subtrackSET[:, 0]))
            endT = int(min(subtrackSET[:, 1]))
            time_i = tracks[triplet[0]].t
            time_j = tracks[triplet[1]].t
            time_k = tracks[triplet[2]].t

            track_i = track_i[startT - time_i: endT - time_i + 1, :]
            track_j = track_j[startT - time_j: endT - time_j + 1, :]
            track_k = track_k[startT - time_k: endT - time_k + 1, :]

            d_min = -np.inf
            d_max = -np.inf
            for index, t in enumerate(range(startT, endT + 1)):
                R, scale, trans = CalculateTranslationModel(track_i[index], track_j[index], track_i[-1], track_j[-1])
                d_k = ApplyTranslationModel(R, scale, trans, track_i[index], track_j[index], track_k[index], track_k[-1], flowVar[..., t])

                R, scale, trans = CalculateTranslationModel(track_k[index], track_j[index], track_k[-1], track_j[-1])
                d_i = ApplyTranslationModel(R, scale, trans, track_k[index], track_j[index], track_i[index], track_i[-1], flowVar[..., t])

                R, scale, trans = CalculateTranslationModel(track_i[index], track_k[index], track_i[-1], track_k[-1])
                d_j = ApplyTranslationModel(R, scale, trans, track_i[index], track_k[index], track_j[index], track_j[-1], flowVar[..., t])

                tmp_min = min(d_i, min(d_j, d_k))
                tmp_max = max(d_i, max(d_j, d_k))

                if tmp_min > d_min:
                    d_min = tmp_min
                if tmp_max > d_max:
                    d_max = tmp_max

            cost_min = THETA_ZERO + (THETA_ONE * d_min)
            cost_max = THETA_ZERO + (THETA_ONE * d_max)

            if cost_max > 0:
                cost = cost_min
            elif cost_min < 0:
                cost = cost_max
            else:
                cost = 0

            graph.AddEdge(datas=triplet, cost=cost)

        sys.stdout.write("Calculating Third Order Costs...: {0:.2f}%".format(100.0 * (i + 1) / n) + '\r')
        sys.stdout.flush()
    print("Calculating Third Order Costs...: {0:.2f}%".format(100.0))

def InitialGraphPartion(tracks, graph, rois, seg, frameDims):
    nTracks = len(tracks)
    nTimePoints = len(rois)
    nRois = rois[0].radii.shape[0]
    trackSET = GetTrackStartEndTimes(tracks).astype('int')

    allPoints = np.concatenate([np.array([[track.mox] + track.mx, [track.moy] + track.my]).T for track in tracks], axis=0)

    data = [i for i in range(1, allPoints.shape[0] + 1)]
    cols = np.concatenate([np.arange(trackSET[i, 0], trackSET[i, 1] + 1) for i in range(nTracks)]).tolist()
    rows = np.concatenate([i * np.ones(trackSET[i, 1] - trackSET[i, 0] + 1, dtype='int') for i in range(nTracks)]).tolist()

    m = coo_matrix((data, (rows, cols)), shape=(nTracks, nTimePoints)).tocsc()

    roiIds = -np.ones(nTracks).astype('int')
    roiIds_mat = lil_matrix((nTracks, nTimePoints), dtype='int')

    for t in range(nTimePoints):
        frameMap = -np.ones(frameDims).astype('int')
        for i in range(nRois):
            centre = rois[t].positions[i, :].astype('int')
            radius = rois[t].radii[i].astype('int') + 1

            if radius > 0:
                subFrame = frameMap[centre[1] - radius:centre[1] + radius, centre[0] - radius:centre[0] + radius]
                subSeg = seg[centre[1] - radius:centre[1] + radius, centre[0] - radius:centre[0] + radius, t]
                fill = i * np.ones(subFrame.shape)
                select = subSeg == 0
                fill[select] = -1
                select = subFrame != -1
                fill[select] = -2
                frameMap[centre[1] - radius:centre[1] + radius, centre[0] - radius:centre[0] + radius] = fill

        m_sub = m[:, t].toarray()[:, 0] - 1
        for i in range(nTracks):
            if m_sub[i] >= 0:
                point = allPoints[m_sub[i], :].astype('int')
                idx = frameMap[point[1], point[0]]
                if idx != -1:
                    roiIds_mat[i, t] = idx + 1
                    if roiIds[i] == -1 or roiIds[i] == idx:
                        roiIds[i] = idx
                    else:
                        roiIds[i] = -2

    graphPart = {}
    for _, node in graph.nodes.items():
        tmp = {edge: 0 for edge in node.edges if edge not in graphPart}
        graphPart = {**graphPart, **tmp}

    for i in range(roiIds.max() + 1):
        if i >= 0:
            idx = np.where(roiIds == i)[0]
            if len(idx) > 1:
                nodes = [graph.nodes[j] for j in idx if j in graph.nodes]
                
                tmpEdges = []
                for node in nodes:
                    tmp = [edge for edge in node.edges if edge not in tmpEdges]
                    tmpEdges += tmp

                for edge in tmpEdges:
                    goodEdge = np.array([1 if node in nodes else 0 for node in edge.nodes])
                    if np.all(goodEdge == 1):
                        graphPart[edge] = 1

    return graphPart, roiIds, roiIds_mat.tocsr()

def FindNNPartitions(graph, graphPart, roiIds, nTracks):
    nExtraNodes = roiIds.max() + 1
    m = lil_matrix((nTracks + nExtraNodes, nTracks + nExtraNodes))
    for edge, c in graphPart.items():
        if edge.order() == 2:
            if c == 1:
                for node in edge.nodes:
                    m[node.data, nTracks + roiIds[node.data]] = 1
            else:
                xpart = np.array([1 if roiIds[node.data] >= 0 else 0 for node in edge.nodes])
                xpart = np.any(xpart == 1)
                if not xpart:
                    for i in range(len(edge.nodes)):
                        for j in range(i):
                            m[edge.nodes[i].data, edge.nodes[j].data] = 1

    _, cc_labels = connected_components(m, directed=False, return_labels=True)
    cc_labels = cc_labels[:-nExtraNodes]

    return cc_labels

def FindBoundaryNodes(A, B):
    bdy_A = []
    for node in A:
        for edge in node.edges:
            if edge.order() == 2:
                for node_ in edge.nodes:
                    if node_ in B and node not in bdy_A:
                        bdy_A.append(node)
    bdy_B = []
    for node in B:
        for edge in node.edges:
            if edge.order() == 2:
                for node_ in edge.nodes:
                    if node_ in A and node not in bdy_B:
                        bdy_B.append(node)

    return bdy_A, bdy_B

def ComputeGains(A, B):
    gain_A = np.zeros(len(A))
    for i, node in enumerate(A):
        gain = 0
        for edge in node.edges:
            for node_ in edge.nodes:
                if node_ is not node and node_ in A:
                    gain += edge.cost
        for node_ in B:
            for edge in node_.edges:
                for node__ in edge.nodes:
                    if node__ is node:
                        gain -= edge.cost
        gain_A[i] = gain

    gain_B = np.zeros(len(B))
    for i, node in enumerate(B):
        gain = 0
        for edge in node.edges:
            for node_ in edge.nodes:
                if node_ is not node and node_ in B:
                    gain += edge.cost
        for node_ in A:
            for edge in node_.edges:
                for node__ in edge.nodes:
                    if node__ is node:
                        gain -= edge.cost
        gain_B[i] = gain

    return gain_A.tolist(), gain_B.tolist()

def ComputeJoiningGain(A, B):
    gain = 0
    
    for node in A:
        for edge in node.edges:
            for node_ in edge.nodes:
                if node_ in B:
                    gain -= edge.cost
    for node in B:
        for edge in node.edges:
            for node_ in edge.nodes:
                if node_ in A:
                    gain -= edge.cost

    return gain

def SplitPartition(cc_labels, index, graph):
    G = nx.Graph()
    A = [graph.nodes[k] for k in np.where(cc_labels == index)[0] if k in graph.nodes]

    if len(A) < 2:
        return

    A_full = A
    A = FindBoundaryNodes(A, [node for _, node in graph.nodes.items() if node not in A])[0]

    if len(A) < 2:
        return

    completedEdges = []
    minCost = 0

    for node in A:
        G.add_node(node)

    for node in A:
        for edge in node.edges:
            if edge not in completedEdges:
                completedEdges.append(edge)
                for node_ in edge.nodes:
                    if node_ is not node and node_ in A:
                        if G.has_edge(node, node_):
                            G[node][node_]['weight'] += edge.cost
                        else:
                            G.add_edge(node, node_, weight=edge.cost)
                        if G[node][node_]['weight'] < minCost:
                            minCost = G[node][node_]['weight'] 

    for node, nbrs in G.adjacency():
        for node_, eattr in nbrs.items():
            if node.data < node_.data:
                G[node][node_]['weight'] = 1 / (G[node][node_]['weight'] - minCost + 1)

    for i in range(len(A)):
        for j in range(i):
            if not G.has_edge(A[i], A[j]):
                G.add_edge(A[i], A[j], weight=(1 / (1 - minCost)))

    partitions = nx.stoer_wagner(G)[1]

    if len(partitions[0]) > len(partitions[1]):
        part_0 = list(partitions[0]) + [node for node in A_full if node not in partitions[0] and node not in partitions[1]]
        part_1 = partitions[1]
    else:
        part_1 = list(partitions[1]) + [node for node in A_full if node not in partitions[0] and node not in partitions[1]]
        part_0 = partitions[0]

    bdy_A, bdy_B = FindBoundaryNodes(part_0, part_1)
    joingGain = ComputeJoiningGain(bdy_A, bdy_B)

    if joingGain < 0:
        newId = cc_labels.max() + 1
        for node in partitions[0]:
            cc_labels[node.data] = newId

    BreakHere = 1

def UpdateBipartition(cc_labels, index_a, index_b, graph):
    moveMade = False
    nNodesMoved = 0

    A = [graph.nodes[k] for k in np.where(cc_labels == index_a)[0] if k in graph.nodes]
    B = [graph.nodes[k] for k in np.where(cc_labels == index_b)[0] if k in graph.nodes]

    bdy_A, bdy_B = FindBoundaryNodes(A, B)
    bdy = bdy_A + bdy_B
    
    if len(bdy) == 0:
        return False, 0

    gain_A, gain_B = ComputeGains(bdy_A, bdy_B)
    joingGain = ComputeJoiningGain(bdy_A, bdy_B)
    gain = gain_A + gain_B
    M = []
    S = [0]

    def SamePartition(nodes):
        in_A = np.array([1 if node in A else 0 for node in nodes])
        in_B = np.array([1 if node in B else 0 for node in nodes])

        return np.all(in_A == 1) or np.all(in_B == 1)

    # def NConnectedNodes(node, exclude):
    #     n = []
    #     for edge in node.edges:
    #         tmp = [node for node in edge.nodes if node is not exclude and node not in n]
    #         n += tmp

    #     return len(n)

    completedList = []

    for i in range(len(bdy)):
        v_index = np.argmax([g if i not in completedList else -np.inf for i, g in enumerate(gain)])
        v = bdy[v_index]
        M.append(v)
        M.append(0 if v in A else 1)
        for edge in v.edges:
            if SamePartition([node for node in edge.nodes if node is not v]):
                for node in [node for node in edge.nodes if node is not v]:
                    if node in bdy:
                        if SamePartition([node] + [v]) and edge.order() == 2: #NConnectedNodes(node, v) == 1:
                            gain[bdy.index(node)] -= (2 * edge.cost)
                        elif SamePartition([node] + [v]) and edge.order() > 2: #NConnectedNodes(node, v) > 1:
                            gain[bdy.index(node)] -= edge.cost
                        elif not SamePartition([node] + [v]) and edge.order() == 2: #NConnectedNodes(node, v) == 1:
                            gain[bdy.index(node)] += (2 * edge.cost)
                        elif not SamePartition([node] + [v]) and edge.order() > 2: #NConnectedNodes(node, v) > 1:
                            gain[bdy.index(node)] += edge.cost
            else:
                for node in [node for node in edge.nodes if node is not v]:
                    if node in bdy and SamePartition([node_ for node_ in edge.nodes if node_ is not v and node_ is not node]):
                        if SamePartition([node] + [v]):
                            gain[bdy.index(node)] += edge.cost
                        else:
                            gain[bdy.index(node)] -= edge.cost
        
        S.append(S[-1] + gain[v_index])
        # gain.remove(gain[v_index])
        # bdy.remove(bdy[v_index])
        completedList.append(v_index)
        if v in A:
            A.remove(v)
            B.append(v)
        else:
            B.remove(v)
            A.append(v)

    k = np.argmax(np.array(S[1:])) + 1

    if joingGain > S[k] and joingGain > 0:
        moveMade = True
        nNodesMoved = np.sum(cc_labels == index_b)
        cc_labels[cc_labels == index_b] = index_a

    elif S[k] > 0:
        moveMade = True
        nNodesMoved = k
        for i in range(0, 2 * k, 2):
            if M[i + 1] == 0:
                cc_labels[M[i].data] = index_b
            else:
                cc_labels[M[i].data] = index_a

    return moveMade, nNodesMoved

def KL(cc_labels, graph, roiIds):
    it = 0
    loop = True
    statusString = "iter {0:}/{1:}, {4:.2f}% :: {2:} nodes moved, {3:} graph partitions...".format(1, KLALG_MAX_ITER, 0, len(np.unique(cc_labels)), 0)
    sys.stdout.write(statusString + '\r')
    sys.stdout.flush()
    history = -np.ones((KLARG_HISTORY_MEMORY, len(cc_labels)))
    historyRepeat = False
    history_n = 0
    l = 0

    while loop and it < KLALG_MAX_ITER:
        nMax = cc_labels.max() + 1
        nPairs = (nMax * (nMax + 1)) / 2
        k = 0
        nNodesMoved_total = 0
        it += 1
        loop = False
        for i in range(nMax):
            for j in range(i):
                k += 1
                moveMade, nNodesMoved = UpdateBipartition(cc_labels, i, j, graph)
                
                if moveMade and (np.any(roiIds[cc_labels == i] >= 0) or np.any(roiIds[cc_labels == j] >= 0)):
                    loop = True

                nNodesMoved_total += nNodesMoved

                statusString = "iter {0:}/{1:}, {4:.2f}% :: {2:} nodes moved, {3:} graph partitions...".format(it, KLALG_MAX_ITER, nNodesMoved_total, len(np.unique(cc_labels)), 100 * k / nPairs)
                sys.stdout.write(statusString + '\r')
                sys.stdout.flush()

        for i in range(nMax): 
            SplitPartition(cc_labels, i, graph)
            k += 1
            statusString = "iter {0:}/{1:}, {4:.2f}% :: {2:} nodes moved, {3:} graph partitions...".format(it, KLALG_MAX_ITER, nNodesMoved_total, len(np.unique(cc_labels)), 100 * k / nPairs)
            sys.stdout.write(statusString + '\r')
            sys.stdout.flush()

        cc_labels_new = cc_labels.copy()
        for i, index in enumerate(np.unique(cc_labels)):
            cc_labels_new[cc_labels == index] = i
        cc_labels = cc_labels_new

        if loop and np.any(np.all(cc_labels == history, axis=1)):
            loop = False
            historyRepeat = True
            history_n = (l - np.where(np.all(cc_labels == history, axis=1))[0])
            history_n[history_n < 0] = KLARG_HISTORY_MEMORY + history_n[history_n < 0]
            history_n = history_n.min()
        else:
            history[l, :] = cc_labels
            l = (l + 1) % KLARG_HISTORY_MEMORY

    print(statusString)
    if historyRepeat:
        print("Solution oscillated between iterations {0:} and {1:}, stopping...".format(it - history_n, it))
    elif loop:
        print("No convergence after {} iterations...".format(KLALG_MAX_ITER))
    else:
        print("Solution converged after {} iterations...".format(it))

def UpdateGraphPartions(graphPart, cc_labels, roiIds):
    for edge in graphPart.keys():
        cc_ids = np.array([cc_labels[node.data] for node in edge.nodes])
        graphPart[edge] = 1 if np.all(cc_ids == cc_ids[0]) and roiIds[edge.nodes[0].data] >= 0 else 0

def UpdateROIs(graphPart, cc_labels, rois, roiIds, roiIds_mat):
    print("Updating ROIs...")

    nRois = roiIds.max() + 1
    nTimePoints = len(rois)
    nUnique = len(np.unique(cc_labels))
    roiMat = lil_matrix((nRois * nTimePoints * nUnique, nTimePoints))
    roiMat_indexes = np.zeros(nRois)
    roiMat_switches = np.zeros(nRois)

    count = 0
    for cc_index in np.unique(cc_labels):
        count += 1
        select = cc_labels == cc_index
        sub_roiIds_mat = roiIds_mat[select, :]
        
        for t in range(nTimePoints):
            tp = sub_roiIds_mat[:, t].toarray()[:, 0] - 1
            tp = tp[tp >= 0]
            uniqueIds = np.unique(tp)

            if len(uniqueIds) == 1:
                roiMat_switches[uniqueIds[0]] = 1
                roiMat[(uniqueIds[0] * nTimePoints) + roiMat_indexes[uniqueIds[0]], t] = 1

            elif len(uniqueIds) > 1:
                for idx in uniqueIds:
                    if roiMat_switches[idx] == 1:
                        roiMat_indexes[idx] += 1
                    roiMat_switches[idx] = 0
                    roiMat[(idx * nTimePoints) + roiMat_indexes[idx], t] = 1
                    roiMat_indexes[idx] += 1

        roiMat_indexes = (count * nRois * nTimePoints) * np.ones(nRois)

        statusString = "Dividing Mergers, {0:.2f}%...".format(count * 100 / nUnique)
        sys.stdout.write(statusString + '\r')
        sys.stdout.flush()

    print(statusString)

    n = roiMat.shape[0]
    roiMat = roiMat.tocsr()
    graphJumpN = 128
    for i in range(0, n, graphJumpN):
        nToJump = min(i + graphJumpN, n)
        tmp = diags((((np.arange(i, nToJump) % (nRois * nTimePoints)) / nTimePoints).astype('int') + 1), 0) * roiMat[i:nToJump, :]

        statusString = "Finding unique partitions, {0:.2f}%...".format(nToJump * 100 / n)
        sys.stdout.write(statusString + '\r')
        sys.stdout.flush()

        if tmp.nnz > 0:
            roiMat_new = tmp
            break

    for i in range(i + nToJump, n, graphJumpN):
        nToJump = min(i + graphJumpN, n)
        tmp = diags((((np.arange(i, nToJump) % (nRois * nTimePoints)) / nTimePoints).astype('int') + 1), 0) * roiMat[i:nToJump, :]

        statusString = "Finding unique partitions, {0:.2f}%...".format(nToJump * 100 / n)
        sys.stdout.write(statusString + '\r')
        sys.stdout.flush()

        if tmp.nnz > 0:
            roiMat_new = csr_matrix(np.unique(np.concatenate((roiMat_new.toarray(), tmp.toarray()), axis=0), axis=0))

    print(statusString)

    roiMat = roiMat_new
    roiMat = roiMat[np.array(roiMat.sum(axis=1))[:, 0] > 0, :]
    n = roiMat.shape[0]

    ### Gap Filling ###
    # labeled_array, num_features = label(roiMat[0, :].toarray()[0, :])
    # roiMat_new = csr_matrix(np.array([np.where(labeled_array == i, roiMat[0, :].toarray()[0, :], np.zeros(nTimePoints)) for i in range(1, num_features + 1)]))
    
    # statusString = "Dividing track fragments, {0:.2f}%...".format(100 / n)
    # sys.stdout.write(statusString + '\r')
    # sys.stdout.flush()
    
    # for i in range(1, roiMat.shape[0]):
    #     labeled_array, num_features = label(roiMat[i, :].toarray()[0, :])
    #     roiMat_new = vstack((roiMat_new, csr_matrix(np.array([np.where(labeled_array == j, roiMat[i, :].toarray()[0, :], np.zeros(nTimePoints)) for j in range(1, num_features + 1)]))))
        
    #     statusString = "Dividing track fragments, {0:.2f}%...".format((i + 1) * 100 / n)
    #     sys.stdout.write(statusString + '\r')
    #     sys.stdout.flush()

    # print(statusString)
    
    # roiMat = roiMat_new
    # n = roiMat.shape[0]
    roiMat = roiMat.tocsr()
    for i in range(0, n, graphJumpN):
        nToJump = min(i + graphJumpN, n)
        tmp = roiMat[i:nToJump, :]

        statusString = "Finding unique partitions, {0:.2f}%...".format(nToJump * 100 / n)
        sys.stdout.write(statusString + '\r')
        sys.stdout.flush()

        if tmp.nnz > 0:
            roiMat_new = tmp
            break

    for i in range(i + nToJump, n, graphJumpN):
        nToJump = min(i + graphJumpN, n)
        tmp = roiMat[i:nToJump, :]

        statusString = "Finding unique partitions, {0:.2f}%...".format(nToJump * 100 / n)
        sys.stdout.write(statusString + '\r')
        sys.stdout.flush()

        if tmp.nnz > 0:
            roiMat_new = csr_matrix(np.unique(np.concatenate((roiMat_new.toarray(), tmp.toarray()), axis=0), axis=0))

    print(statusString)

    roiMat = roiMat_new
    roiMat = roiMat[np.array(roiMat.sum(axis=1))[:, 0] > 0, :]
    n = roiMat.shape[0] 

    # ######## DEBUG ########
    # roiMat = roiMat[:128, :]
    # ######## DEBUG ########

    mainLoop = True
    roiMat_previous = roiMat.copy()

    print("---\n---")
    print("Main track construction loop...")
    print("---\n---")
    mainIter = 1
    while mainLoop:
        
        roiMat = roiMat.toarray()
        ret = generateCojoinedTracks(roiMat, "(Iteration {}) ".format(mainIter))
        roiMat = np.concatenate((roiMat, ret.extraTracks), axis=0)
        del ret
        roiMat = csc_matrix(roiMat)

        statusString = "(Iteration {1:}) Finding unique tracks, {0:.2f}%...".format(0, mainIter)
        sys.stdout.write(statusString + '\r')
        sys.stdout.flush()
        nMax = None

        loop = True
        while loop:
            roiMat = roiMat.tocsc()
            confs = findTrackConflicts(roiMat)
            roiMat = roiMat.tolil()

            if nMax is None:
                tmp = []
                for conf in confs:
                    tmp += conf
                nMax = len(tmp)

            if len(confs) == 0:
                loop = False
            else:
                toDel = np.zeros(len(confs))
                for k, conf in enumerate(confs):

                    conf = np.array(conf)

                    subMat = roiMat[conf, :].toarray().astype('int')
                    nnz = np.sum(subMat > 0, axis=1)
                    nnz_min = nnz.min()
                    select = nnz == nnz_min
                    n_select = select.sum()

                    if n_select == 1:
                        toDel[k] = conf[select]
                    
                    elif nnz_min == 1:
                        toDel[k] = conf[select][0]
                    
                    elif nnz_min == 2:
                        conf = conf[select]
                        subMat = subMat[select, :]
                        dis = np.zeros(n_select)

                        for i in range(n_select):
                            pos = np.array([rois[t].positions[j - 1, :] for t, j in enumerate(subMat[i, :]) if j > 0])
                            dis[i] = np.sqrt(np.sum(np.diff(pos, axis=0) ** 2, axis=1))

                        toDel[k] = conf[np.abs(dis - dis.mean()).argmax()]                

                    else:
                        conf = conf[select]
                        subMat = subMat[select, :]
                        var = np.zeros(n_select)

                        for i in range(n_select):
                            pos = np.array([rois[t].positions[j - 1, :] for t, j in enumerate(subMat[i, :]) if j > 0])
                            var[i] = np.var(np.sqrt(np.sum(np.diff(pos, axis=0) ** 2, axis=1)))

                        toDel[k] = conf[var.argmax()]
                    
                roiMat[toDel, :] = 0
                roiMat = roiMat[np.array(roiMat.sum(axis=1))[:, 0] > 0, :]

            tmp = []
            for conf in confs:
                tmp += conf
            tmp = len(tmp)
            statusString = "(Iteration {1:}) Finding unique tracks, {0:.2f}%...".format((nMax - tmp + 1) * 100 / (nMax + 1), mainIter)
            sys.stdout.write(statusString + '\r')
            sys.stdout.flush()

        print(statusString)

        ### Grow Tracks ###
        statusString = "(Iteration {1:}) Growing tracks, {0:.2f}%...".format(0, mainIter)
        sys.stdout.write(statusString + '\r')
        sys.stdout.flush()

        count = 0
        for i in range(2):
            if i == 1:
                roiMat = roiMat.tocsr()
                roiMat.indices = -roiMat.indices + roiMat.shape[1] - 1
                rois.reverse()

            for t in range(nTimePoints - 1):
                roiMat = roiMat.tocsc()
                select = np.array(np.sum(roiMat[:, t:(t + 2)] > 0, axis=1))[:, 0] == 1

                if np.any(select):
                    subMat = roiMat[select, t:(t + 2)].toarray().astype('int')
                    roiMat = roiMat.tolil()

                    select_fwd = subMat[:, 0] > 0
                    if np.any(select_fwd):
                        subMat_fwd = subMat[select_fwd, :]
                        nCurrent = select_fwd.sum()
                        
                        roiPos = rois[t + 1].positions
                        selectRoi = roiPos[:, 0] >= 0
                        roiPos = roiPos[selectRoi, :]
                        nRoi = roiPos.shape[0]

                        if nRoi > 0:
                            currentRois = rois[t].positions[subMat_fwd[:, 0] - 1, :]
                            disMat = distance.cdist(currentRois, roiPos)
                            current_assign, roi_assign = linear_sum_assignment(disMat)

                            roiMat[np.where(select)[0][np.where(select_fwd)[0]][current_assign], t + 1] = (np.where(selectRoi)[0][roi_assign] + 1).reshape((-1, 1))

                    select_bk = subMat[:, 1] > 0
                    if np.any(select_bk):
                        subMat_bk = subMat[select_bk, :]
                        nCurrent = select_bk.sum()
                        
                        roiPos = rois[t].positions
                        selectRoi = roiPos[:, 0] >= 0
                        roiPos = roiPos[selectRoi, :]
                        nRoi = roiPos.shape[0]

                        if nRoi > 0:
                            currentRois = rois[t + 1].positions[subMat_bk[:, 1] - 1, :]
                            disMat = distance.cdist(currentRois, roiPos)
                            current_assign, roi_assign = linear_sum_assignment(disMat)

                            roiMat[np.where(select)[0][np.where(select_bk)[0]][current_assign], t] = (np.where(selectRoi)[0][roi_assign] + 1).reshape((-1, 1))

                count += 1
                statusString = "(Iteration {1:}) Growing tracks, {0:.2f}%...".format(count * 100 / (2 * (nTimePoints - 1)), mainIter)
                sys.stdout.write(statusString + '\r')
                sys.stdout.flush()

        roiMat = roiMat.tocsr()
        roiMat.indices = -roiMat.indices + roiMat.shape[1] - 1
        rois.reverse()

        print(statusString)

        test = roiMat != roiMat_previous
        if type(test) is bool:
            mainLoop = test
        else:
            mainLoop = test.sum() > 0

        if mainLoop:
            roiMat_previous = roiMat.copy()
            mainIter += 1
            print("")

    return roiMat

if __name__ == "__main__":
    np.random.seed(865858544)
    testTrack_path = os.path.dirname(os.path.abspath(__file__)) + "/../TestData/test_tracks.pickle"
    testROI_path = os.path.dirname(os.path.abspath(__file__)) + "/../TestData/test_rois.pickle"
    testData_path = os.path.dirname(os.path.abspath(__file__)) + "/../TestData/test_data.h5"
    testSeg_path = os.path.dirname(os.path.abspath(__file__)) + "/../TestData/test_seg.h5"
    testKL_OneRun_path = os.path.dirname(os.path.abspath(__file__)) + "/../TestData/test_KL_OneRun.h5"
    testRoiUpdate_Onerun_path = os.path.dirname(os.path.abspath(__file__)) + "/../TestData/testRoiUpdate_Onerun.h5"
    import h5py
    import pickle
    with open(testTrack_path, 'rb') as f_in:
        tracks = pickle.load(f_in)
    with open(testROI_path, 'rb') as f_in:
        rois = pickle.load(f_in)
    with h5py.File(testData_path, 'r') as f_in:
        movie = f_in['movie'][:]
        flowVar = f_in['flowVar'][:]
    with h5py.File(testSeg_path, 'r') as f_in:
        seg = f_in['seg'][:]
   
    tracks = [track for track in tracks if len(track.mx) > 0]
    graph = GraphTracks(tracks)

    thirdOrderCostsToCalc = CalculatePairwiseCosts(tracks, graph, movie, flowVar)
    CalculateThirdOrderCosts(tracks, graph, flowVar, thirdOrderCostsToCalc)

    graphPart, roiIds, roiIds_mat = InitialGraphPartion(tracks, graph, rois, seg, movie.shape[:2])
    cc_labels = FindNNPartitions(graph, graphPart, roiIds, len(tracks))

    if os.path.exists(testKL_OneRun_path):
        with open(testKL_OneRun_path, 'rb') as f_in:
            cc_labels = pickle.load(f_in)
    else:
        KL(cc_labels, graph, roiIds)
        with open(testKL_OneRun_path, 'wb') as f_out:
            pickle.dump(cc_labels, f_out)
    
    UpdateGraphPartions(graphPart, cc_labels, roiIds)

    if os.path.exists(testRoiUpdate_Onerun_path):
        with open(testRoiUpdate_Onerun_path, 'rb') as f_in:
            roiIds_mat = pickle.load(f_in)
    else:
        roiIds_mat = UpdateROIs(graphPart, cc_labels, rois, roiIds, roiIds_mat)
        with open(testRoiUpdate_Onerun_path, 'wb') as f_out:
            pickle.dump(roiIds_mat, f_out)

    # BreakHere = 1
    # from VisuGraph import VisuGraph
    # VisuGraph(movie, tracks, graph, graphPart)

