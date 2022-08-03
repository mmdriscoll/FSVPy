import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import skimage.draw
import networkx as nx
import itertools
import pylab
from scipy.spatial import KDTree
from skimage.morphology import medial_axis
from skimage.filters import difference_of_gaussians, window, gaussian
import matplotlib as mpl
import pandas as pd


#########################################################
'''
centerlines identifies the medial axis of each contour found in an image

helper functions: polycenterline: uses polygon defined by contour coords to find medial axis via skeltoniazation
'''
#########################################################

'''

Uses a contour to find the centerline (medial axis)

Input: contours, list of contours, each item in the list contains x & y positions of each contour
	   image: numpy array, image corresponding to contours list

Output: list of x,y points in the medial axis

'''


def centerlines(contours, image):
	ma = []
	for contour in contours:
		try:
			ctrline = poly_centerline(contour, image)
			srt_ctrline = sort_points(ctrline)
			ma.append(srt_ctrline)
		except:
			ma.append([])

	return ma


'''

poly_centerline: returns the medial axis of the polygon defined by points found in the array image_result.  
					This is a wrapper on skimage's medial axis function, which processes a thresholded image,
					and thus the returned coordinates are ints representing the skeleton (no-subpixel resolution)

inputs: points: numpy array of x,y coordinates defining a polygon, (x_points are in column 1
				and the y points are in column 0)

		image: numpy array that is the image the polygon was found in, needed because a thresholded image
				is needed in this workflow



outputs: coords: numpy array (x=coords[:,0], y = coords[:,1]) of the coordinates of the medial axis
'''
def poly_centerline(points, image):

	#now use the polygon points to define a thresholded image
	thresh = threshold_from_contour(points, image)

	#now find the medial axis (this is a mask on the image)
	cl = medial_axis(thresh)

	#now extract the coordinates from the mask
	coords = np.transpose((cl == True).nonzero())   

	return coords


'''
wrapper around skimage polygon function to return thresholded image as defined by contour points
'''


def threshold_from_contour(points, image):
	rr, cc = skimage.draw.polygon(points[:,0],points[:,1])   
	mask = np.zeros(image.shape) 
	mask[rr,cc]=1   

	return mask




'''
sort_points: creates a graph from points, and then uses the shortest path in the graph to sort the points

inputs: points: numpy array of x,y coordinates defining the medial axis of a polygon 

outputs: coords: numpy array (x=coords[:,0], y = coords[:,1]) of the coordinates of the medial axis but now ordered
'''
def sort_points(points):


	G = nx.Graph()  # A graph to hold the nearest neighbours
	G.add_nodes_from(range(len(points)))  

	tree = KDTree(points, leafsize=2)  # Create a distance tree
	#nn = tree.query(points, k = 3) #identify 2 nearest neighbors (first one is yourself)
	nn = tree.query_pairs(1.5)

	#define edges from nn list
	for item in nn:
		for entry in item[1::]:
			G.add_edge(item[0], entry)

	#identify degree of each node
	degrees = np.array([val for (node, val) in G.degree()])

	free_ends = np.where(degrees < 2)[0]  #dentify nodes with degree < 2, these are candidate source/target nodes
	shortest_paths = []
	source_target_combos = list(itertools.combinations(free_ends,2))  
	for pair in source_target_combos:
		shortest_paths.append(nx.shortest_path(G, source=pair[0], target=pair[1]))

	#find the longest shortest path in case there were stubby ends
	sort_order = max(shortest_paths, key = len)

	sorted_points = points[sort_order]

	return sorted_points






