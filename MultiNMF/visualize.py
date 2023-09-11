#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 12:16:41 2023

@author: zach
"""
# Necessary Imports

import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from utils import read_coords, read_cellnames, scale_components, check_dir


# Function to visualize components
def visualize_components(factors, image_names, coords, output_dir, scale_bool, size=20):
    """
    Function to visualize MultiNMF components, creates a grid of plots in which each plot represents one component from
    MultiNMF, and a plot for each round of MultiNMF completed is generated
    :param factors: List of 2D numpy arrays containing MultiNMF components [Sample x K components]
    :type factors: list
    :param image_names: File Name for coordinate file containing tab separated pixel coordinates [ x, y ]
    :type image_names: string
    :param coords: File Name for barcode file containing one barcode per line which corresponds to the same pixel coordinate
    :type coords: string
    :param output_dir: Output directory to save results to
    :type output_dir: string
    :param scale_bool: Boolean which determines if MultiNMF components are scaled
    :type scale_bool: bool
    :param size: Dot size for plot
    :type size: float
    """

    # Check results directory
    check_dir(output_dir)

    # Cell names and locations
    x_pixel, y_pixel = read_coords(coords)
    barcodes = read_cellnames(image_names)
    loc = dict(zip(barcodes, tuple(zip(x_pixel, y_pixel))))

    # Make folders to results per round
    for rnd in range(len(factors)):
        print("Round   ", rnd + 1)
        rd = os.path.join(output_dir, "Visualized_Components", 'Round{rnd}'.format(rnd=rnd + 1))
        if not os.path.isdir(rd):
            os.makedirs(rd)

        # Scale matrices
        if scale_bool == True:
            scale_components(factors[rnd])

        # Params
        size = size
        K = min(factors[rnd].shape)

        # Plot for probabilistic rep values
        n = int(np.ceil(np.sqrt(K)))
        f, axn = plt.subplots(n, n, sharex=True, sharey=True, figsize=(15, 13), dpi=300)
        axn[0,0].invert_yaxis()
        for i in range(K):
            all_x = np.array([loc[c][0] for c in barcodes])
            all_y = np.array([loc[c][1] for c in barcodes])
            kth_plot = axn.flat[i].scatter(all_x, all_y, c=factors[rnd][:, i], s=size)
            axn.flat[i].xaxis.set_ticklabels([])
            axn.flat[i].yaxis.set_ticklabels([])
            axn.flat[i].set_xticks([])
            axn.flat[i].set_yticks([])
        plt.suptitle("MultiNMF Components Round {}".format(rnd))
        plt.savefig(rd + "/MultiNMF_Components-Round{}.png".format(rnd + 1), bbox_inches="tight")


# Function to visualize components
def visualize_clusters(labels, image_names, coords, output_dir, size=20, color_palette="husl"):
    """
    Function to visualize MultiNMF labels, creates a plot which visualizes clustering in a spatial context
    :param labels: Array containing for clustering labels in order matching barcode and coordinate files
    :type labels: array
    :param image_names: File Name for coordinate file containing tab separated pixel coordinates [ x, y ]
    :type image_names: string
    :param coords: File Name for barcode file containing one barcode per line which corresponds to the same pixel coordinate
    :type coords: string
    :param output_dir: Output directory to save results to
    :type output_dir: string
    :param color_palette: String to sepecify seaborn color see https://seaborn.pydata.org/tutorial/color_palettes.html
    :type color_palette: string
    :param size: Dot size for plot
    :type size: float
    """

    # Check results directory
    check_dir(output_dir)

    # Cell names and locations
    x_pixel, y_pixel = read_coords(coords)
    barcodes = read_cellnames(image_names)
    loc = dict(zip(barcodes, tuple(zip(x_pixel, y_pixel))))

    # Plot Clusters
    num_celltype = len(np.unique(labels))
    size = size * num_celltype * 10
    plot_color = sns.color_palette(color_palette, num_celltype)
    plot_color = np.array(plot_color)
    clusters = np.unique(labels)
    all_x = np.array([loc[c][0] for c in barcodes])
    all_y = np.array([loc[c][1] for c in barcodes])
    colors = {clusters[y]: plot_color[y] for y in range(len(clusters))}
    f, ax = plt.subplots(figsize=(15, 13), dpi=300)
    for g in np.unique(labels):
        ix = np.where(labels == g)
        ax.scatter(alpha=1, x=all_x[ix], y=all_y[ix], color=colors[g], s=size, label=g)
        ax.set_aspect('equal', 'box')
    ax.invert_yaxis()
    # Hide X and Y axes label marks
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.yaxis.set_tick_params(labelleft=False)
    # Hide X and Y axes tick marks
    ax.set_xticks([])
    ax.set_yticks([])
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 6})
    plt.title("MultiNMF Clusters")
    plt.savefig(output_dir + "/MultiNMF_Clusters.png", bbox_inches="tight")


# test = np.load("V_centroid.npy")
# print(test.shape)
# coords = "coordinates_CID4290.txt"
# barcodes = "barcodes_CID4290.txt"
# output_dir = "./Results"
# labels = np.loadtxt("./Results/cluster_labels.txt")
# scale_bool = True
#
# #visualize_components(test, barcodes, coords, output_dir, scale_bool=False)
# visualize_clusters(labels, barcodes, coords, output_dir, scale_bool)

# test = np.load("V_centroid.npy_seeded.npy")
# print(test.shape)
# coords = "coordinates_CID4290.txt"
# barcodes = "barcodes_CID4290.txt"
# output_dir = "./Results_seeded"
# labels = np.loadtxt("./Results_seeded/cluster_labels.txt")
# scale_bool = True
#
# visualize_components(test, barcodes, coords, output_dir, scale_bool=False)
# visualize_clusters(labels, barcodes, coords, output_dir, scale_bool)
