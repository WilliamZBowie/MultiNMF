#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 11:14:30 2023

@author: zach
"""

import numpy as np
import smfishHmrf.visualize as visualize
from scipy.stats import zscore
from smfishHmrf.DatasetMatrix import DatasetMatrixSingleField
from smfishHmrf.HMRFInstance import HMRFInstance
from utils import read_coords, read_cellnames, check_dir


def do_HMRF(dissim, image_names, coords, labels, output_dir, options):
    """
    Function which perform HMRF on MultiNMF sample dissimilarity matrix generated from MultiNMF components
    :param dissim: List of 2D numpy arrays containing MultiNMF components [Sample x K components]
    :type dissim: list
    :param image_names: File Name for coordinate file containing tab separated pixel coordinates [ x, y ]
    :type image_names: string
    :param coords: File Name for barcode file containing one barcode per line which corresponds to the same pixel coordinate
    :type coords: string
    :param labels: Final Kmeans label returned as array of int values corresponding to row order of dissimilarity matrix input
    :type labels: array
    :param output_dir: Output directory to save results to
    :type output_dir: string
    :param options: Dictionary containing settings for HMRF, see documentation for detail on specific variables
    :type options: dict
    """
    # Check results directory
    outdir = output_dir + "/HMRF"
    check_dir(outdir)

    # Read Barcodes and coordinates
    x_pixel, y_pixel = read_coords(coords)
    image_names = read_cellnames(image_names)
    loc = dict(zip(image_names, tuple(zip(x_pixel, y_pixel))))

    # Params
    num_clusters = options["k"]
    tolerance = options["tolerance"]
    nstart = options["n_start"]
    start = options["beta_range"][0]
    end = options["beta_range"][1]
    step = options["beta_range"][2]

    # Preprocess Dissimilarity Matrix
    num_spot = dissim.shape[0]
    avg_mat = np.empty((num_clusters, num_spot), dtype="float32")
    for i in range(num_clusters):
        m = np.where(labels == i)[0]
        avg_mat[i, :] = np.sum(dissim[m, :], axis=0)

    avg_mat = zscore(avg_mat, axis=1)
    Xcen = np.empty((len(image_names), 2), dtype="float32")
    for ic, c in enumerate(image_names):
        Xcen[ic, :] = [loc[c][0], loc[c][1]]

    # Process Spatial Information
    this_dset = DatasetMatrixSingleField(avg_mat, ["comp_%d" % d for d in range(0, num_clusters)], None, Xcen)
    this_dset.test_adjacency_list([0.1, 0.15, 0.2], metric="euclidean")
    this_dset.calc_neighbor_graph(0.10, metric="euclidean")
    this_dset.calc_independent_region()

    # Run HMRF
    print("Running HMRF...")

    this_hmrf = HMRFInstance("HMRF", outdir, this_dset, num_clusters, start, step, end, tolerance=tolerance)
    this_hmrf.init(nstart=nstart)
    this_hmrf.run()

    t_beta = 0
    for i in range(end):
        visualize.domain(this_hmrf, num_clusters, t_beta, dot_size=65, size_factor=10,
                         outfile="%s/k_%d/visualize.beta.%.1f.png" % (outdir, num_clusters, t_beta))
        t_beta += step

    print("Done")
