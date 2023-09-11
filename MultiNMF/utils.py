#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 10:37:13 2023

@author: zach
"""
# Necessary Imports

import os
import numpy as np
import pandas as pd
import scipy.stats
import scipy.stats
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans


# Function to check directory
def check_dir(output_dir):
    """
    Check if output directory exists, and if not create output directory
    :param output_dir: Directory Location to save function outputs
    :type output_dir: string
    """
    results_dir = os.path.join(output_dir)
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)


# Function to read in coordinates
def read_coords(n):
    """
    Reads in coordinate file using pandas dataframe
    :param n: File Name for coordinate file containing tab separated pixel coordinates [ x, y ]
    :type n: string
    :return: Two numpy arrays consisting of x and y pixel coordinates
    :rtype: arrays
    """
    df = pd.read_table(n, header=None)
    x = df[0].to_numpy()
    y = df[1].to_numpy()
    return x, y


# Function to read in barcodes
def read_cellnames(n):
    """
    Reads in barcode file using pandas dataframe
    :param n: File Name for barcode file containing one barcode per line which corresponds to the same pixel coordinate
    :type n: string
    :return: Numpy array which contains barcodes in the same order as pixel coordinate arrays
    :rtype: array
    """
    barcodes = pd.read_table(n, header=None)[0].to_numpy()
    return barcodes


# Function to scale data
def scale_components(a):
    """
    Scales MultiNMF component matrix by 95 percentile component value (Column Wise)
    :param a: 2D numpy array containing MultiNMF components [Sample x K components]
    :type a: array
    :return: Returns original array with each component column scaled by the 95% value
    :rtype: array
    """
    for i in range(a.shape[1]):
        p = np.percentile(a[:, i], 95)
        a[:, i] = a[:, i] / p


# Function to rank components
def rank_transform_matrix(mat, rbp_p=0.995, reverse=True):
    """
    Creates dissimilarity matrix from MultiNMF components by transforming components into the rank space
    :param mat: 2D numpy array containing MultiNMF components [Sample x K components]
    :type mat: array
    :param rbp_p: Weight of ranked_based clustering. Value recommended 0.995 or higher.
    :type rbp_p: float
    :param reverse: Whether to perform ranking again column wise
    :type reverse: bool
    :return: Dissimilarity Matrix calculated from rank matrices
    :rtype: array
    """
    # Initialize Rank Matrix
    dim1 = mat.shape[0]
    dim2 = mat.shape[1]
    rank_forward = np.empty([dim1, dim2])

    # Forward and Backward Ranking
    print("Start ranking forward...")
    for c1 in range(dim1):
        rd = scipy.stats.rankdata(mat[c1, :])
        if reverse:
            rd = dim2 - rd + 1
        rank_forward[c1, :] = rd
        if c1 % 1000 == 0:
            print("Done %d" % c1)

    print("Finished ranking forward...")
    rank_backward = np.empty([dim1, dim2])
    print("Start ranking backward...")
    for c1 in range(dim2):
        rd = scipy.stats.rankdata(mat[:, c1])
        if reverse:
            rd = dim1 - rd + 1
        rank_backward[:, c1] = rd
        if c1 % 1000 == 0:
            print("Done %d" % c1)

    # Create dissimilarity matrix
    print("Finished ranking backward...")
    print("Calculate mutual rank...")
    ma = np.sqrt(np.multiply(rank_forward, rank_backward))
    print("Calculate exponential transform...")
    mutual_rank_rbp = np.multiply(1 - rbp_p, np.power(rbp_p, np.subtract(ma, 1)))
    print("Finished exponential transform...")
    print("Calculate dissimilarity...")
    dissimilarity = np.subtract(1, np.divide(mutual_rank_rbp, 1 - rbp_p))
    print("Finished dissimilarity...")
    return dissimilarity


# Function to perform clustering
def do_Kmeans(factors, n_clusters, n_init, scale_bool, output_dir, rank_weight, seed=777):
    """
    Performs Kmeans on sample to sample dissimilarity matrix generated from MultiNMF components
    :param factors: List of 2D numpy arrays containing MultiNMF components [Sample x K components]
    :type factors: list
    :param n_clusters: Number of K clusters to generate
    :type n_clusters: int
    :param n_init: Number of random initializations for Kmeans
    :type n_init: int
    :param scale_bool: Boolean which determines if MultiNMF components are scaled
    :type scale_bool: bool
    :param output_dir: Output directory to save results to
    :type output_dir: string
    :param rank_weight: Weight of ranked_based clustering. Value recommended 0.995 or higher.
    :type rank_weight: float
    :param seed: Set seed for Kmeans
    :type seed: int
    :return: Final Kmeans label returned as array of int values corresponding to row order of factors input
    :rtype: array
    """

    # Check results directory
    check_dir(output_dir)

    # Perform Clustering
    num_rnds = len(factors)
    dissim = None
    for rnd in range(num_rnds):
        # Scale matrices
        if scale_bool:
            scale_components(factors[rnd])
        # Find Ranks
        euc = squareform(pdist(factors[rnd], metric="euclidean"))
        dissim_rnd = rank_transform_matrix(euc, reverse=False, rbp_p=rank_weight)
        if rnd == 0:
            dissim = dissim_rnd
        else:
            dissim = np.add(dissim, dissim_rnd)

    np.save(output_dir + '/Rank_Matrix.npy', dissim, allow_pickle=True)

    # Do Kmeans
    print("Performing Clustering")
    km_rnd = KMeans(n_clusters=n_clusters, init="random", n_init=n_init, random_state=seed)
    km_rnd.fit(dissim)
    final_labels = km_rnd.labels_
    print("Finished Clustering")
    np.savetxt(output_dir + "/cluster_labels.txt", final_labels.astype(int), fmt='%s')
    return final_labels
