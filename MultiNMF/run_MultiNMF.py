#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 10:54:34 2023

@author: zach
"""

import os, argparse
import numpy as np
from MultiNMF import MultiNMF
from visualize import visualize_clusters, visualize_components
from extract_image_features import process_morphology
from utils import do_Kmeans
from HMRF import do_HMRF


# Custom Types
def weight(string):
    value = float(string)
    if value < 0 or value > 1:
        raise argparse.ArgumentTypeError('Value has to be between 0 and 1')
    return value


# Input Parsing
def parse_arguments(prog='MultiNMF'):
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument(
        '--view_files', type=str, default='', nargs='+', required=True,
        help='Tab Seperated matrix text files with dimensions N cells by G genes, which contain each data modality to be inputted.'
    )
    parser.add_argument(
        '--view_types', type=str, default='', nargs='+', choices=("expr", "morphology", "atac"), required=True,
        help='String Arguments which specify data type of view files, inputted in same order as view matricies. Must be "expr" "morphology" or "atac"'
    )
    parser.add_argument(
        '--cell_coordinates', type=str, default='', required=True,
        help='File which contains cell coordinates coor x and coor y as two tab seperated columns in the same order as the Expression Matrix.'
    )
    parser.add_argument(
        '--cell_names', type=str, default='', required=True,
        help='File which contains cell names / barcodes in the same order as the Expression Matrix, each barcode on a new line.'
    )
    parser.add_argument(
        '--image_file', type=str, default='',
        help='image file correctly scaled to cell coordinates such that cell coordinates are in pixels and correspond to physical location in morphology image.'
    )
    parser.add_argument(
        '--scale_factor', type=float,
        help='Cell or Spot diameter measured in pixels'
    )
    parser.add_argument(
        '--output_dir', type=str, default='./Results',
        help='Valid output directory to store MultiNMF results'
    )
    # hyperparameters
    parser.add_argument(
        '--K', type=int, default=20,
        help='Number labels of components to factorize views into'
    )
    parser.add_argument(
        '--k', type=int, default=20,
        help='Number of clusters to cluster components into'
    )
    parser.add_argument(
        '--k_init', type=int, default=10000,
        help='Number of initializations for Kmeans'
    )
    parser.add_argument(
        '--view_weights', type=weight, default=0.01, nargs='+', required=True,
        help='Weights to tune influence of individual views to overall consensus of clustering structure, recommended 0.01 per view.'
    )
    parser.add_argument(
        '--size', type=int, default=20,
        help='Spot size for visualization plots'
    )
    parser.add_argument(
        '--color', type=str, default="husl",
        help='String to sepecify seaborn color see https://seaborn.pydata.org/tutorial/color_palettes.html'
    )
    parser.add_argument(
        '--HMRF', type=bool, default=False,
        help='Boolean specifying whether to use optional HMRF smoothing .'
    )
    parser.add_argument(
        '--normalize', type=bool, default=False,
        help='Boolean specifying whether normalize MultiNMF by 95th percentile of component values .'
    )
    parser.add_argument(
        '--rank_weight', type=weight, default=0.995,
        help='Weight of ranked_based clustering. Value recommended 0.995 or higher'
    )
    parser.add_argument(
        '--beta_range', type=float, default=[0, 10, 0.5], nargs=3, metavar=('start', 'end', 'step'),
        help='Specify range of number of rounds and increment values to be used for HMRF.'
    )
    parser.add_argument(
        '--HMRF_tolerance', type=float, default=1e-20,
        help='Tolerance threshold for HMRF'
    )
    parser.add_argument(
        '--HMRF_init', type=int, default=10000,
        help='Number of initializations for HMRF'
    )
    # =============================================================================
    #     parser.add_argumelabelsnt(
    #         '--MultiNMF_kwargs', type=eval, default=dict(),
    #         help='A dictionary specifying arguments for MultiNMF model'
    #     )
    # =============================================================================
    return parser.parse_args()


# Run HMRF
if __name__ == '__main__':
    # Parse Arguments
    args = parse_arguments()
    print(args)

    # Results Dir
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Set MultiNMF Args
    options = {}
    options["maxIter"] = 200
    options["error"] = 1e-6
    options["nRepeat"] = 30
    options["minIter"] = 50
    options["meanFitRatio"] = 0.1
    options["rounds"] = 30
    # options.kmeans means whether to run kmeans on v^* or not
    # options alpha is an array of weights for different views
    options["alpha"] = [0.01, 0.01]
    options["kmeans"] = 1

    # Process Inputs for Multi NMF
    data = []
    if "expr" in args.view_types:
        expr = np.loadtxt(args.view_files[args.view_types.index("expr")])
        expr = np.transpose(expr)
        data.append(expr)
    elif "atac" in args.view_types:
        atac = np.loadtxt(args.view_files[args.view_types.index("atac")])
        atac = np.transpose(atac)
        data.append((atac))

    morpho = process_morphology(args.image_file,
                                args.cell_coordinates,
                                args.cell_names,
                                args.scale_factor,
                                args.output_dir)
    morpho = np.transpose(morpho)
    morpho = -morpho
    Min = abs(np.min(morpho))
    morpho = morpho + Min
    data.append(morpho)

    # Normalize
    for i in range(len(data)):
        data[i] = data[i] / sum(sum(data[i]))

    # Do MultiNMF
    U_final = np.empty((3, 0)).tolist()
    V_final = np.empty((3, 0)).tolist()
    V_centroid = np.empty((3, 0)).tolist()

    for i in range(3):
        U_final[i], V_final[i], V_centroid[i] = MultiNMF(data, args.K, options)

    np.save('MultiNMF_outs.npy', np.array([U_final, V_final, V_centroid], dtype=object), allow_pickle=True)
    np.save('V_centroid.npy', V_centroid, allow_pickle=True)

    # Visualize components
    visualize_components(V_centroid, args.cell_names, args.cell_coordinates, args.output_dir, args.normalize)

    # Do Kmeans
    labels = do_Kmeans(V_centroid, args.k, args.k_init, args.normalize, args.output_dir, rank_weight=args.rank_weight,
                       seed=777)
    # Visualize clusters
    visualize_clusters(labels, args.cell_names, args.cell_coordinates, args.output_dir, args.normalize)

    # Optional HMRF
    if args.HMRF:
        # make HMRF dictionary
        HMRF_options = {}
        HMRF_options["k"] = args.k
        HMRF_options["beta_range"] = args.beta_range
        HMRF_options["tolerance"] = args.HMRF_tolerance
        HMRF_options["init"] = args.HMRF_init
        HMRF_options["n_start"] = 1000

    # run HMRF
    #dissim = np.load(os.path.join(args.output_dir, "Rank_Matrix.npy"))
    #labels = np.loadtxt(os.path.join(args.output_dir, "cluster_labels.npy"))
    #do_HMRF(dissim, args.cell_names, args.cell_coordinates, labels, args.output_dir, HMRF_options)
