#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 16:32:09 2023

@author: zach
"""

import os,csv,re
import numpy as np
import pandas as pd
import warnings
import tensorflow as tf
warnings.filterwarnings("ignore")
import cv2
import tensorflow as tf
from alexnet import AlexNet
from caffe_classes import class_names
from tensorflow.python.platform import gfile
from tensorflow.python.framework import graph_util


#Function to read in coordinates
def read_coords(n):
    df = pd.read_table(n,header=None)
    x = df[0].to_numpy()
    y = df[1].to_numpy()
    return(x,y)
# =============================================================================
# def read_coords(n):
#     f = open(n)
#     f.readline()
#     x = []
#     y = []
#     for l in f:
#         l = l.rstrip("\n")
#         ll = l.split("\t")
#         this_x = int(float(ll[0]))
#         this_y = int(float(ll[1]))
#         x.append(this_x)
#         y.append(this_y)
#     f.close()
#     x = np.array(x)
#     y = np.array(y)
#     return x,y
# =============================================================================

def read_cellnames(n):
    barcodes = pd.read_table(n,header=None)[0].to_numpy()
    return barcodes
# =============================================================================
# def read_cellnames(n):
#     f = open(n)
#     f.readline()
#     barcodes = []
#     for l in f:
#         l = l.rstrip("\n")
#         barcodes.append(l)
#     f.close()
#     return barcodes
# =============================================================================

#Function to Make Alexnet Morphology features
def make_image_features(folder):
    resize_length = 227
    #folder='output_nissl_cor_3'
    img_file = folder
    save_dir = folder + "/feature/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    imagenet_mean = np.array([104., 117., 124.], dtype=np.float32)
    x = tf.placeholder(tf.float32, [None, 227, 227, 3])
    #############################################
    #x = tf.compat.v1.placeholder(tf.float32, [None, 227, 227, 3])
    #tf.compat.v1.disable_eager_execution()
    #############################################
    keep_prob = tf.constant(1.0)

    model = AlexNet(x, keep_prob, 1000, [])
    feature = model.feature_out

    with tf.Session() as sess:
        load_op = model.load_initial_weights(sess)
        sess.run(load_op)
        allimgs = np.zeros(shape=(1, 227, 227, 3), dtype=np.float)
        for path in os.listdir(img_file):
            print('read image path: '+img_file + path)
            #if path.endswith("path"):
            if path.endswith(".png"):
                print("img found")
                img_name = path
                img_full_path = img_file + "/" + path
                img = cv2.imread(img_full_path)  # img with 3 same channels
                allimgs[0] = img
                feature_realval = sess.run(feature, feed_dict={x: allimgs})
                image_save_path = save_dir
                if not os.path.exists(image_save_path):
                    os.makedirs(image_save_path)
                np.save(image_save_path + '/' + img_name+'.npy', feature_realval)


#Function to read morphology vectors and create morphology matrix
def load_npy(n, log=True, rearrange=None): #n is directory
    img_file = n + "/feature/"
    flist = []
    image_names = []
    ind = 0
    for path in os.listdir(img_file):
        if path.endswith(".png.npy"):
            img_name = path
            img_full_path = img_file + '/' + path
            flist.append(img_full_path)
            image_names.append(img_name.split(".")[0])
            if ind%1000==0:
                print("Finished %d..." % ind)
            ind+=1
    mat = np.empty((len(flist), 4096), dtype="float32")

    for ind,afile in enumerate(flist):
        mat[ind,:] = np.load(afile)
    xx = mat.flatten()
    min_val = np.min(xx[np.nonzero(xx)])/2

    if log:
        mat = np.log2(mat+min_val)

    map_cells = {}
    for ind,val in enumerate(image_names):
        map_cells[val] = ind

    if rearrange is not None:
        indices = []
        for ix in rearrange:
            indices.append(map_cells[ix])
        indices = np.array(indices)
        mat = mat[indices,:]
        image_names = np.array(image_names)[indices]
        image_names = list(image_names)

    n2 = n.replace("/image_crops","")
    np.save(n2 + "/morphology.npy",mat)
    np.savetxt(n2 + "/morphology.txt",mat,delimiter = "\t")
    return mat, image_names

#Function to crop images and mark morphology matrix
def process_morphology(image_file,coord_file,cellname_file,scalefactor,results_dir):
    
    #Create Image Directory
    results_dir = results_dir + "/image_crops"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    #Load Image
    img = cv2.imread(image_file)
    
    #Set coordinates
    x_pixel,y_pixel = read_coords(coord_file)
    barcodes = read_cellnames(cellname_file)
    print(len(x_pixel))
    print(len(barcodes))
    beta = int(scalefactor)
    beta_half=round(beta/2)

    #Test coordinates on the image
    img_new=img.copy()
    sq = int(beta * 0.057)
    for i in range(len(x_pixel)):
        x=x_pixel[i]
        y=y_pixel[i]
        img_new[int(x-sq):int(x+sq), int(y-sq):int(y+sq),:]=0
    
    cv2.imwrite(results_dir + '_sample_coordinates.png', img_new)

    #Create Image Crops
    for j in range(len(x_pixel)):
        max_x=img.shape[0]
        max_y=img.shape[1]
        nbs=img[max(0,x_pixel[j]-beta_half):min(max_x,x_pixel[j]+beta_half+1),max(0,y_pixel[j]-beta_half):min(max_y,y_pixel[j]+beta_half+1)]
        t_factor=float(227)/nbs.shape[0]
        resized = cv2.resize(nbs, None, fx=t_factor, fy=t_factor, interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(results_dir + '/{}.png'.format(barcodes[j]),resized)
    #Create AlexNet Image Vectors
    make_image_features(results_dir)
    
    #Create Morphology Matrix
    morpho_mat,img_names= load_npy(results_dir, log=True, rearrange=barcodes)
    
    return(morpho_mat)
    
    

import sys
if __name__=="__main__":
    process_morphology(sys.argv[1],sys.argv[2],sys.argv[3],int(sys.argv[4]),sys.argv[5])