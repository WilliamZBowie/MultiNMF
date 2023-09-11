#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 11:53:28 2023

@author: zach
"""

import numpy as np
from NMF import NMF
from PerViewNMF import PerViewNMF
from time import time
    
def MultiNMF(X ,K ,options):
    """
    Function to run Multi-View Non-negative Matrix Factorization (MultiNMF)
    Takes inputs and performs MultiNMF algorithm
    :param X: A list containing Matrices for each view in the dimensions of [feature x sample]
    :type X: list
    :param K: Number of components/factors to generate with MultiNMF
    :type K:int
    :param options: Dictionary containing auxiliary parameters for MultiNMF, see documentation for detail on specific variables
    :type options: dict
    :return: Arrays containing basis and component matrices for each view, and then the final combined component matrix, for each view
    :rtype: array structures
    """

    #Set params
    viewNum = len(X) 
    Rounds = options["rounds"]
    alpha = options["alpha"]

    #Intialize MultiNMF objects matricies
    U_ = []
    V_ = []
    U = np.empty((viewNum,0)).tolist()
    V = np.empty((viewNum,0)).tolist()
    j = 0
    log = [0]
    ac = 0

    while j < 3:
        j = j + 1
        if j == 1:
            print('started j==1')
            U[0],V[0] = NMF(X[0],K,options,U_,V_)
            print('reached j==1')
        else:
            print('started x2')
            U[0],V[0] = NMF(X[0],K,options,U_,V[viewNum-1])
            print('reached x2')
        for i in range(1,viewNum):
            print('started for i = 2')
            U[i],V[i] = NMF(X[i],K,options,U_,V[i - 1])
            print('reached for i = 2:viewNum')

    
    optionsForPerViewNMF = options.copy()
    oldL = 100
    
    tic = time()
    j = 0
    oldU = U
    oldV = V
    while j < Rounds:

        j = j + 1
        if j == 1:
            centroidV = V[0]
        else:
            centroidV = alpha[0] * V[0]
            for i in range(1,viewNum):
                centroidV = centroidV + alpha[i] * V[i]
            centroidV = centroidV / np.sum(alpha,axis=0)
        logL = 0
        for i in range(0,viewNum):
            tmp1 = X[i] - U[i] @ np.transpose(V[i])
            tmp2 = V[i] - centroidV
            logL = logL + np.sum(np.square(tmp1)) + alpha[i]*np.sum(np.square(tmp2))
        log.append(logL)
        print("LogL = ",logL)
        if (oldL < logL):
            U = oldU
            V = oldV
            logL = oldL
            j = j - 1
            print('restart this iteration')
        else:
            # ac(end+1) = printResult(centroidV, label, K, options.kmeans);
            print('Reached ac(end+1)')
        oldU = U
        oldV = V
        oldL = logL
        
        for i in range(0,viewNum):
            optionsForPerViewNMF["alpha"] = alpha[i]
            U[i],V[i] = PerViewNMF(X[i],K,centroidV,optionsForPerViewNMF,U[i],V[i])

    
    toc = time() - tic
    print("Elapsed time is {} seconds.".format(toc))
    return U,V,centroidV
