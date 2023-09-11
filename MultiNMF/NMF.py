# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 15:02:27 2023

@author: zach
"""

from time import time
import numpy as np
import numpy.matlib


def CalculateObj(X=None, U=None, V=None, deltaVU=None, dVordU=None):
    X = X.astype("float32")
    U = U.astype("float32")
    V = V.astype("float32")

    if deltaVU == None:
        deltaVU = 0

    if dVordU == None:
        dVordU = 1

    dV = []
    maxM = 62500000
    mFea, nSmp = X.shape
    mn = np.asarray(X).size
    nBlock = int(np.floor(mn * 3 / maxM))
    if mn < maxM:
        dX = np.matmul(U, V.conj().T) - X
        obj_NMF = np.sum(np.power(dX, 2))
        if deltaVU:
            if dVordU:
                dV = dX.conj().T * U
                dV = dX * V
    else:
        obj_NMF = 0
        if deltaVU:
            if dVordU:
                dV = np.zeros((V.shape, V.shape))
            else:
                dV = np.zeros((U.shape, U.shape))
        for i in np.arange(1, np.ceil(nSmp / nBlock) + 1).reshape(-1):
            if i == np.ceil(nSmp / nBlock):
                smpIdx = np.arange((i - 1) * nBlock + 1, nSmp + 1, dtype=int) - 1
            else:
                smpIdx = np.arange((i - 1) * nBlock + 1, i * nBlock + 1, dtype=int) - 1
            dX = np.matmul(U, V[smpIdx, :].conj().T) - X[:, smpIdx]
            obj_NMF = obj_NMF + np.sum(np.power(dX, 2))
            if deltaVU:
                if dVordU:
                    dV[smpIdx, :] = np.transpose(dX) * U
                else:
                    pass
                    # dV = dU + dX * V[smpIdx,:]
        if deltaVU:
            if dVordU:
                dV = dV

    obj = obj_NMF
    return obj


def NormalizeUV(U=None, V=None, NormV=None, Norm=None):
    U = U.astype("float32")
    V = V.astype("float32")

    nSmp = V.shape[1 - 1]
    mFea = U.shape[1 - 1]
    if Norm == 2:
        if NormV:
            norms = np.sqrt(np.sum(V ** 2, axis=0))
            norms = np.maximum(norms, 1e-10)
            V = V / np.matlib.repmat(norms, nSmp, 1)
            U = np.multiply(U, np.matlib.repmat(norms, mFea, 1))
        else:
            norms = np.sqrt(np.sum(U ** 2, axis=0))
            norms = np.maximum(norms, 1e-10)
            U = U / np.matlib.repmat(norms, mFea, 1)
            V = np.multiply(V, np.matlib.repmat(norms, nSmp, 1))
    else:
        if NormV:
            norms = np.sum(np.abs(V), axis=0)
            norms = np.maximum(norms, 1e-10)
            V = V / np.matlib.repmat(norms, nSmp, 1)
            U = np.multiply(U, np.matlib.repmat(norms, mFea, 1))
        else:
            norms = np.sum(np.abs(U), axis=0)
            U = U / np.matlib.repmat(norms, mFea, 1)
            V = np.multiply(V, np.matlib.repmat(norms, nSmp, 1))
        return U, V


def Normalize(U=None, V=None):
    U = U.astype("float32")
    V = V.astype("float32")

    U, V = NormalizeUV(U, V, 0, 1)
    return (U, V)


def NMF(X=None, k=None, options=None, U_=None, V_=None):
    differror = options['error']
    maxIter = options['maxIter']
    nRepeat = options['nRepeat']
    minIterOrig = options['minIter']
    minIter = minIterOrig - 1
    meanFitRatio = options['meanFitRatio']
    alpha = options['alpha']
    Norm = 1
    NormV = 0
    mFea, nSmp = X.shape
    bSuccess = type('', (), {})()
    bSuccess.bSuccess = 1
    selectInit = 1

    X = X.astype("float32")

    options['Converge'] = False
    if len(U_) == 0:
        U = np.abs(np.random.rand(mFea, k))
        norms = np.sqrt(np.sum(U ** 2, axis=0))
        norms = np.maximum(norms, 1e-10)
        U = U / np.matlib.repmat(norms, mFea, 1)
        if len(V_) == 0:
            V = np.abs(np.random.rand(nSmp, k))
            V = V / (np.sum(V))
        else:
            V = V_
    else:
        U = U_
        if len(V_) == 0:
            V = np.abs(np.random.rand(nSmp, k))
            V = V / (np.sum(V))
        else:
            V = V_

    U, V = NormalizeUV(U, V, NormV, Norm)

    if nRepeat == 1:
        selectInit = 0
        minIterOrig = 0
        minIter = 0
        if len(maxIter) == 0:
            objhistory = CalculateObj(X, U, V)
            meanFit = objhistory * 10
        else:
            if "Converge" in options and options["Converge"] == True:
                objhistory = CalculateObj(X, U, V)
    else:
        if "Converge" in options and options["Converge"] == True:
            raise Exception('Not implemented!')
    tryNo = 0
    print(X.shape, U.shape, V.shape)
    while tryNo < nRepeat:

        tmp_T = time()
        tryNo = tryNo + 1
        nIter = 0
        maxErr = 1
        nStepTrial = 0
        while (maxErr > differror):

            # ===================== update V ========================
            XU = np.matmul(X.conj().T, U, dtype="float32")
            UU = np.matmul(U.conj().T, U, dtype="float32")
            VUU = np.matmul(V, UU, dtype="float32")
            V = np.multiply(V, np.divide(XU, np.maximum(VUU, 1e-10)))
            # ===================== update U ========================
            XV = np.matmul(X, V, dtype="float32")
            VV = np.matmul(V.conj().T, V, dtype="float32")
            UVV = np.matmul(U, VV, dtype="float32")
            U = np.multiply(U, (np.divide(XV, np.maximum(UVV, 1e-10))))

            nIter = nIter + 1
            if nIter > minIter:
                if selectInit:
                    objhistory = CalculateObj(X, U, V)
                    maxErr = 0
                else:
                    if maxIter == None:
                        newobj = CalculateObj(X, U, V)
                        objhistory = np.array([objhistory, newobj])
                        meanFit = meanFitRatio * meanFit + (1 - meanFitRatio) * newobj
                        maxErr = (meanFit - newobj) / meanFit
                    else:
                        if "Converge" in options and options["Converge"] == True:
                            newobj = CalculateObj(X, U, V)
                            objhistory = np.array([objhistory, newobj])
                        maxErr = 1
                        if nIter >= maxIter:
                            maxErr = 0
                            if "Converge" in options and options["Converge"] == True:
                                print("Converge Error")
                                pass
                            else:
                                objhistory = 0

        # elapse = cputime - tmp_T
        elapse = time() - tmp_T
        if tryNo == 1:
            U_final = U
            V_final = V
            nIter_final = nIter
            elapse_final = elapse
            objhistory_final = objhistory
            bSuccess.nStepTrial = nStepTrial
        else:
            if objhistory < objhistory_final:
                U_final = U
                V_final = V
                nIter_final = nIter
                objhistory_final = objhistory
                bSuccess.nStepTrial = nStepTrial
                if selectInit:
                    elapse_final = elapse
                else:
                    elapse_final = elapse_final + elapse
        if selectInit:
            if tryNo < nRepeat:
                # re-start
                if len(U_) == 0:
                    U = np.abs(np.random.rand(mFea, k))
                    norms = np.sqrt(np.sum(U ** 2, axis=0))
                    norms = np.maximum(norms, 1e-10)
                    U = U / np.matlib.repmat(norms, mFea, 1)
                    if len(V_) == 0:
                        V = np.abs(np.random.rand(nSmp, k))
                        V = V / np.sum(V)
                    else:
                        V = V_
                else:
                    U = U_
                    if len(V_) == 0:
                        V = np.abs(np.random.rand(nSmp, k))
                        V = V / np.sum(V)
                    else:
                        V = V_
                U, V = NormalizeUV(U, V, NormV, Norm)
            else:
                tryNo = tryNo - 1
                minIter = 0
                selectInit = 0
                U = U_final
                V = V_final
                objhistory = objhistory_final
                meanFit = objhistory * 10

    nIter_final = nIter_final + minIterOrig
    U_final, V_final = Normalize(U_final, V_final)

    return U_final, V_final
    # return U_final,V_final,nIter_final,elapse_final,bSuccess,objhistory_final
