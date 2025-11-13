# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 20:21:08 2023

@author: uqalim8
"""

import numpy as np
import torch, datasets, neuralNetwork, optAlgs, os, json
import matplotlib.pyplot as plt
from constants import cCUDA, cTYPE

TEXT = "{:<20} : {:>20}"

def makeFolder(folder_path):
    if folder_path[:2] != "./":
        folder_path = "./" + folder_path
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)

def saveRecords(folder_path, name, file):
    makeFolder(folder_path)
    if folder_path[-1] != "/":
        folder_path += "/"
    folder_path += name + ".json"
    with open(folder_path, "w") as f:
        json.dump(file, f)

def openRecords(folder_path, name):
    if folder_path[-1] != "/":
        folder_path += "/"
    with open(folder_path + name, "r") as f:
        return json.load(f)
    
def openAllRecords(folder_path):
    if folder_path[-1] != "/":
        folder_path += "/"
    files = filter(lambda x : ".json" in x, os.listdir(f"{folder_path}"))
    records = {}
    for i in files:
        record = openRecords(folder_path, i)
        records[i.split(".json")[0]] = record
    return records
                
def execute(folder_path, dataset, problem, algo, x0, const, verbose):
    makeFolder(folder_path)
    
    if problem == "GN":
        one_hot = False
    else:
        one_hot = True
        
    trainX, trainY, _, _ = datasets.prepareData(folder_path, one_hot, dataset)
    trainX = trainX.to(cCUDA)
    trainY = trainY.to(cCUDA)
    print(TEXT.format("No. Samples", trainY.shape[0]))
    x0, pred, func = initFunc_x0(x0, problem, trainX, trainY)
    algo = initAlg(func, x0, algo, const)
    algo.optimize(verbose, pred)
    return algo, x0

def initFunc_x0(x0, problem, trainX, trainY):
    d = trainX.shape[1]

    if problem == "GN":
        ffnn = neuralNetwork.FFN_GN(d)
        func = neuralNetwork.NLSGaussNewton(ffnn, trainX, trainY)
        
    elif problem == "NN":
        ffnn = neuralNetwork.FFN_NN(d)
        fun = neuralNetwork.nnWrapper(ffnn, torch.nn.MSELoss())
        func = lambda x, v : fun(x, v, trainX, trainY)
    
    w = torch.nn.utils.parameters_to_vector(ffnn.parameters())
    print(TEXT.format("Dimensions", w.shape[0]))
    if not x0 == "torch":
        w = initx0(x0, w.shape[0]).detach()
    else:
        w = initx0(w, None).detach()
        
    return w, lambda v : 0, func

def initx0(x0_type, size):
    
    if not type(x0_type) == str:
        print(TEXT.format("x0", "initialised"))
        x0_type = x0_type.to(cCUDA)
        return x0_type.to(cTYPE)
    
    if x0_type == "ones":
        print(TEXT.format("x0", x0_type))
        return torch.ones(size, dtype = cTYPE, device = cCUDA)
    
    if x0_type == "zeros":
        print(TEXT.format("x0", x0_type))
        return torch.zeros(size, dtype = cTYPE, device = cCUDA)
    
    if x0_type == "normal":
        print(TEXT.format("x0", x0_type))
        return torch.randn(size, dtype = cTYPE, device = cCUDA)
    
    if x0_type == "uniform":
        print(TEXT.format("x0", x0_type))
        return torch.rand(size, dtype = cTYPE, device = cCUDA)
    
def initAlg(fun, x0, algo, c):
    
    if algo == "NewtonCG":
        print(TEXT.format("Algorithm", algo))
        return optAlgs.NewtonCG(fun, x0, c.alpha0, c.gradtol, c.maxite, c.maxorcs, 
                                c.restol, c.inmaxite, c.lineMaxite, c.lineBeta, c.lineRho)
        
    if algo == "NewtonMR_NC":
        print(TEXT.format("Algorithm", algo))
        return optAlgs.NewtonMR_NC(fun, x0, c.alpha0, c.gradtol, c.maxite, 
                                   c.maxorcs, c.restol, c.inmaxite, c.lineMaxite, 
                                   c.lineBetaB, c.lineRho, c.lineBetaFB, 1)   

    if algo == "NewtonCR_NC":
        print(TEXT.format("Algorithm", algo))
        return optAlgs.NewtonCR_NC(fun, x0, c.alpha0, c.gradtol, c.maxite, 
                                   c.maxorcs, c.restol, c.inmaxite, c.lineMaxite, 
                                   c.lineBetaB, c.lineRho, c.lineBetaFB, 1)                
    
    if algo == "NewtonCG_TR_Steihaug":
        print(TEXT.format("Algorithm", algo))
        return optAlgs.NewtonCG_TR_Steihaug(fun, x0, c.gradtol, c.maxite, c.maxorcs, 
                                            c.restol, c.inmaxite, c.deltaMax, c.delta0, 
                                            c.eta, c.eta1, c.eta2, c.gamma1, c.gamma2, 1)
    
    if algo == "L-BFGS":
        print(TEXT.format("Algorithm", algo))
        return optAlgs.L_BFGS(fun, x0, c.alpha0, c.gradtol, c.m, 
                              c.maxite, c.maxorcs, c.lineMaxite)
    
def full_matrix(D):
    n = D.shape[0]
    M = torch.rand((n, n), dtype = torch.float64)
    V = torch.qr(M + M.T)[0]
    return V.T @ torch.diag(D) @ V, V

