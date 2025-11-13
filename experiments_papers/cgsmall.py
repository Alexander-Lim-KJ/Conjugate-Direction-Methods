# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 14:08:06 2023

@author: uqalim8
"""

import matplotlib.pyplot as plt
import matplotlib, sys
import torch

sys.path.append("..")

from solvers import ConjugateGradient
from utils import makeFolder


FONT = {'size':14, 'weight':'bold'}
matplotlib.rc('font', **FONT)
    
N = 10
RANGE = range(5, 11)
TERMINATE = RANGE
TOL = 1e-8
REO = False
VERBOSE = True
FOLDER = "./smallcg"

makeFolder(FOLDER) 

for i in RANGE:
    torch.manual_seed(1234)
    b = torch.randn(N, dtype = torch.float64)
    for j in range(0, 1):
        D = torch.randn(N, dtype = torch.float64)
        file_name = FOLDER + f"/CG{i}in.png"
        # if j:
        #     D = torch.randn(N, dtype = torch.float64)
        #     file_name = FOLDER + f"/CG{i}in.png"
        # else:
        #     D = torch.rand(N, dtype = torch.float64)
        #     file_name = FOLDER + f"/CG{i}.png"

        D[i:] = 0
        H = torch.diag(D)
        print("\n", 40 * "=", " Conjugate Gradient ", 40 * "=", "\n")
        CG = ConjugateGradient(H, b, maxit = i, tol = TOL, reO = REO, prinT = VERBOSE)
        CG.solve()
        
        CG = CG.stat
        print("CG:", CG.keys())
        ite = len(CG['|rk|/|b|'])
        
        plt.figure(figsize=(7, 5))
        plt.semilogy(range(0, ite), CG['|Apk|/|Ab|'], linestyle = "--", linewidth = 2, label = "$||\mathbf{Ap}_k||/||\mathbf{Ab}||$")
        plt.semilogy(range(0, ite), CG['|Ark|/|Ab|'], linestyle = "-.", linewidth = 2, label = "$||\mathbf{Ar}_k||/||\mathbf{Ab}||$")
        plt.semilogy(range(0, ite), CG['|rk|/|b|'][:ite], linewidth = 2, label = "$||\mathbf{r}_k||/||\mathbf{b}||$")
        plt.ylim(1e-10, 1e4)
        plt.title(fr"Grades : {min(i + 1, 10)}; dim(Null(A)) : {10 - i}")
        plt.xlabel("iteration k")
        plt.legend()
        plt.savefig(file_name)
        plt.close()