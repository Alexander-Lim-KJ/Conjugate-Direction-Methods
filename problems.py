# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 14:08:06 2023

@author: uqalim8
"""

import matplotlib.pyplot as plt
import matplotlib
import torch, draw
import numpy as np
from solvers import (ConjugateGradient,
                     ConjugateResidual,
                     MinimalResidual)

N = 1000
RANGE = 800
TERMINATE = RANGE
TOL = 1e-8
REO = False
VERBOSE = True
CG, CR, MR = None, None, None
ALGO = ["MR"]

torch.manual_seed(4321)

#D = torch.randn(N, dtype = torch.float64)
D = torch.rand(N, dtype = torch.float64)
#D = torch.tensor(np.random.weibull(a = 5, size = N), dtype = torch.float64)

#D[:int(RANGE * 0.1)] = -D[:int(RANGE * 0.1)]
D[RANGE:] = 0
H = torch.diag(D)

b = torch.randn(N, dtype = torch.float64)
#b = H @ true_sol
#b[RANGE:] = 0

if "CG" in ALGO:
    print("\n", 40 * "=", " Conjugate Gradient ", 40 * "=", "\n")
    CG = ConjugateGradient(H, b, maxit = TERMINATE, tol = TOL, reO = REO, prinT = VERBOSE)
    CG.solve()

if "CR" in ALGO:
    print("\n", 40 * "=", " Conjugate Residual ", 40 * "=", "\n")
    CR = ConjugateResidual(H, b, maxit = TERMINATE, tol = TOL, reO = REO, prinT = VERBOSE)
    CR.solve()

if "MR" in ALGO:
    print("\n", 40 * "=", " Minimal Residual ", 40 * "=", "\n")
    MR = MinimalResidual(H, b, maxit = TERMINATE, tol = TOL, reO = REO, prinT = VERBOSE)
    MR.solve()

if "CGCR" in ALGO:
    print("\n", 40 * "=", " Conjugate ResGrad ", 40 * "=", "\n")
    MR = MinimalResidual(H, b, maxit = TERMINATE, tol = TOL, reO = REO, prinT = VERBOSE)
    MR.solve()
# plt.semilogy(range(0, MR.ite + 1), torch.norm(MR._X - CR._X, float("inf"), dim = 0), 
#              label = "$||\mathbf{x}_k^{MINRES} - \mathbf{x}_k^{CR}||_{\infty}$")
# plt.title(f"(Indefinite Matrix) Grades : {RANGE}")
# plt.xlabel("iteration k")
# plt.ylim(1e-18, 1e-8)
# plt.legend()
# plt.show()

#draw.plotGraphs(CG, CR, MR, "./comparisons")