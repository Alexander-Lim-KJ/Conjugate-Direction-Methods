# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 09:47:02 2025

@author: uqalim8
"""

import json, torch, matplotlib 
import numpy as np
import matplotlib.pyplot as plt

FONT = {'size':14, 'weight':'bold'} 
matplotlib.rc('font', **FONT)

with open("./gauss_eigs.json", "r") as f:
    eigs = json.load(f)
    
eigs = np.array(eigs, dtype = np.float64)

A = np.outer(eigs,eigs).reshape(-1)

#logbins = np.logspace(np.log10(min(abs(A))), np.log10(max(abs(A))), 400)

plt.figure(figsize=(8,3))
#plt.hist(abs(A), bins = logbins, log = True)
#plt.xscale("log")
plt.plot(np.sort(abs(A))[::-1])
plt.yscale("log")
plt.title("Singular values of Gaussian blurring matrix")
plt.xlabel("$n^{th}$ singular values")
plt.ylabel("Singular values")
plt.show()