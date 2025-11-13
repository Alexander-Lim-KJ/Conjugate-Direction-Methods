# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 18:31:13 2022

@author: uqalim8
"""

from linesearch import (backwardArmijo, 
                        backForwardArmijo,
                        backForwardArmijo_mod, 
                        dampedNewtonCGLinesearch, 
                        dampedNewtonCGbackForwardLS, 
                        lineSearchWolfeStrong)
import torch, time
from CG import CG, CGSteihaug
from CR import CR
from MINRES import myMINRES
from constants import cTYPE

GD_STATS = {"ite":"g", "orcs":"g", "time":".2f", "f":".4e", 
            "g_norm":".4e", "alpha":".2e", "acc":".2f"}

SGD_STATS = {"ite":"g", "orcs":"g", "time":".2f", "f":".4e", 
            "g_norm":".4e", "acc":".2f"}

NEWTON_STATS = {"ite":"g", "inite":"g", "orcs":"g", "time":".2f", 
                "f":".4e", "g_norm":".4e", "alpha":".2e", "acc":".2f"}
                   
NEWTON_NC_STATS = {"ite":"g", "inite":"g", "dtype":"", "relRes":".2e", "relARes":".2e", "res-res":".2e", "ortho":".2e", "orcs":"g",
                   "f":".2e", "g_norm":".2e", "alpha":".2e"}

NEWTON_TR_STATS = {"ite":"g", "inite":"g", "dtype":"", "orcs":"g", "time":".2f",
                   "f":".4e", "g_norm":".4e", "delta":".2e", "acc":".2f"}

L_BFGS_STATS = {"ite":"g", "orcs":"g", "time":".2f", "f":".4e", "g_norm":".4e", "iteLS":"g", 
                "alpha":".2e", "acc":".2f"}

class Optimizer:
    
    def __init__(self, fun, x0, alpha0, gradtol, maxite, maxorcs):
        self.fun = fun
        self.xk = x0
        self.alpha0 = alpha0
        self.maxorcs = maxorcs
        self.k, self.orcs, self.toc, self.lineite = 0, 0, 0, 0
        self.gknorm, self.record = None, None
        self.maxite = maxite
        self.gradtol = gradtol
        self.alphak = 1
        self.record = dict(((i, []) for i in self.info.keys()))
        
    def recording(self, stats):
        for n, i in enumerate(self.record.keys()):
            self.record[i].append(stats[n])
            
    def printStats(self):
        if self.k == 0:
            print(5 * len(self.info) * "..")
            form = ["{:^10}"] * len(self.info)
            print("|".join(form).format(*self.info.keys()))
            print(5 * len(self.info) * "..")
        form = ["{:^10" + i + "}" for i in self.info.values()]
        print("|".join(form).format(*(self.record[i][-1] for i in self.info.keys())))        
    
    def progress(self, verbose, pred, print_skip = 1):
        self.k += 1
        self.oracleCalls()
        self.recordStats(pred(self.xk))
        if verbose and self.k % print_skip == 0:
            self.printStats()

    def termination(self):
        return self.k > self.maxite or self.gknorm < self.gradtol or self.orcs > self.maxorcs or self.alphak < 1e-18
    
    def optimize(self, verbose, pred):
        self.recordStats(pred(self.xk))
        self.printStats()
        while not self.termination():
            tic = time.time()
            self.step()
            self.toc += time.time() - tic
            self.progress(verbose, pred)

    def recordStats(self):
        raise NotImplementedError
        
    def step(self):
        raise NotImplementedError
    
    def oracleCalls(self):
        raise NotImplementedError
        
class NewtonCG(Optimizer):
    
    def __init__(self, fun, x0, alpha0, gradtol, maxite, maxorcs, restol, inmaxite, 
                 lineMaxite, lineBetaB, lineRho):
        self.info = NEWTON_NC_STATS
        self.restol = restol
        self.inmaxite = inmaxite
        self.lineMaxite = lineMaxite
        self.lineBetaB = lineBetaB
        self.lineBetaFB = 1e-4
        self.lineRho = lineRho
        self.alpha_npc = 1
        super().__init__(fun, x0, alpha0, gradtol, maxite, maxorcs)
        
    def step(self):
        pk, self.inite, self.dtype, self.relre, self.relAre, self.stable, self.bAp = CG(self.hk, -self.gk, rtol = self.restol, maxit = self.inmaxite)
        
        if self.dtype == "Sol" or self.dtype == "MAX":
            self.alphak, self.lineite = backwardArmijo(lambda x : self.fun(x, "0"), self.xk, self.fk, self.gk, self.alpha0, pk,
                                         self.lineBetaB, self.lineRho, self.lineMaxite)
        else:
            self.alphak, self.lineite = backForwardArmijo_mod(lambda x : self.fun(x, "0"), 
                                                              self.xk, self.fk, self.gk, self.alpha_npc, pk, 
                                                              self.lineBetaFB, self.lineRho, self.lineMaxite)
            self.alpha_npc = self.alphak
            
        self.xk += self.alphak * pk
        self.fk, self.gk, self.hk = self.fun(self.xk, "012")
    
    def recordStats(self, acc):
        if self.k == 0:
            self.fk, self.gk, self.hk = self.fun(self.xk, "012")
            self.inite = 0
            self.gknorm = torch.linalg.norm(self.gk, float('inf'))
            self.recording((0, 0, "None", 0, 0, 0, 0, 0, float(self.fk), float(self.gknorm), 0))
        else:
            self.gknorm = torch.linalg.norm(self.gk, float('inf'))
            self.recording((self.k, self.inite, self.dtype, float(self.relre), float(self.relAre), 
                            float(self.stable), float(self.bAp), self.orcs, float(self.fk), float(self.gknorm), self.alphak))
        
    def oracleCalls(self):
        self.orcs += 2 + 2 * self.inite + self.lineite
        
class NewtonMR_NC(Optimizer):
    
    def __init__(self, fun, x0, alpha0, gradtol, maxite, maxorcs, restol, inmaxite, 
                 lineMaxite, lineBetaB, lineRho, lineBetaFB, Hsub):
        self.info = NEWTON_NC_STATS
        self.restol = restol
        self.inmaxite = inmaxite
        self.lineMaxite = lineMaxite
        self.lineBetaB = lineBetaB
        self.lineRho = lineRho
        self.lineBetaFB = lineBetaFB
        self.Hsub = Hsub
        self.alpha_npc = 1
        super().__init__(fun, x0, alpha0, gradtol, maxite, maxorcs)
        
    def step(self):
        pk, self.inite, r, self.dtype, self.relre, self.relAre, self.stable, self.bdotv = myMINRES(self.hk, -self.gk, rtol = self.restol, maxit = self.inmaxite)
        
        if self.dtype == "Sol" or self.dtype == "MAX":
            self.alphak, self.lineite = backwardArmijo(lambda x : self.fun(x, "0"), 
                                                       self.xk, self.fk, self.gk, self.alpha0, pk, 
                                                       self.lineBetaB, self.lineRho, self.lineMaxite)
        else:
            self.alphak, self.lineite = backForwardArmijo_mod(lambda x : self.fun(x, "0"), 
                                                          self.xk, self.fk, self.gk, self.alpha_npc, r, 
                                                          self.lineBetaFB, self.lineRho, self.lineMaxite)
            self.alpha_npc = self.alphak
            pk = r
        
        self.xk += self.alphak * pk
        self.fk, self.gk, self.hk = self.fun(self.xk, "012")
        
    def recordStats(self, acc):
        if self.k == 0:
            self.fk, self.gk, self.hk = self.fun(self.xk, "012")
            self.inite = 0
            self.gknorm = torch.linalg.norm(self.gk, float('inf'))
            self.recording((0, 0, "None", 0, 0, 0, 0, 0, float(self.fk), float(self.gknorm), 0))
        else:
            self.gknorm = torch.linalg.norm(self.gk, float('inf'))
            self.recording((self.k, self.inite, self.dtype, float(self.relre), float(self.relAre), float(self.stable), float(self.bdotv), 
                            self.orcs, float(self.fk), float(self.gknorm), self.alphak))
        
    def oracleCalls(self):
        self.orcs += 2 + 2 * self.inite * self.Hsub + self.lineite
        
class NewtonCR_NC(Optimizer):
    
    def __init__(self, fun, x0, alpha0, gradtol, maxite, maxorcs, restol, inmaxite, 
                 lineMaxite, lineBetaB, lineRho, lineBetaFB, Hsub):
        self.info = NEWTON_NC_STATS
        self.restol = restol
        self.inmaxite = inmaxite
        self.lineMaxite = lineMaxite
        self.lineBetaB = lineBetaB
        self.lineRho = lineRho
        self.lineBetaFB = lineBetaFB
        self.Hsub = Hsub
        self.alpha_npc = 1
        super().__init__(fun, x0, alpha0, gradtol, maxite, maxorcs)
        
    def step(self):
        pk, self.inite, self.dtype, self.relre, self.relAre, self.stable, self.bdotr = CR(self.hk, -self.gk, rtol = self.restol, maxit = self.inmaxite)
        
        if self.dtype == "Sol" or self.dtype == "MAX":
            self.alphak, self.lineite = backwardArmijo(lambda x : self.fun(x, "0"), 
                                                       self.xk, self.fk, self.gk, self.alpha0, pk, 
                                                       self.lineBetaB, self.lineRho, self.lineMaxite)
        else:
            self.alphak, self.lineite = backForwardArmijo_mod(lambda x : self.fun(x, "0"), 
                                                          self.xk, self.fk, self.gk, self.alpha_npc, pk, 
                                                          self.lineBetaFB, self.lineRho, self.lineMaxite)
            self.alpha_npc = self.alphak
        
        self.xk += self.alphak * pk
        self.fk, self.gk, self.hk = self.fun(self.xk, "012")
        
    def recordStats(self, acc):
        if self.k == 0:
            self.fk, self.gk, self.hk = self.fun(self.xk, "012")
            self.inite = 0
            self.gknorm = torch.linalg.norm(self.gk, float('inf'))
            self.recording((0, 0, "None", 0, 0, 0, 0, 0, float(self.fk), float(self.gknorm), 0))
        else:
            self.gknorm = torch.linalg.norm(self.gk, float('inf'))
            self.recording((self.k, self.inite, self.dtype, float(self.relre), float(self.relAre), float(self.stable), float(self.bdotr), 
                            self.orcs, float(self.fk), float(self.gknorm), self.alphak))
        
    def oracleCalls(self):
        self.orcs += 2 + 2 * self.inite * self.Hsub + self.lineite
        
class NewtonCG_TR_Steihaug(Optimizer):
    
    def __init__(self, fun, x0, gradtol, maxite, maxorcs, restol, inmaxite, 
                 deltaMax, delta0, eta, eta1, eta2, gamma1, gamma2, Hsub):
        
        if not (0 < eta1 and eta1 <= eta2 and eta2 < 1 and eta < eta1):
            raise Exception("etas 0 < eta < eta1 <= eta2 < 1")
        
        if not ((0 < gamma1 and gamma1 < 1) and (gamma2 > 1)):
            raise Exception("0 < gamma1 < 1 and gamma2 > 1")
        
        self.info = NEWTON_TR_STATS
        self.restol = restol
        self.inmaxite = inmaxite
        self.delta = delta0
        self.deltaMax = deltaMax
        self.eta = eta
        self.eta1 = eta1
        self.eta2 = eta2
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.Hsub = Hsub
        super().__init__(fun, x0, 0, gradtol, maxite, maxorcs)
        
    def step(self):
        pk, self.dtype, m, self.inite = CGSteihaug(self.hk, self.gk, self.delta, self.restol, self.inmaxite)
        self.rho = (self.fk - self.fun(self.xk + pk, "0")) / m
        
        if self.rho < self.eta1:
            self.delta *= self.gamma1
        
        else:
            if self.rho > self.eta2 and self.dtype == "SOL,=":
                self.delta = min(self.delta * self.gamma2, self.deltaMax)
        
        if self.rho > self.eta:
            self.xk = self.xk + pk
            self.fk, self.gk, self.hk = self.fun(self.xk, "012")
            
    def recordStats(self, acc):
        if self.k == 0:
            self.fk, self.gk, self.hk = self.fun(self.xk, "012")
            self.inite = 0
            self.gknorm = torch.linalg.norm(self.gk, 2)
            self.recording((0, 0, "None", 0, 0, float(self.fk), 
                             float(self.gknorm), self.delta, acc))
        else:
            self.gknorm = torch.linalg.norm(self.gk, 2)
            self.recording((self.k, self.inite, self.dtype, self.orcs, 
                               self.toc, float(self.fk), float(self.gknorm), self.delta, acc))  
            
    def oracleCalls(self):
        self.orcs += 2 + 2 * self.inite * self.Hsub + 2

class L_BFGS(Optimizer):

    def __init__(self, fun, x0, alpha0, gradtol, m, maxite, maxorcs, lineMaxite):
        self.info = L_BFGS_STATS
        self.m = m
        self.lineMaxite = lineMaxite
        super().__init__(fun, x0, alpha0, gradtol, maxite, maxorcs)
        
    def _twoloop(self, w):
        k = self.s.shape[0]
        alpha, rho = torch.zeros(k, dtype = cTYPE), torch.zeros(k, dtype = cTYPE)
        for i in range(k):
            rho[i] = 1/torch.dot(self.s[i], self.y[i])
            alpha[i] = rho[i] * torch.dot(self.s[i], w)
            w = w - alpha[i] * self.y[i]
            
        w = ((torch.dot(self.s[0], self.y[0])) / torch.dot(self.y[0], self.y[0])) * w
        for i in range(k - 1, -1, -1):
            beta = rho[i] * torch.dot(self.y[i], w)
            w = w + (alpha[i] - beta) * self.s[i]

        return w
    
    def step(self):
        
        if not self.k:
            pk = -self.gk
        else:
            pk = self._twoloop(-self.gk).detach()
        
        self.alpha, self.lineite, self.lineorcs = lineSearchWolfeStrong(lambda x : self.fun(x, "01"), self.xk, pk, 
                                                         self.fk, self.gk, self.alpha0, 1e5, 1e-4, 0.9, self.lineMaxite)
        xkp1 = self.xk + self.alpha * pk
        self.fk, gkp1 = self.fun(xkp1, "01")

        # kill small alpha and terminate
        if self.alpha == 0:
            self.orcs = self.maxorcs
            self.lineorcs = 0

        if self.k and self.s.shape[0] >= self.m:
            self.s = self.s[:-1]
            self.y = self.y[:-1]
            
        temps = xkp1 - self.xk
        tempy = gkp1 - self.gk     
        
        if not self.k:
            self.s = temps.reshape(1, -1)
            self.y = tempy.reshape(1, -1)
        else:
            self.s = torch.cat([temps.reshape(1, -1), self.s], dim = 0)
            self.y = torch.cat([tempy.reshape(1, -1), self.y], dim = 0)
            
        self.gk = gkp1
        self.xk = xkp1

    def recordStats(self, acc):
        if self.k == 0:
            self.fk, self.gk = self.fun(self.xk, "01")
            self.inite = 0
            self.gknorm = torch.linalg.norm(self.gk, 2)
            self.recording((0, 0, 0, float(self.fk), 
                             float(self.gknorm), 0, 0, acc))
        else:
            self.gknorm = torch.linalg.norm(self.gk, 2)
            self.recording((self.k, self.orcs, self.toc, float(self.fk), 
                            float(self.gknorm), self.lineite, float(self.alpha), acc))  
            
    def oracleCalls(self):
        self.orcs += 2 + self.lineorcs

if __name__ == "__main__":
    from loss_funcs import logisticFun, logisticModel
    from loadData import loadData
    
    A_train, b_train, *_ = loadData()
    fun = lambda x, v : logisticFun(x, A_train, b_train, 1, v)
    x0 = torch.zeros(A_train.shape[-1], dtype = cTYPE)
    Lg = torch.linalg.matrix_norm(A_train, 2)**2 / 4 + 1
    
    def pred(x):
        b_pred = torch.round(logisticModel(A_train, x))
        return torch.sum(b_train == b_pred) / len(b_train) * 100
    
    # print("=================== LineSearch GD ========================")
    # optGD = linesearchGD(fun, x0.clone(), 10/Lg, 10e-4, 1000, 1000, 100, 10e-4, 0.9)
    # optGD.optimize(True, pred)
    
    # print("=================== Adam ========================")
    # optAD = Adam(fun, x0.clone(), 10e-4, 1000, 1000)
    # optAD.optimize(True, pred)
    
    # print("=================== Newton CG ========================")
    # optNEWTON = NewtonCG(fun, x0.clone(), 1, 10e-4, 1000, 1000, 10e-2, 100, 100, 10e-4, 0.9)
    # optNEWTON.optimize(True, pred)
    
    # print("=================== Newton MR NC ========================")
    # optNEWTONMR = NewtonMR_NC(fun, x0.clone(), 1, 10e-4, 1000, 1000, 10e-2, 100, 100, 10e-4, 0.9, 10e-4, 1)
    # optNEWTONMR.optimize(True, pred)
    
    # print("=================== Newton CG NC ========================")
    # optNEWTONCG = NewtonCG_NC(fun, x0.clone(), 1, 10e-4, 1000, 1000, 10e-2, 1000, 1000, 10e-4, 0.9, 10e-4, 1)
    # optNEWTONCG.optimize(True, pred)
    
    print("=================== L-BFGS ========================")
    optL_BFGS = L_BFGS(fun, x0.clone(), 1, 10e-4, 20, 1000, 1000, 1000)
    optL_BFGS.optimize(True, pred)
    
    # print("=================== Newton TR ========================")
    # optNewtonTR = NewtonCG_TR_Steihaug(fun, x0.clone(), 10e-4, 1000, 1000, 
    #                                    10e-2, 1000, 1e10, 1e5, 1/8, 0.25, 0.75, 0.25, 2, 1)
    # optNewtonTR.optimize(True, pred)
    
    import matplotlib.pyplot as plt
    
    fig = plt.figure()
    # plt.loglog(torch.tensor(optGD.record["orcs"]) + 1, optGD.record["f"], label = "GD")
    # plt.loglog(torch.tensor(optAD.record["orcs"]) + 1, optAD.record["f"], label = "Adam")
    # plt.loglog(torch.tensor(optNEWTON.record["orcs"]) + 1, optNEWTON.record["f"], label = "NewtonCG")
    # plt.loglog(torch.tensor(optNEWTONMR.record["orcs"]) + 1, optNEWTONMR.record["f"], label = "NewtonMR_NC")
    # plt.loglog(torch.tensor(optNEWTONCG.record["orcs"]) + 1, optNEWTONCG.record["f"], label = "NewtonCG_NC")
    # plt.loglog(torch.tensor(optNewtonTR.record["orcs"]) + 1, optNewtonTR.record["f"], label = "NewtonTR")
    plt.loglog(torch.tensor(optL_BFGS.record["orcs"]) + 1, optL_BFGS.record["f"], label = "L_BFGS")

    plt.legend()
    plt.show()
    
    fig = plt.figure()
    # plt.loglog(torch.tensor(optGD.record["orcs"]) + 1, optGD.record["g_norm"], label = "GD")
    # plt.loglog(torch.tensor(optAD.record["orcs"]) + 1, optAD.record["g_norm"], label = "Adam")
    # plt.loglog(torch.tensor(optNEWTON.record["orcs"]) + 1, optNEWTON.record["g_norm"], label = "NewtonCG")
    # plt.loglog(torch.tensor(optNEWTONMR.record["orcs"]) + 1, optNEWTONMR.record["g_norm"], label = "NewtonMR_NC")
    # plt.loglog(torch.tensor(optNEWTONCG.record["orcs"]) + 1, optNEWTONCG.record["g_norm"], label = "NewtonCG_NC")
    # plt.loglog(torch.tensor(optNewtonTR.record["orcs"]) + 1, optNewtonTR.record["g_norm"], label = "NewtonTR")
    plt.loglog(torch.tensor(optL_BFGS.record["orcs"]) + 1, optL_BFGS.record["g_norm"], label = "L_BFGS")

    plt.legend()
    plt.show()
