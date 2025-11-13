# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 10:32:10 2023

@author: uqalim8
"""

import torch
from constants import cTYPE, cCUDA

class Solvers:
    
    def __init__(self, A, b, maxit, tol, reO, x0, prinT):
        
        if x0 is None:
            self.xk = torch.zeros_like(b, dtype = cTYPE, device = cCUDA)
        else:
            self.xk = x0
            
        if maxit is None:
            self._maxit = b.shape[0]
        else:
            self._maxit = maxit
            
        self.A = A
        self.b = b
        self._tol = tol
        self._reO = reO
        self._print = prinT
        self.ite = 0
        
    def reOrthogonalize(self, M, vec):
        if M is None: 
            M = vec.reshape(-1, 1) / torch.norm(vec)
        else:
            vec = vec - M @ Avec(M.T, vec) 
            M = torch.concat([M, vec.reshape(-1, 1) / torch.norm(vec)], dim = 1)
        return M, vec
    
    def terminate(self, k, norm):
        return k < self._maxit and norm > self._tol

    def storePrintStats(self, k, *args):
        if self._print:
            length = len(self._STATS)
            items = "{:^5}" + (length - 1) * " | {:^11}"
            statistics = "{:^5}" + (length - 1) * " | {:^11.2e}"
            if not k % 100:
                print(length * 13 * "-" + "-")
                print(items.format(*self._STATS))
                print(length * 13 * "-" + "-")
            print(statistics.format(k, *args))
        
        for key, val in zip(self.stat.keys(), map(float, args)):
            self.stat[key].append(val)
        
    def iterate(self):
        raise NotImplementedError()
        
    def solve(self):
        raise NotImplementedError()
    
class ConjugateGradient(Solvers):
    # "αk"
    #_STATS = ("ite", "αk", "|rk|/|b|", "|Ar|/|Ab|", "|Apk|/|Ab|", "<b, Apk>", "<b, rk>", "A(αp)-(αA)p", "|b-Ax-r|")
    _STATS = ("ite", "|rk|/|b|", "|Ark|/|Ab|", "|Apk|/|Ab|", "|b-Ax-r|", "pred")

    def __init__(self, A, b, maxit = None, tol = 1e-8, reO = False, x0 = None, prinT = True):
        super().__init__(A, b, maxit, tol, reO, x0, prinT)
        self.stat = {k : [] for k in self._STATS[1:]}
        
    def iterate(self, xk, rk, pk, Ap, pAp, norm_r, R):
        alpha = norm_r ** 2 / pAp
        xk = xk + alpha * pk
        rk = rk - alpha * Ap
        
        # re-orthogonalization
        if not R is None:
            rk = rk - R @ Avec(R.T, rk)
            norm_rk = torch.norm(rk)
            R = torch.concat([R, rk.reshape(-1, 1) / norm_rk], dim = 1)
        else:
            norm_rk = torch.norm(rk)
        
        beta = (norm_rk / norm_r) ** 2   
        pk = rk + beta * pk
        return xk, rk, pk, norm_rk, R
    
    def solve(self, pred = lambda x : 0):
        rk = self.b - Avec(self.A, self.xk)
        r0, pk = rk, rk
        Ap = Avec(self.A, pk)
        pAp = torch.dot(Ap, pk)
        norm_r0 = torch.norm(r0)
        norm_rk = norm_r0
        norm_Ar0 = torch.norm(Ap)
        norm_Ap, norm_Ark = norm_Ar0, norm_Ar0
        
        # re-orthogonalization
        if self._reO:
            R = rk.reshape(-1, 1) / norm_rk
        else:
            R = None
        
        # Rx = rk.reshape(-1, 1) / torch.norm(rk)
        # AP = Ap.reshape(-1, 1) / torch.norm(Ap)
        # P = pk.reshape(-1, 1) / torch.norm(pk)
        # PAP = [float(pAp / (torch.norm(Ap) * torch.norm(pk)))]
        self.storePrintStats(self.ite, 
                             relnorm := norm_rk / norm_r0, 
                             relAnorm := norm_Ark / norm_Ar0, 
                             relApnorm := torch.norm(Ap) / norm_Ar0,
                             # abs(torch.dot(r0, Ap)) / (norm_r0 * torch.norm(Ap)), 
                             # abs(torch.dot(r0, rk)) / (norm_r0 * norm_rk), 
                             torch.norm((self.b - Avec(self.A, self.xk)) - rk),
                             pred(self.xk))    

        while self.terminate(self.ite, relApnorm):
            
            self.xk, rk, pk, norm_rk, R = self.iterate(self.xk, rk, pk, Ap, pAp, norm_rk, R)
            
            # update 
            Ap = Avec(self.A, pk)
            pAp = torch.dot(pk, Ap)
            self.ite += 1
            
            
            # Rx = torch.concat([Rx, rk.reshape(-1, 1) / torch.norm(rk)], dim = -1)
            # AP = torch.concat([AP, Ap.reshape(-1, 1) / torch.norm(Ap)], dim = -1)
            # P = torch.concat([P, pk.reshape(-1, 1) / torch.norm(pk)], dim = -1)
            # PAP.append(float(pAp / (torch.norm(Ap) * torch.norm(pk))))
            
            norm_Ark = torch.norm(Avec(self.A, rk))
            self.storePrintStats(self.ite,
                                  relnorm := norm_rk / norm_r0, 
                                  relAnorm := norm_Ark / norm_Ar0, 
                                  relApnorm := torch.norm(Ap) / norm_Ar0,
                                  # abs(torch.dot(r0, Ap)) / (norm_r0 * torch.norm(Ap)), 
                                  # abs(torch.dot(r0, rk)) / (norm_r0 * norm_rk), 
                                  torch.norm((self.b - Avec(self.A, self.xk)) - rk),
                                  pred(self.xk))
            
            # self.storePrintStats(self.ite,
            #                       relnorm := norm_rk / norm_r0, 
            #                       relAnorm := norm_Ark / norm_Ar0, 
            #                       torch.norm(Ap) / norm_Ar0, 
            #                       torch.norm(torch.diag(torch.tensor(PAP)) - P.T @ AP, p = 2),
            #                       torch.norm(torch.eye(self.ite + 1) - Rx.T @ Rx, p = 2),
            #                       torch.norm((self.b - Avec(self.A, self.xk)) - rk))
        
        self.stat["xk"] = self.xk.tolist()
        self.xk = self.xk - torch.dot(self.xk, rk) * rk / (norm_rk ** 2)
        self.stat["xk_lifted"] = self.xk.tolist()
            
class ConjugateResidual(Solvers):
    
    #_STATS = ("ite", "|rk|/|b|", "|Ark|/|Ab|", "<b, Ark>", "<Ab, Ap>", "A(αp)-(αA)p", "|b-Ax-r|")
    _STATS = ("ite", "|rk|/|b|", "|Ark|/|Ab|", "|b-Ax-r|", "pred")

    def __init__(self, A, b, maxit = None, tol = 1e-8, reO = False, x0 = None, prinT = True):
        super().__init__(A, b, maxit, tol, reO, x0, prinT)
        self.stat = {k : [] for k in self._STATS[1:]}
        
    def iterate(self, xk, Ar, rk, Ap, pk, pAAp, rAr, AP):
        alpha = rAr / pAAp
        xk = xk + alpha * pk
        rk = rk - alpha * Ap
        
        if not AP is None:
            rk = rk - AP @ Avec(AP.T, rk)
            
        Ar = Avec(self.A, rk)
        rArp = torch.dot(Ar, rk)
        beta = rArp / rAr   
        pk = rk + beta * pk
        Ap = Ar + beta * Ap
        
        # re-orthogonalization
        if not AP is None:
            #Ap = Ap - AP @ Avec(AP.T, Ap)
            AP = torch.concat([AP, Ap.reshape(-1, 1) / torch.norm(Ap)], dim = 1)
            
        return xk, Ar, rk, Ap, pk, rArp, AP
    
    def solve(self, pred = lambda x : 0):
        rk = self.b - Avec(self.A, self.xk)
        r0, pk = rk.clone(), rk.clone()
        Ar = Avec(self.A, rk)
        rAr = torch.dot(Ar, rk)
        Ar0, Ap = Ar.clone(), Ar.clone()
        norm_Ap = torch.norm(Ap)
        pAAp = norm_Ap ** 2
        norm_r0 = torch.norm(rk)
        norm_rk = norm_r0.clone()
        norm_Ar0 = torch.norm(Ar)
        norm_Ar = norm_Ar0.clone()
        
        # re-orthogonalization
        if self._reO:
            AP = Ap.reshape(-1, 1) / norm_Ap
        else:
            AP = None
            
        # R = rk.reshape(-1, 1) / norm_r0
        # AR = Ar.reshape(-1, 1) / torch.norm(Ar)
        # APx = Ap.reshape(-1, 1) / torch.norm(Ap)
        # RAR = [float(rAr / (norm_Ar * norm_r0))]
        
        # self._X = self.xk.reshape(-1, 1)
        
        self.storePrintStats(self.ite,
                             relnorm := norm_rk / norm_r0, 
                             relAnorm := norm_Ar / norm_Ar0,
                             # abs(torch.dot(r0, Ar)) / (norm_r0 * norm_Ar), 
                             # abs(torch.dot(Ar0, Ap)) / (norm_Ar0 * norm_Ap), 
                             torch.norm((self.b - Avec(self.A, self.xk)) - rk),
                             pred(self.xk))    

        while self.terminate(self.ite, relAnorm):
            
            self.xk, Ar, rk, Ap, pk, rAr, AP = self.iterate(self.xk, Ar, rk, Ap, pk, pAAp, rAr, AP)
            
            # self._X = torch.cat([self._X, self.xk.reshape(-1, 1)], dim = -1)
            
            # update 
            pAAp = torch.dot(Ap, Ap)
            norm_r = torch.norm(rk)
            norm_Ar = torch.norm(Ar)
            self.ite += 1
            
            self.storePrintStats(self.ite,
                                  relnorm := norm_r / norm_r0, 
                                  relAnorm := norm_Ar / norm_Ar0,
                                  # abs(torch.dot(r0, Ar)) / (norm_r0 * norm_Ar), 
                                  # abs(torch.dot(Ar0, Ap)) / (norm_Ar0 * torch.norm(Ap)), 
                                  torch.norm((self.b - Avec(self.A, self.xk)) - rk),
                                  pred(self.xk))
            # R = torch.cat([R, rk.reshape(-1, 1) / torch.norm(rk)], dim = -1)
            # AR = torch.cat([AR, Ar.reshape(-1, 1) / torch.norm(Ar)], dim = -1)
            # APx = torch.cat([APx, Ap.reshape(-1, 1) / torch.norm(Ap)], dim = -1)
            # RAR.append(float(rAr / (norm_r * norm_Ar)))
            # self.storePrintStats(self.ite,
            #                       relnorm := norm_r / norm_r0, 
            #                       relAnorm := norm_Ar / norm_Ar0, 
            #                       torch.norm(torch.diag(torch.tensor(RAR)) - R.T @ AR, p = 2), 
            #                       torch.norm(torch.eye(self.ite + 1) - APx.T @ APx, p = 2), 
            #                       torch.norm((self.b - Avec(self.A, self.xk)) - rk))
                                 
        self.stat["xk"] = self.xk.tolist()
        self.xk = self.xk - torch.dot(self.xk, rk) * rk / (norm_r ** 2)
        self.stat["xk_lifted"] = self.xk.tolist()

class MinimalResidual(Solvers):
    
    #_STATS = ("ite", "|rk|/|b|", "|Ark|/|Ab|", "<b, Ar>", "<Ad0, Adk>", "<v1, vk>", "A(αp)-(αA)p", "|b-Ax-r|")
    _STATS = ("ite", "|rk|/|b|", "|Ark|/|Ab|", "|b-Ax-r|", "pred")

    def __init__(self, A, b, maxit = None, tol = 1e-8, reO = False, x0 = None, prinT = True):
        super().__init__(A, b, maxit, tol, reO, x0, prinT)
        self.stat = {k : [] for k in self._STATS[1:]}
        self._zero = 0
        
    def iterate(self, xkm1, rkm1, vk, vkm1, dkm1, dkm2, betak, ckm1, 
                skm1, delta1k, phikm1, epsk, V):
        pk = Avec(self.A, vk)
        alphak = torch.dot(vk, pk)
        pk = pk - betak * vkm1
        pk = pk - alphak * vk
        betakp1 = torch.norm(pk)
        vkp1 = pk / betakp1
        
        if not V is None:
            print("reOrtho")
            vkp1 = vkp1 - V @ (Avec(V.T, vkp1))
            #for i in range(V.shape[-1]):
            #    vkp1 = vkp1 - V[:, i] * torch.dot(V[:, i], vkp1) 
            V = torch.concat([V, vkp1.reshape(-1, 1)], dim = 1)
            
        delta2k = ckm1 * delta1k + skm1 * alphak
        gamma1k = skm1 * delta1k - ckm1 * alphak
        epskp1 = skm1 * betakp1
        delta1kp1 = -ckm1 * betakp1
        
        gamma2k = torch.sqrt(gamma1k ** 2 + betakp1 ** 2)
            
        if gamma2k > self._zero:
            ck = gamma1k / gamma2k
            sk = betakp1 / gamma2k
            tauk = ck * phikm1
            phik = sk * phikm1
            dk = (vk - delta2k * dkm1 - epsk * dkm2) / gamma2k
            xk = xkm1 + tauk * dk
            if betakp1 > self._zero:
                rk = sk ** 2 * rkm1 - phik * ck * vkp1
            else:
                rk = torch.zeros_like(self.b)
                print("beta is less than zero")
                return xk, rk, vkp1, vk, dk, dkm1, betakp1, ck, sk, delta1kp1, phik, epskp1, gamma2k, tauk, V 
            
        else:
            ck, sk, tauk, phik = 0, 1, 0, phikm1
            rk = rkm1
            xk = xkm1
            print("gamma is less than zero")
            return xk, rk, vkp1, vk, dk, dkm1, betakp1, ck, sk, delta1kp1, phik, epskp1, gamma2k, tauk, V
        
        return xk, rk, vkp1, vk, dk, dkm1, betakp1, ck, sk, delta1kp1, phik, epskp1, gamma2k, tauk, V
    
    def solve(self, pred = lambda x : 0):
        rkm1 = self.b - Avec(self.A, self.xk)
        Ar0 = Avec(self.A, rkm1)
        r0 = rkm1
        betak = torch.norm(rkm1)
        vk = rkm1 / betak
        vkm1, xkm1, dkm1, dkm2 = 4 * [torch.zeros_like(rkm1, dtype = cTYPE, device = cCUDA)]
        ckm1, skm1, phikm1 = -1, 0, betak
        delta1k, epsk = 0, 0
        norm_r0, norm_rk = betak, betak
        Ar = Ar0
        norm_Ar0 = torch.norm(Ar0)
        norm_Ar = norm_Ar0
        
        # re-orthogonalization
        if self._reO:
            V = vk.reshape(-1, 1)
        else:
            V = None
            
        # R = vk.reshape(-1, 1)
        # AD = Ar.reshape(-1, 1) / torch.norm(Ar)
        # AR = Ar.reshape(-1, 1) / torch.norm(Ar)
        # RAR = [float(torch.dot(r0, Ar) / (torch.norm(Ar) * torch.norm(r0)))]
        # self._X = self.xk.reshape(-1, 1)

        self.storePrintStats(self.ite, 
                             relnorm := norm_rk / norm_r0, 
                             relAnorm := norm_Ar / norm_Ar0, 
                             torch.norm((self.b - Avec(self.A, self.xk)) - rkm1),
                             pred(self.xk))
        
        # k - 1 because we want to detect gamma = 0
        while self.terminate(self.ite - 1, relAnorm):
            
            return_vals = self.iterate(self.xk, rkm1, vk, vkm1, dkm1, dkm2, betak, 
                                       ckm1, skm1, delta1k, phikm1, epsk, V)
            
            self.xk, rkm1, vk, vkm1, dkm1, dkm2, betak, ckm1, skm1, delta1k, phikm1, epsk, gamma, tau, V = return_vals                                  
            
            # update 
            self.ite += 1
            Ar = Avec(self.A, rkm1)
            norm_Ar = torch.norm(Ar)
            
            # Adk = Avec(self.A, dkm1)
            # R = torch.cat([R, rkm1.reshape(-1, 1) / torch.norm(rkm1)], dim = -1)
            # AR = torch.cat([AR, Ar.reshape(-1, 1) / torch.norm(Ar)], dim = -1)
            # AD = torch.cat([AD, Adk.reshape(-1, 1) / torch.norm(Adk)], dim = -1)
            # RAR.append(float(torch.dot(Ar, rkm1) / (torch.norm(Ar) * torch.norm(rkm1))))
            # self._X = torch.cat([self._X, self.xk.reshape(-1, 1)], dim = -1)

            self.storePrintStats(self.ite, 
                                  relnorm := torch.norm(rkm1) / norm_r0, 
                                  relAnorm := norm_Ar / norm_Ar0,
                                  # abs(torch.dot(r0, Ar)) / (norm_r0 * norm_Ar), 
                                  # abs(torch.dot(Ar0, Adk)) / (norm_Ar0 * torch.norm(Adk)),
                                  # abs(torch.dot(r0, vk)) / norm_r0, 
                                  torch.norm((self.b - Avec(self.A, self.xk)) - rkm1),
                                  pred(self.xk))
            
            # self.storePrintStats(self.ite, 
            #                       relnorm := torch.norm(rkm1) / norm_r0, 
            #                       relAnorm := norm_Ar / norm_Ar0,  
            #                       torch.norm(torch.diag(torch.tensor(RAR)) - R.T @ AR, p = 2), 
            #                       torch.norm(torch.eye(self.ite) - AD[:, 1:].T @ AD[:, 1:], p = 2),
            #                       #abs(torch.dot(r0, vk)) / norm_r0, 
            #                       torch.norm((self.b - Avec(self.A, self.xk)) - rkm1))
        self.stat["xk"] = self.xk.tolist()
        self.xk = self.xk - torch.dot(self.xk, rkm1) * rkm1 / (torch.norm(rkm1) ** 2)
        self.stat["xk_lifted"] = self.xk.tolist()
            
def Avec(A, x):
    if callable(A):
        return A(x)
    return torch.mv(A, x)        
