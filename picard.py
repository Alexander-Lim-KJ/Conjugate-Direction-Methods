# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 10:34:39 2024

@author: uqalim8
"""

import torch, numpy
import numpy as np

cTYPE = torch.float64
N = 100
# A = torch.rand(N, N, dtype = torch.float64)
# A = A.T @ A
# eigval, eigvec = torch.linalg.eigh(A)
# b = torch.rand(N, dtype = torch.float64)
# inner_b = torch.abs(eigvec.T @ b)

#e = torch.tensor(np.random.exponential(1, N)) ** 1.5
A = torch.rand(N, dtype = torch.float64)
A[:50] = 0
b = torch.abs(torch.randn(N, dtype = torch.float64))

# fig = plt.figure()
# ax = plt.gca()
# #ax.plot(range(N), torch.sort(e, descending = True)[0], ".")
# ax.plot(range(N), A, '.', c = 'red')
# ax.plot(range(N), b, '.', c = 'green')
# ax.plot(range(N), b / A, '.', c = 'blue')
# ax.set_yscale('log')
# plt.show()

def CG(A, b, rtol = 1e-6, maxit = 100):
    xk = torch.zeros(b.shape[0], dtype = cTYPE)
    rk = b - Ax(A, xk)
    pk = rk.clone()
    Apk = Ax(A, pk)
    norm_b = torch.norm(b)
    xk_correct = xk
    k = 1
    pAp = torch.dot(pk, Apk)
    norm_rk = torch.norm(rk)
    norm_pksq = torch.norm(pk) ** 2

    norm_Ab = torch.norm(Apk)
    R = rk.reshape(-1, 1) / torch.norm(rk)
    pre = 1
    while norm_rk / norm_b > rtol and k < maxit:
        
        alpha = norm_rk ** 2 / pAp
        
        xk = xk + alpha * pk
        
        print("CG:", bool(pre >= torch.dot(xk, b) / (torch.norm(xk) * torch.norm(b))))
        pre = torch.dot(xk, b) / (torch.norm(xk) * torch.norm(b))
        #print(pre)
        
        rk = rk - alpha * Apk
        
        rk = rk - R @ (R.T @ rk)
        R = torch.concat([R, rk.reshape(-1, 1) / torch.norm(rk)], dim = 1)
        
        norm_rkp1 = torch.norm(rk)
        
        beta = (norm_rkp1 / norm_rk) ** 2
        pk = rk + beta * pk
        
        norm_pksq = torch.norm(pk) ** 2
        norm_rk = norm_rkp1
        Apk = Ax(A, pk)
        pAp = torch.dot(pk, Apk)
        k += 1
 
    return xk, k

def CR(A, b, rtol = 1e-6, maxit = 100):
    
    xk = torch.zeros_like(b, dtype = cTYPE)
    pk, rk = b - Ax(A, xk), b - Ax(A, xk)
    Ar = Ax(A, rk)
    Ap = Ar
    rAr = torch.dot(rk, Ar)
    pAAp = torch.dot(Ar, Ar)
    norm_b = torch.norm(b)
    k = 1

    AP = Ap.reshape(-1, 1) / torch.norm(Ap)
    
    pre = 1
    
    while torch.norm(rk) / norm_b > rtol and k < maxit and torch.norm(Ap) > 1e-12:
        alpha = rAr/pAAp
        
        xk = xk + alpha * pk
        
        print(k, float(pre - (torch.dot(xk, b) / (torch.norm(xk) * torch.norm(b)))))
        pre = torch.dot(xk, b) / (torch.norm(xk) * torch.norm(b))
        #print("CR rk / b:", float(torch.norm(rk)/torch.norm(b))**2)
        rk = rk - alpha * Ap
        rk = rk - AP @ (AP.T @ rk)
        
        Ar = Ax(A, rk)
        rArp = torch.dot(rk, Ar)
        beta = rArp / rAr
        pk = rk + beta * pk
        Ap = Ar + beta * Ap
        AP = torch.concat([AP, Ap.reshape(-1, 1) / torch.norm(Ap)], dim = 1)
        rAr = rArp
        pAAp = torch.dot(Ap, Ap)
        k += 1
        
        #print(torch.norm(rk))
        
    return xk, k

def MCR(A, b, rtol = 1e-6, maxit = 100):
    
    xk = torch.zeros_like(b, dtype = cTYPE)
    rk = b
    pk = Ax(A, rk)
    rAr = torch.dot(rk, pk)
    pAAp = torch.dot(pk, pk)
    norm_b = torch.norm(b)
    norm_rk = norm_b
    k = 1
    res = [1]
    R = rk.reshape(-1, 1)
    while norm_rk / norm_b > rtol and maxit > k:
        alpha = rAr / pAAp
        xk = xk + alpha * rk
        rk = rk - alpha * pk
        
        R = torch.concat([R, rk.reshape(-1, 1)], dim = 1)
        pk = Ax(A, rk)
        print(R.T @ pk)

        rAr = torch.dot(rk, pk)
        pAAp = torch.dot(pk, pk)
        norm_rk = torch.norm(rk)
        res.append(norm_rk / norm_b)
        k += 1
    return xk, k, res, None

def lanczos(A, vn, vnm1, beta):
    Avn = Ax(A, vn)
    alpha = torch.dot(vn, Avn)
    vnp1 = Avn - alpha * vn - beta * vnm1
    norm_vnp1 = torch.norm(vnp1)
    return vnp1 / norm_vnp1, alpha, norm_vnp1

def form_T(T, alpha, beta):
    if T is None:
        return torch.tensor([[alpha], [beta]], dtype = cTYPE)
    
    betap = T[-1, -1]
    m, n = T.shape
    T = torch.vstack([T, torch.zeros((1, n), dtype = cTYPE)])
    new_column = torch.zeros((m + 1, 1), dtype = cTYPE)
    new_column[-3:, 0] = torch.tensor([betap, alpha, beta], dtype = cTYPE)
    T = torch.hstack([T, new_column])
    return T

def MINRES(A, b, maxit):
    norm_b = torch.norm(b)
    vn = b / norm_b
    beta = 0
    vnm1 = torch.zeros_like(b)
    T = None
    V = vn.reshape(-1, 1)
    k = 1
    
    pre = 1
    while k < maxit:
        vnp1, alpha, betap1 = lanczos(A, vn, vnm1, beta)
        
        vnp1 = vnp1 - V @ V.T @ vnp1
        #betap1 = torch.norm(vnp1)
        #vnp1 = vnp1 / betap1
        
        T = form_T(T, alpha, betap1)
        V = torch.hstack([V, vnp1.reshape(-1, 1)])
        
        vnm1 = vn
        vn = vnp1
        beta = betap1
        
        Q, R = torch.linalg.qr(T)
        y = torch.linalg.solve(R, Q.T[:,0] * norm_b)
        #print(k, float(pre - y[0] / torch.norm(y)))
        print(y)
        pre = y[0] / torch.norm(y)
        k += 1
    return T, V

def solvetm(A, b, xl, rtl, Delta, xnorm, phil, m_r, iters):    
    xTr = torch.dot(xl, rtl)
    diff = Delta**2 - xnorm**2
    omega = (-xTr + torch.sqrt(xTr**2 + phil**2*diff))/phil**2
    x = xl + omega*rtl   
    m_x = torch.dot(x, -b) + torch.dot(x, Ax(A, x))/2
    iters = iters + 1
    if m_x <= m_r:
        return x, m_x, iters
    else:
        return Delta*rtl/phil, m_r, iters


def MRlanczos(A, v, vp, beta, shift=0):
    Av = Ax(A, v)
    
    if shift == 0:
        pass
    else:
        Av = Av + shift*v
    
    alfa = torch.dot(Av, v)
    vn = Av - v*alfa - vp*beta   
    
    betan = torch.norm(vn)
    vn = vn/betan

    return vn, v, alfa, betan

def qrdecomp(alfa, betan, delta1, sn, cs):
    # delta1n = delta1
    delta2 = cs*delta1 + sn*alfa
    gamma1 = sn*delta1 - cs*alfa
    epsilonn = sn*betan
    delta1n = -cs*betan
    return delta2, epsilonn, delta1n, gamma1

def updates(gamma2, cs, sn, phi, vn, v, wl, wl2, epsilon, delta2, xl, rtl):
    tau = cs*phi
    phi = sn*phi

    w = (v - epsilon*wl2 - delta2*wl)/gamma2
    x = xl + tau*w
    rt = sn**2 * rtl - phi * cs * vn
    return x, w, wl, rt, tau, phi


def myMINRES(A, b, rtol, maxit, shift=0,
             reOrth=False, isZero=1E-15):
    """    
    minres with indefiniteness detector of solving min||b - Ax||
    
    Inputs,
    rtol: inexactness control
    maxit: maximum iteration control.
    shift: perturbations s.t., solving min||b - (A + shift)x||
    detector: bool input, control whether non-positive curvature retur.n or not.
    reOrth: bool input, control whether to apply reorthogonalisation or not.
    isZero: decide a number should be regard as 0 if smaller than isZero.
    
    Output flag explaination,
    Flag = 1, inexactness solution.
    Flag = 2, ||b - Ax|| system inconsistent, given rtol cannot reach, 
        return the best Min-length solution x.
    Flag = 3, non-positive curvature direction detected.
    Flag = 4, the iteration limit was reached.
    """        
    
    r2 = b
    r3 = r2
    beta1 = torch.norm(r2)
        
    ## Initialize
    flag0 = -2
    flag = -2
    iters = 0
    beta = 0
    tau = 0
    cs = -1
    sn = 0
    delta1n = 0
    epsilonn = 0
    gamma2 = 0
#    xnorm = 0
    phi = beta1
    relres = phi / beta1
    betan = beta1
    rnorm = betan
    rt = b 
    norm_b = torch.norm(b)
    vn = r3/betan
    dType = 'Sol'
    # print((A @ b).norm())
    
    x = torch.zeros_like(b)
    w = torch.zeros_like(b)
    wl = torch.zeros_like(b) 
    v = torch.zeros_like(b) 
    
    #b = 0 --> x = 0 skip the main loop
    if beta1 == 0:
        flag = 9        
        
    if reOrth:
        V = vn.reshape(-1, 1)
        
#    normHy2 = tau**2
    while flag == flag0:
        #lanczos
        vn, v, alfa, betan = MRlanczos(A, vn, v, betan, shift)
        iters += 1
        if reOrth:
            if iters == 1:
                vn = vn - (v @ vn) * v
            else:
                vn = vn - torch.mv(V, torch.mv(V.T, vn))
        
        if reOrth:
            V = torch.cat((V, vn.reshape(-1, 1)), axis=1)
                
        ## QR decomposition
        delta1 = delta1n
        epsilon = epsilonn
        delta2, epsilonn, delta1n, gamma1 = qrdecomp(alfa, betan, delta1, sn, cs)
        
        csl = cs
        phil = phi
        rtl = rt
        xl = x  
        nc = csl * gamma1  
        
        ## Check if Lanczos Tridiagonal matrix T is singular
        cs, sn, gamma2 = symGivens(gamma1, betan) 
        
        #if phil*torch.sqrt(gamma2**2 + delta1n**2) < rtol*torch.sqrt(beta1**2-phil**2):
        if torch.norm(rt) / norm_b < rtol:
            flag = 1  ## trustful least square solution
            return xl, iters, rtl, dType
        
        if nc >= -shift: # NPC detection
            flag = 3
            dType = 'NC'
            return xl, iters, rtl, dType
        
        if gamma2 > isZero:
            x, w, wl, rt, tau, phi = updates(gamma2, cs, sn, phi, vn, v, w, 
                                             wl, epsilon, delta2, xl, rtl)
        else:
            ## if gamma1 = betan = 0, Lanczos Tridiagonal matrix T is singular
            ## system inconsitent, b is not in the range of A,
            ## MINRES terminate with phi \neq 0.
            cs = 0
            sn = 1
            gamma2 = 0  
            phi = phil
            rt = rtl
            x = xl
            flag = 2
            print('flag = 2, b is not in the range(A)!')
            maxit += 1
            return x, iters, rt, dType
            
        rnorm = phi
        if iters >= maxit:
            flag = 4  ## exit before maxit
            dType = "MAX"
            # print('Maximun iteration reached', flag, iters)
            return x, iters, rt, dType
    return x, iters, rt, dType

def precond(M, r):
    if callable(M):
        h = M(r)
    else:
        h = torch.mv(torch.pinverse(M), r)
    return h

def symGivens(a, b, device="cpu"):
    """This is used by the function minresQLP.
    
    Arguments:
        a (float): A number.
        b (float): A number.
        device (torch.device): The device tensors will be allocated to.
    """
    if not torch.is_tensor(a):
        a = torch.tensor(float(a), device=device)
    if not torch.is_tensor(b):
        b = torch.tensor(float(b), device=device)
    if b == 0:
        if a == 0:
            c = 1
        else:
            c = torch.sign(a)
        s = 0
        r = torch.abs(a)
    elif a == 0:
        c = 0
        s = torch.sign(b)
        r = torch.abs(b)
    elif torch.abs(b) > torch.abs(a):
        t = a / b
        s = torch.sign(b) / torch.sqrt(1 + t ** 2)
        c = s * t
        r = b / s
    else:
        t = b / a
        c = torch.sign(a) / torch.sqrt(1 + t ** 2)
        s = c * t
        r = a / c
    return c, s, r


def Ax(A,x):
    return A * x #torch.mv(A, x)

xr, kr = CR(A, b, maxit = 30)
print(50 * "=")
xr, kr = MINRES(A, b, maxit = 30)
print(50 * "=")
myMINRES(A, b, 1e-8, maxit = 30)
