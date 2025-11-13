# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 12:15:49 2022

@author: uqalim8
"""
import torch

def backwardArmijo(fun, xk, fk, gk, alpha, pk, beta, rho, maxite, minalpha = 1e-18):
    betagp = torch.dot(gk, pk) * beta
    j = 1
    assert torch.dot(gk, pk) < 0
    while fun(xk + alpha * pk) > fk + alpha * betagp and j < maxite:
        alpha *= rho
        j += 1
        if alpha < minalpha:
            alpha = minalpha
            break
    if j >= maxite:
        print("linesearch max exceeded!")
    return alpha, j

def backForwardArmijo(fun, xk, fk, gk, alpha, pk, beta, rho, maxite):
    betagp = torch.dot(gk, pk) * beta
    assert torch.dot(gk, pk) < 0
    if fun(xk + alpha * pk) > fk + alpha * betagp:
        return backwardArmijo(fun, xk, fk, gk, alpha, pk, beta, rho, maxite)
    else:
        j = 0
        while fun(xk + alpha * pk) <= fk + alpha * betagp and j < maxite:
            alpha /= rho
            j += 1
        if j >= maxite:
            print("linesearch max exceeded!")
        return rho * alpha, j
        
def backForwardArmijo_mod(fun, xk, fk, gk, alpha, pk, beta, rho, maxite):
    betagp = torch.dot(gk, pk) * beta
    assert torch.dot(gk, pk) < 0
    f = fun(xk + alpha * pk)
    if f > fk + alpha * betagp:
        return backwardArmijo(fun, xk, fk, gk, alpha, pk, beta, rho, maxite)
    else:
        j = 1
        while f <= fk + alpha * betagp and j < maxite:
            alpha /= rho
            fn = fun(xk + alpha * pk)
            j += 1
            if f < fn:
                break
            f = fn
        if j >= maxite:
            print("linesearch max exceeded!")
        return rho * alpha, j

def dampedNewtonCGLinesearch(fun, xk, fk, alpha, pk, normpk, beta, rho, maxite):
    const = beta * normpk / 6
    j = 1
    while fun(xk + alpha * pk) > fk - const * (alpha ** 3) and j < maxite:
        alpha *= rho
        j += 1
    if j >= maxite:
        print("linesearch max exceeded!")
    return alpha, j

def dampedNewtonCGbackForwardLS(fun, xk, fk, alpha, pk, normpk, beta, rho, maxite):
    const = beta * normpk / 6
    if fun(xk + alpha * pk) > fk - const * (alpha ** 3):
        return dampedNewtonCGLinesearch(fun, xk, fk, alpha, pk, normpk, beta, rho, maxite)
    else:
        j = 1
        while fun(xk + alpha * pk) <= fk - const * (alpha ** 3) and j < maxite:
            alpha = 2 * alpha
            j += 1
        if j >= maxite:
            print("linesearch max exceeded!")
        return alpha / 2, j

def lineSearchWolfeStrong(objFun, xk, pk, f0, g0, alpha0 = 1, maxalpha = 1e5, c1=1e-4, c2=0.5, linesearchMaxItrs=1000):
    #original c2 = 0.9
    """    
    All vector are column vectors.
    INPUTS:
        objFun: a function handle of both objective function and its gradient
        xk: starting xk
        pk: direction pk
        alpha0: initial step-size
        c1: parameter of Armijo line-search
        c2: parameter of strong Wolfe condition
        linesearchMaxItrs: maximum iteration of line search with strong Wolfe
        
    OUTPUTS:
        alpha: proper step-size
        T: iterations
    """
    itrLS = 0
    itrs2 = 0
    a1 = 0
    #a2 = infun(0, amax)
    a2 = alpha0
    #f0, g0 = objFun(xk)
    fb = f0 # f preview
    while itrLS < linesearchMaxItrs:
        itrLS += 1
        fa, ga = objFun(xk + a2*pk)
        g0p = torch.dot(g0, pk) #g0.T.dot(pk)
        if fa > f0 + a2*c1*g0p or (fa >= fb and itrLS > 0):
            alpha, itrs2 = zoomf(a1, a2, objFun, f0, fa, g0p, xk, pk, c1, c2, itrLS, linesearchMaxItrs)
            break
        
        gap = torch.dot(ga, pk)
        if abs(gap) <= -c2*g0p:
            alpha = a2
            itrs2 = 0
            break

        if gap >= 0:
            alpha, itrs2 = zoomf(a2, a1, objFun, f0, fa, g0p, xk, pk, c1, c2, itrLS, linesearchMaxItrs)
            break

        a2 = a2*2
        fb = fa
        if a2 > maxalpha:
            alpha = maxalpha
            break 

    orcs = itrLS * 2 + itrs2 * 6
    itrLS += itrs2
    if alpha < 1e-18:        
        alpha = 0
    return alpha, itrLS, orcs

def zoomf(a1, a2, objFun, f0, fa, g0p, xk, pk, c1, c2, itrLS, linesearchMaxItrs):
    itrs2 = 0
    while (itrs2 + itrLS) < linesearchMaxItrs :
        #quadratic
        itrs2 += 1
        # lower bound
        fa1, ga1 = objFun(xk + a1*pk)
        ga1p = torch.dot(ga1, pk) #ga1.T.dot(pk)
        # upper bound
        fa2, ga2 = objFun(xk + a2*pk)
        ga2p = torch.dot(ga2, pk) #ga2.T.dot(pk)
        aj = cubicInterp(a1, a2, fa1, fa2, ga1p, ga2p)
        a_mid = (a1 + a2)/2
        if not inside(aj, a1, a_mid):
            aj = a_mid
        faj, gaj = objFun(xk + aj*pk)
        if faj > f0 + aj*c1*g0p or faj >= fa1:
            a2 = aj
        else:
            gajp = torch.dot(gaj, pk) #gaj.T.dot(pk)
            if torch.abs(gajp) <= -c2*g0p:
                break
            if gajp*(a2 - a1) >= 0:
                a2 = a1
            a1 = aj
    return aj, itrs2

def cubicInterp(x1, x2, f1, f2, g1, g2):
    # find minimizer of the Hermite-cubic polynomial interpolating a
    # function of one variable, at the two points x1 and x2, using the
    # function (f1 and f2) and derivative (g1 and g2).
    # Nocedal and Wright Eqn (3.59)
#    print(x1 - x2)
    d1 = g1 + g2 - 3 * (f1 - f2) / (x1 - x2)
#    if d1**2 - g1*g2 <= 0:
#        d2 = 0
#    else:
#        d2 = np.sign(x2 - x1)*np.sqrt(d1**2 - g1*g2)
    sign = x2 - x1
    if sign < 0:
        sign = -1
    elif sign > 0:
        sign = 1
        
    d2 = sign * torch.sqrt(d1**2 - g1*g2)
    xmin = x2 - (x2 - x1)*(g2 + d2 - d1) / (g2 - g1 + 2*d2);
    return xmin
    
def inside(x, a, b):
    # test x \in (a, b) or not
    l = 0
    if not torch.isreal(x):
        return l
    if a <= b:
        if x >= a and x <= b:
            l = 1
    else:
        if x >= b and x <= a:
            l = 1
    return l

def Ax(A, x):
    if callable(A):
        Ax = A(x)
    else:
        Ax = A.dot(x)
    return Ax
