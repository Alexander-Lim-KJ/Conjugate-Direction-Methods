# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 12:54:55 2022

@author: uqalim8
"""
import torch
from constants import cCUDA, cTYPE, cZERO

def CG(A, b, rtol = 1e-2, maxit = 100):
    xk = torch.zeros(b.shape[0], dtype = cTYPE, device = cCUDA)
    rk = b - Ax(A, xk)
    pk = rk.clone()
    Apk = Ax(A, pk)
    norm_Ab = torch.norm(Apk)
    norm_b = torch.norm(b)
    xk_correct = xk
    k = 1
    pAp = torch.dot(pk, Apk)
    norm_rk = torch.norm(rk)
    norm_pksq = torch.norm(pk) ** 2
         
    while norm_rk / norm_b > rtol and pAp > cZERO and torch.norm(Apk) / norm_Ab > cZERO and k < maxit:
        
        alpha = norm_rk ** 2 / pAp
        
        xk = xk + alpha * pk
        xk_correct = xk_correct + norm_pksq / ((norm_rk ** 2) * pAp) * pk
        
        rk = rk - alpha * Apk
        norm_rkp1 = torch.norm(rk)
        beta = (norm_rkp1 / norm_rk) ** 2
        pk = rk + beta * pk
        
        norm_pksq = torch.norm(pk) ** 2
        norm_rk = norm_rkp1
        Apk = Ax(A, pk)
        pAp = torch.dot(pk, Apk)
        k += 1
         
    if pAp / (torch.norm(pk) * torch.norm(Apk)) <= cZERO:
        dtype = "NPC"
        if k == 1:
            return rk, k, dtype, 0, 0
        
        if torch.norm(Apk) / torch.norm(pk) <= cZERO:
            dtype = "ZC"
            xk = xk - (norm_rk ** 4) / (norm_pksq) * xk_correct
            return xk, k, dtype, torch.norm(rk - (b-Ax(A,xk))), torch.abs(torch.dot(b, Apk)) / (norm_b * torch.norm(Apk))
        
    elif torch.norm(rk) / norm_b <= rtol:
        dtype = "Sol"
        
    elif k == maxit:
        dtype = "MAX"
 
    return xk, k, dtype, torch.norm(rk - (b-Ax(A,xk))), torch.abs(torch.dot(b, Apk)) / (norm_b * torch.norm(Apk))

def CGSteihaug(H, g, delta, tol, maxite):
    
    z = torch.zeros_like(g)
    # if torch.norm(g) < tol:
    #     return z, "||g||<tol", 1, 0
    
    j = 0
    d, r = -g.clone(), g.clone()
    norm_b = torch.norm(g)
    while j <= maxite:
        Bd = Ax(H, d)
        dBd = torch.dot(d, Bd)
        j += 1
        if dBd <= 0:
            dz = torch.dot(d, z)
            norm_d, norm_z = torch.norm(d), torch.norm(z)
            numerator = - dz + torch.sqrt(dz**2  - norm_d**2 * (norm_z**2 - delta**2))
            tau = numerator / norm_d**2
            p = z + tau * d
            m0_mk = - torch.dot(g, p) - torch.dot(p, Ax(H, p)) / 2
            return p, "NC", m0_mk, j
        
        norm_r = torch.dot(r, r)
        alpha = norm_r / dBd
        zp1 = z + alpha * d
        if torch.norm(zp1) >= delta:
            dz = torch.dot(d, z)
            norm_d, norm_z = torch.norm(d), torch.norm(z)
            numerator = - dz + torch.sqrt(dz**2  - norm_d**2 * (norm_z**2 - delta**2))
            tau = numerator / norm_d**2
            p = z + tau * d
            m0_mk = - torch.dot(g, p) - torch.dot(p, Ax(H, p)) / 2
            return p, "SOL,=", m0_mk, j

        z = zp1
        r = r + alpha * Bd
        if torch.norm(r) < tol:
            p = z
            m0_mk = - torch.dot(g, p) - torch.dot(p, Ax(H, p)) / 2
            return p, "SOL,<", m0_mk, j
        
        norm_rp1 = torch.dot(r, r)
        beta = norm_rp1 / norm_r
        d = -r + beta * d
        norm_r = norm_rp1
    
    p = z
    m0_mk = - torch.dot(g, p) - torch.dot(p, Ax(H, p)) / 2

    return p, "MAX,<", m0_mk, j
        

def CappedCG(H, b, zeta, epsilon, maxiter, M=0):
    g = -b
    y =  torch.zeros_like(g)
    kappa, tzeta, tau, T = para(M, epsilon, zeta)
    tHy = y.clone()
    tHY = y.reshape(-1, 1)
    Y = y.reshape(-1, 1)
    r = g
    p = -g
    tHp = Ax(H, p) + 2*epsilon*p
    j = 1
    ptHp = torch.dot(p, tHp)
    norm_g = torch.norm(g)
    norm_p = norm_g
    rr = torch.dot(r, r)
    dType = 'Sol'
    relres = 1
    if ptHp < epsilon*norm_p**2:
        d = p
        dType = 'NC'
        return d, dType, j, ptHp, 1
    norm_Hp = torch.norm(tHp - 2*epsilon*p)
    if norm_Hp > M*norm_p:
        M = norm_Hp/norm_p
        kappa, tzeta, tau, T = para(M, epsilon, zeta)    
    while j < maxiter:
        alpha = rr/ptHp
        y = y + alpha*p
        #Y = torch.cat((Y, y.reshape(-1, 1)), 1) #record y
        norm_y = torch.norm(y)
        tHy = tHy + alpha*tHp
        #tHY = torch.cat((tHY, tHy.reshape(-1, 1)), 1) # record tHy
        norm_Hy = torch.norm(tHy - 2*epsilon*y)
        r = r + alpha*tHp
        rr_new = torch.dot(r, r) 
        beta = rr_new/rr
        rr = rr_new
        p = -r + beta*p #calculate Hr
        norm_p = torch.norm(p)    
        tHp_new = Ax(H, p) + 2*epsilon*p #the only Hessian-vector product
        j = j + 1
        tHr = beta*tHp - tHp_new #calculate Hr
        tHp = tHp_new
        norm_Hp = torch.norm(tHp - 2*epsilon*p)
        ptHp = torch.dot(p, tHp)  
        if  norm_Hp> M*norm_p:
            M = norm_Hp/norm_p
            kappa, tzeta, tau, T = para(M, epsilon, zeta)
        if norm_Hy > M*norm_y:
            M = norm_Hy/norm_y
            kappa, tzeta, tau, T = para(M, epsilon, zeta)
        norm_r = torch.norm(r)
        relres = norm_r/norm_g
#        print(norm_r/norm_g, tzeta)
        norm_Hr = torch.norm(tHr - 2*epsilon*r)
#        print(norm_r, torch.norm(H(y) + g))
        if  norm_Hr> M*norm_r:
            M = norm_Hr/norm_r         
            kappa, tzeta, tau, T = para(M, epsilon, zeta)
        if torch.dot(y, tHy) < epsilon*norm_y**2:
            d = y
            dType = 'NC'
            # print('y')
            return d, dType, j, torch.dot(y, tHy), relres
        elif norm_r < tzeta*norm_g:
            # print('relres', relres)
            d = y
            return d, dType, j, 0, relres
        elif torch.dot(p, tHp) < epsilon*norm_p**2:
            d = p
            dType = 'NC'
            # print('p')
            return d, dType, j, torch.dot(p, tHp), relres
        elif norm_r > torch.sqrt(T*tau**j)*norm_g:
            print('Uncomment tensors Y, tHY')
            alpha_new = rr/ptHp
            y_new = y + alpha_new*p            
            tHy_new = tHy + alpha_new*tHp
            for i in range(j):
                dy = y_new - Y[:, i]
                dtHy = tHy_new - tHY[:, i]
                if torch.dot(dy, dtHy) < epsilon*torch.norm(dy)**2:
                    d = dy
                    dType = 'NC'
                    print('dy')
                    return d, dType, j, torch.dot(dy, dtHy), relres
    print('Maximum iteration exceeded!')
    return y, dType, j, 0, relres

def para(M, epsilon, zeta):
    # if torch.tensor(M):
    #     M = M.item()
    kappa = (M + 2*epsilon)/epsilon
    tzeta = zeta/3/kappa
    # print('kappa', kappa)
    sqk = torch.sqrt(torch.tensor(float(kappa)))
    tau = sqk/(sqk + 1)
    T = 4*kappa**4/(1 + torch.sqrt(tau))**2
    return kappa, tzeta, tau, T    

def Ax(A, x):
    if callable(A):
        Ax = A(x)
    else:
        Ax = A @ x
    return Ax

# if __name__ == "__main__":
#     A = np.random.rand(100, 100)
#     A = A.T @ A
#     b = A @ np.ones((100, 1))
#     x, k, r = CG(A, b)
#     print(np.linalg.norm(A @ x - b), k, r)
