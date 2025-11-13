# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 18:26:18 2023

@author: uqalim8
"""
import torch, scipy.sparse, utils, math
from scipy.sparse import coo_matrix
import numpy as np
from constants import cTYPE, cCUDA
from solvers import ConjugateGradient, ConjugateResidual, MinimalResidual
import matplotlib.pyplot as plt
from matplotlib import cm

FOLDER = "./pde2"
TOL = 1e-10
MAXIT = 300
SOLVER = " "
RESOL = 32#128
CENTER = (0.001, 0.001)
WIDTH = 20

class MeshGrid:
    """
    +-----------------...---------------+
    |                                   |
    |                                   |
    |                                   |
    .                                   .
    .                                   .
    .                                   .
    | (0,2) | (1,2) |                   |
    | (0,1) | (1,1) |                   |
    | (0,0) | (1,0) |                   |
    +-----------------...---------------+
    """
    def __init__(self, center, total_width, N):
        assert N >= 1
        self._N = N
        self._center = center
        self._gridSize = total_width / N
        self._numPoints = (N + 1) ** 2
        self._Xa = center[0] - total_width / 2
        self._Xb = center[0] + total_width / 2
        self._Ya = center[1] - total_width / 2
        self._Yb = center[1] + total_width / 2
    
    def ijtoIndex(self, i, j):
        return i + j * (self._N + 1)

    def indextoIJ(self, index):
        return index % (self._N + 1), index // (self._N + 1)
    
    def corners(self, index = True):
        ij = torch.tensor(((0, 0), (self._N, 0), (0, self._N), (self._N, self._N)))
        if not index:
            return ij
        return self.ijtoIndex(ij[:, 0], ij[:, 1])
        
    def sidesAll(self, index = True):
        ij = torch.concat([self.sidesTopBottom(False), 
                           self.sidesLeftRight(False)], dim = 0)
        if not index:
            return ij
        return self.ijtoIndex(ij[:,0], ij[:,1])
    
    def sideTop(self, index = True):
        upto = torch.arange(1, self._N)
        N = torch.tensor([self._N])
        ij = torch.cartesian_prod(upto, N)
        if not index:
            return ij
        return self.ijtoIndex(ij[:,0], ij[:,1])
    
    def sideBottom(self, index = True):
        upto = torch.arange(1, self._N)
        zero = torch.tensor([0])
        ij = torch.cartesian_prod(upto, zero)
        if not index:
            return ij
        return self.ijtoIndex(ij[:,0], ij[:,1])
    
    def sideLeft(self, index = True):
        upto = torch.arange(1, self._N)
        zero = torch.tensor([0])
        ij = torch.cartesian_prod(zero, upto)
        if not index:
            return ij
        return self.ijtoIndex(ij[:,0], ij[:,1])
    
    def sideRight(self, index = True):
        upto = torch.arange(1, self._N)
        N = torch.tensor([self._N])
        ij = torch.cartesian_prod(N, upto)
        if not index:
            return ij
        return self.ijtoIndex(ij[:,0], ij[:,1])
    
    def sidesTopBottom(self, index = True):
        ij = torch.concat([self.sideTop(False),
                           self.sideBottom(False)], dim = 0)
        if not index:
            return ij
        return self.ijtoIndex(ij[:,0], ij[:,1])
    
    def sidesLeftRight(self, index = True):
        ij = torch.concat([self.sideLeft(False), 
                           self.sideRight(False)], dim = 0)
        if not index:
            return ij
        return self.ijtoIndex(ij[:,0], ij[:,1])
    
    def interior(self, index = True):
        upto = torch.arange(1, self._N)
        ij = torch.cartesian_prod(upto, upto)
        if not index:
            return ij
        return self.ijtoIndex(ij[:,0], ij[:,1])
        
    def allXY(self):
        xy = torch.arange(self._N + 1, dtype = cTYPE)
        xy = torch.index_select(torch.cartesian_prod(xy, xy), 1, torch.LongTensor([1,0]))
        return torch.tensor(self._Xa, dtype = cTYPE) + xy[:,0] * self._gridSize, \
            torch.tensor(self._Ya, dtype = cTYPE) + xy[:,1] * self._gridSize
        
    def getXYs(self, ij):
        return self._Xa + ij[:,0] * self._gridSize, self._Ya + ij[:,1] * self._gridSize
    
    def printGrid(self, index = True):
        if not index:
            space = "|" + 13 * " "
            space = (self._N + 1) * space
            space = 3 * "   " + space[:-13] + "\n"
            space = 3 * space
            print(space[:-1])
            for j in range(self._N + 1):
                row = 3 * "--"
                for i in range(self._N + 1):
                    row += " {:^3} ".format(str((i,self._N - j))) + 3 * "--"
                print(row)
                space = "|" + 13 * " "
                space = (self._N + 1) * space
                space = 3 * "   " + space[:-13] + "\n"
                space = 3 * space
                print(space[:-1])
        else:
            space = "|" + 11 * " "
            space = (self._N + 1) * space
            space = 3 * "   " + space[:-11] + "\n"
            space = 3 * space
            print(space[:-1])
            for j in range(self._N + 1):
                row = 3 * "--"
                for i in range(self._N, 0 - 1, -1):
                    row += " ({:^2}) ".format(str(self._numPoints - self.ijtoIndex(i, j) - 1)) + 3 * "--"
                print(row)
                space = "|" + 11 * " "
                space = (self._N + 1) * space
                space = 3 * "   " + space[:-11] + "\n"
                space = 3 * space
                print(space[:-1])
            
class Laplace(MeshGrid):
    
    def __init__(self, center, total_width, N, f, g):
        super().__init__(center, total_width, N)
        self._g, self._f = g, f
        
    def laplaceSparse(self):
        diagonal = torch.zeros(self._numPoints) - 4
        ldiagonal = torch.ones(self._numPoints - 1)
        udiagonal = torch.ones(self._numPoints - self._N - 1)
        
        diagonal[self.corners()] /= 4
        diagonal[self.sidesAll()] /= 2
        
        ldiagonal[-1 + (self._N + 1) * torch.arange(1, self._N + 1)] = 0.
        ldiagonal[torch.arange(0, self._N)] -= 0.5
        ldiagonal[self._N * (self._N + 1) + torch.arange(0, self._N)] -= 0.5
        udiagonal[(self._N + 1) * torch.arange(0, self._N)] -= 0.5
        udiagonal[-1 + (self._N + 1) * torch.arange(1, self._N + 1)] -= 0.5
        
        return -scipy.sparse.diags((udiagonal, ldiagonal, diagonal, ldiagonal, udiagonal), 
                                  (-self._N - 1, -1, 0, 1, self._N + 1))
        
    def laplaceFull(self):
        L = torch.zeros(self._numPoints, self._numPoints)
        for j in range(self._N + 1):
            for i in range(self._N + 1):
                # scaling to maintain symmetry
                if (i == 0 or i == self._N) and (j == 0 or j == self._N):
                    L[self.ijtoIndex(i, j)] = self.laplaceRow(i, j) / 4
                
                elif (i == 0 or i == self._N) or (j == 0 or j == self._N):
                    L[self.ijtoIndex(i, j)] = self.laplaceRow(i, j) / 2
                
                else:
                    L[self.ijtoIndex(i, j)] = self.laplaceRow(i, j)
        return -L.to(cTYPE) #change to positive semi-definite
        
    def b(self):
        values = (self._gridSize ** 2) * (self._f(*self.allXY()))# - self._f_mean)
        values[self.corners()] /= 4
        values[self.sidesAll()] /= 2
        
        vecs = (2 * self.corners(False) / self._N - 1) / math.sqrt(2)
        xy = vecs * self._g(*self.getXYs(self.corners(False)))
        values[self.corners()] -= self._gridSize * (xy[:, 0] + xy[:, 1])# - self._g_mean)
        values[self.sideLeft()] -= self._gridSize * (-self._g(*self.getXYs(self.sideLeft(False)))[:, 0])# - self._g_mean)
        values[self.sideRight()] -= self._gridSize * (self._g(*self.getXYs(self.sideRight(False)))[:, 0])# - self._g_mean)
        values[self.sideBottom()] -= self._gridSize * (-self._g(*self.getXYs(self.sideBottom(False)))[:, 1])# - self._g_mean)
        values[self.sideTop()] -= self._gridSize * (self._g(*self.getXYs(self.sideTop(False)))[:, 1])# - self._g_mean)
        return -values.to(cTYPE)
    
    def corrected_rowth(self, i, j):
        if i == -1:
            return self.corrected_rowth(i + 2, j)
        if j == -1:
            return self.corrected_rowth(i, j + 2)
        if i == self._N + 1:
            return self.corrected_rowth(i - 2, j)
        if j == self._N + 1:
            return self.corrected_rowth(i, j - 2)
        return self.ijtoIndex(i, j)
        
    def laplaceRow(self, i, j):
        """
        5 point stencil Laplace matrix
        """
        row = torch.zeros(self._numPoints)
        row[self.corrected_rowth(i + 1, j)] += 1
        row[self.corrected_rowth(i - 1, j)] += 1
        row[self.corrected_rowth(i, j + 1)] += 1
        row[self.corrected_rowth(i, j - 1)] += 1
        row[self.corrected_rowth(i, j)] -= 4
        return row

def toTorchSparse(scipy_sparse):
    scipy_sparse = coo_matrix(scipy_sparse)
    values = scipy_sparse.data
    indices = np.vstack((scipy_sparse.row, scipy_sparse.col))
    i = torch.LongTensor(indices)
    v = torch.tensor(values, dtype = cTYPE)
    shape = scipy_sparse.shape
    return torch.sparse.FloatTensor(i, v, torch.Size(shape))

def sinsqrtx2py2():
    """
    sin(sqrt(x^2 + y^2))
    """
    sqt = lambda x, y : torch.sqrt(x ** 2 + y ** 2)
    sq = lambda x, y : x ** 2 + y ** 2
    
    def h(x, y):
        return torch.sin(sqt(x, y))
    
    def g(x, y):
        X = x * torch.cos(sqt(x, y)) / sqt(x, y)
        Y = y * torch.cos(sqt(x, y)) / sqt(x, y)
        return torch.concat([X.reshape(-1, 1), Y.reshape(-1, 1)], dim = 1)
    
    def f(x, y):
        fst = (y ** 2) * torch.cos(sqt(x, y)) / (sq(x, y) ** (3/2)) - (x ** 2) * torch.sin(sqt(x, y)) / sq(x, y)
        snd = (x ** 2) * torch.cos(sqt(x, y)) / (sq(x, y) ** (3/2)) - (y ** 2) * torch.sin(sqt(x, y)) / sq(x, y)
        return fst + snd
    
    return h, f, g

def x3py4pxymx2my2():
    """
    x^3 - y^4 + xy - (x^2 + y^2)
    """
    def h(x, y):
        return x ** 3 - y ** 4 + x * y - (x ** 2 + y ** 2)
    
    def f(x, y):
        return 6 * x - 12 * y ** 2  - 4
    
    def g(x, y):
        X = 3 * x ** 2 + y - 2 * x
        Y = - 4 * y ** 3 + x - 2 * y
        return torch.concat([X.reshape(-1, 1), Y.reshape(-1, 1)], dim = 1)
    
    return h, f, g
    
def xexpxyexpy():
    """
    x * y * exp((x + y) / scale)
    """
    scale = 2
    expxy = lambda x, y : torch.exp((x + y) / scale)
    
    def h(x, y):
        return x * y * expxy(x, y)
        
    def f(x, y):
        fst = (x + 2 * scale) * y * expxy(x, y) / (scale ** 2)
        snd = (y + 2 * scale) * x * expxy(x, y) / (scale ** 2)
        return fst + snd
        
    def g(x, y):
        X = (x + scale) * y * expxy(x, y) / scale
        Y = (y + scale) * x * expxy(x, y) / scale
        return torch.concat([X.reshape(-1, 1), Y.reshape(-1, 1)], dim = 1)
        
    return h, f, g

def twohills():
    """
    (exp(-(x - 1)^2 - (y - 1)^2) - exp(-x^2 - y^2))^2
    """
    exp2 = lambda x, y : torch.exp(- (x - 1) ** 2 - (y - 1) ** 2)
    exp1 = lambda x, y : torch.exp(- x ** 2 - y ** 2)
    e = torch.exp(torch.tensor(2.))
    
    def h(x, y):
        return (exp2(x, y) - exp1(x, y)) ** 2
    
    def g(x, y):
        X = -4 * torch.exp(-2 * (x ** 2 + y ** 2 + 2)) * (e - torch.exp(2 * (x + y))) * (e * x - (x - 1) * torch.exp(2 * (x + y)))
        Y = -4 * torch.exp(-2 * (x ** 2 + y ** 2 + 2)) * (e - torch.exp(2 * (x + y))) * (e * y - (y - 1) * torch.exp(2 * (x + y)))
        return torch.concat([X.reshape(-1, 1), Y.reshape(-1, 1)], dim = 1)
    
    def f(x, y):
        fst = 2 * (2 * (x - 1) * exp2(x, y) - 2 * x * exp1(x, y)) ** 2
        snd = 2 * (exp1(x, y) - exp2(x, y))
        trd = -2 * exp1(x, y) + 4 * (x ** 2) * exp1(x, y) - 4 * ((x - 1) ** 2) * exp2(x, y) + 2 * exp2(x, y)
    
        total = fst + snd * trd
    
        fst = 2 * (2 * (y - 1) * exp2(x, y) - 2 * y * exp1(x, y)) ** 2
        snd = 2 * (exp1(x, y) - exp2(x, y))
        trd = -2 * exp1(x, y) + 4 * (y ** 2) * exp1(x, y) - 4 * ((y - 1) ** 2) * exp2(x, y) + 2 * exp2(x, y)
    
        total += fst + snd * trd
        return total
    return h, f, g
    
def expxsiny():

    def h(x, y):
        return torch.exp(x) * torch.sin(y)
        
    def g(x, y):
        X = torch.exp(x) * torch.sin(y)
        Y = torch.exp(x) * torch.cos(y)
        return torch.concat([X.reshape(-1, 1), Y.reshape(-1, 1)], dim = 1)
        
    def f(x, y):
        return torch.zeros_like(x)
    return h, f, g

def x2y2():
    
    def h(x, y):
        return x ** 2 + y ** 2
    
    def g(x, y):
        X = 2 * x
        Y = 2 * y
        return torch.concat([X.reshape(-1, 1), Y.reshape(-1, 1)], dim = 1)
    
    def f(x, y):
        return 4 * torch.ones_like(x)
    
    return h, f, g
    
if "__main__" == __name__:

    h, f, g = x2y2()
    laplace = Laplace(CENTER, WIDTH, RESOL, f, g)
    L = laplace.laplaceSparse()
    b = laplace.b().to(cCUDA)

    import scipy, numpy
    import matplotlib.pyplot as plt
    eigvals, eigvecs = scipy.sparse.linalg.eigsh(L, k = 1088, which = "LM")
    smeval, smevec = scipy.sparse.linalg.eigsh(L, k = 10, which = "SM")
    eigvecs = numpy.concatenate([smevec[:,0].reshape(-1, 1), eigvecs], axis = 1)
    eigvals = numpy.append([1e-7], eigvals)
    sort_b = numpy.abs(eigvecs.T @ b.numpy())
    plt.semilogy(eigvals, label = "eigvals")
    plt.scatter(range(1089), sort_b, label = "b")
    plt.scatter(range(1089), sort_b / eigvals, label = "b / eigvals")
    plt.legend()
    plt.show()
    
    L = toTorchSparse(L).to(cCUDA)
    
    if "CG" in SOLVER:
        CG = ConjugateGradient(L, b, maxit = MAXIT, tol = TOL)
        CG.solve()
        xk = CG.xk.reshape(RESOL + 1, RESOL + 1)
        utils.saveRecords(FOLDER, SOLVER, CG.stat)
        
    elif "CR" in SOLVER:
        CR = ConjugateResidual(L, b, maxit = MAXIT, tol = TOL)
        CR.solve()
        xk = CR.xk.reshape(RESOL + 1, RESOL + 1)
        utils.saveRecords(FOLDER, SOLVER, CR.stat)
            
    elif "MR" in SOLVER:
        MR = MinimalResidual(L, b, maxit = MAXIT, tol = TOL)
        MR.solve()
        xk = MR.xk.reshape(RESOL + 1, RESOL + 1)
        utils.saveRecords(FOLDER, SOLVER, MR.stat)
    
    X = np.arange(CENTER[0] - WIDTH / 2, CENTER[0] + WIDTH / 2, WIDTH / (RESOL+1))
    Y = np.arange(CENTER[0] - WIDTH / 2, CENTER[0] + WIDTH / 2, WIDTH / (RESOL+1))
    X, Y = np.meshgrid(X, Y)
    xk = xk.cpu()
    xk_true = h(*laplace.allXY()).reshape(RESOL + 1, RESOL + 1).cpu()
    
    #xk = xk.reshape(-1)
    #xk_true = xk_true.reshape(-1).double()
    
    #diff = b - torch.mv(L, xk)
    #diff_true = b - torch.mv(L, xk_true)
    
    #print("relative error:", torch.norm(diff) / torch.norm(b))
    #print("true relative error:", torch.norm(diff_true) / torch.norm(b))
    x = plt.contour(X, Y, xk, 10)
    plt.clabel(x, inline=True, fontsize = 10)
    plt.savefig(f"./{FOLDER}/{SOLVER}_contour.png")
    plt.close()
    
    fig, ax = plt.subplots(nrows = 1, ncols = 1, subplot_kw={"projection": "3d"})
    ax.plot_surface(X, Y, xk, cmap = cm.coolwarm)
    plt.savefig(f"./{FOLDER}/{SOLVER}_3d.png")
    plt.close()
    
    x = plt.contour(X, Y, xk_true, 10)
    plt.clabel(x, inline=True, fontsize = 10)
    plt.savefig(f"./{FOLDER}/original_contour.png")
    plt.close()
    
    fig, ax = plt.subplots(nrows = 1, ncols = 1, subplot_kw={"projection": "3d"})
    ax.plot_surface(X, Y, xk, cmap = cm.coolwarm)
    plt.savefig(f"./{FOLDER}/original_3d.png")
    plt.close()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        