# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 12:33:43 2023

@author: uqalim8
"""

import torch.nn as nn
import torch
from constants import cTYPE
from functorch import make_functional

class FFN_NN(nn.Module):
    
    """
    Do not initialise the weights at zeros
    """
    
    def __init__(self, input_dim):
        super().__init__()
        self._nn = nn.Sequential(nn.Linear(input_dim, 128),
                                 nn.Tanh(),
                                 nn.Linear(128, 64),
                                 nn.Tanh(),
                                 nn.Linear(64, 32),
                                 nn.Tanh(),
                                 nn.Linear(32, 16),
                                 nn.Tanh(),
                                 nn.Linear(16, 7),
                                 nn.Softmax(dim = 1))
    
    def forward(self, x):
        return self._nn(x)
    
class FFN_GN(nn.Module):
    
    """
    Do not initialise the weights at zeros
    """
    
    def __init__(self, input_dim):
        super().__init__()
        self._nn = nn.Sequential(nn.Linear(input_dim, 128),
                                 nn.Tanh(),
                                 nn.Linear(128, 64),
                                 nn.Tanh(),
                                 nn.Linear(64, 32),
                                 nn.Tanh(),
                                 nn.Linear(32, 16),
                                 nn.Tanh(),
                                 nn.Linear(16, 1),
                                 nn.Sigmoid())
    
    def forward(self, x):
        return self._nn(x)
    
class NLSGaussNewton:

    def __init__(self, neuralnet, X, Y):
        self._nn, self._X, self._Y = neuralnet, X, Y.reshape(-1, 1)
        self._n = X.shape[0]
        self._d = nn.utils.parameters_to_vector(self._nn.parameters()).shape[0]
    
    def _jacob_holder(self):
        return torch.zeros((self._n, self._d), dtype = self._X.dtype, device = self._X.device)
    
    def _toModule_toFunctional(self, w):
        w = w.detach().requires_grad_(True)
        nn.utils.vector_to_parameters(w, self._nn.parameters())
        return make_functional(self._nn, disable_autograd_tracking = False)
    
    def _jacobian(self, w, full_matrix = True):
        if not full_matrix:
            raise NotImplementedError("Jacobian functional has not been implemented") 
        
        jacob = self._jacob_holder()
        functional, w = self._toModule_toFunctional(w)
        val = functional(w, self._X)
        for i, val_element in enumerate(val):
            g = torch.autograd.grad(val_element, w, retain_graph = True)
            g = nn.utils.parameters_to_vector(g)
            jacob[i] = g
        return jacob
    
    def f(self, w):
        functional, w = self._toModule_toFunctional(w)
        with torch.no_grad():
            return torch.norm(functional(w, self._X) - self._Y) ** 2 / 2
    
    def g(self, w):
        jacob = self._jacobian(w)
        functional, w = self._toModule_toFunctional(w)
        with torch.no_grad():
            return jacob.T @ (functional(w, self._X) - self._Y)
    
    def GNv(self, w):
        jacob = self._jacobian(w)
        with torch.no_grad():
            return lambda v : Avec(jacob.T, (Avec(jacob, v)))
    
    def fg(self, w):
        jacob = self._jacobian(w)
        functional, w = self._toModule_toFunctional(w)
        with torch.no_grad():
            diff = functional(w, self._X) - self._Y
            f = torch.norm(diff) ** 2 / 2
            g = jacob.T @ diff
            return f, g.flatten()
        
    def fgGNv(self, w):
        jacob = self._jacobian(w)
        functional, w = self._toModule_toFunctional(w)
        with torch.no_grad():
            diff = functional(w, self._X) - self._Y
            f = torch.norm(diff) ** 2 / 2
            g = jacob.T @ diff
            return f, g.flatten(), lambda v : Avec(jacob.T, (Avec(jacob, v))), #jacob.T @ jacob 
        
    def __call__(self, w, derv_args):
        
        if derv_args == "0":
            return self.f(w)
        
        elif derv_args == "01":
            return self.fg(w)
        
        elif derv_args == "012":
            return self.fgGNv(w)
        
class Wrapper:
    
    def __init__(self):
        self._funcs = {"0" : self.f, "1" : self.g, "2" : self.Hv,
                       "01" : self.fg, "02" : self.fHv, "12" : self.gHv,
                       "012": self.fgHv}
        
    def f(self, w):
        raise NotImplementedError
    
    def g(self, w):
        raise NotImplementedError
    
    def Hv(self, w):
        raise NotImplementedError
    
    def fg(self, w):
        raise NotImplementedError
    
    def fgHv(self, w):
        raise NotImplementedError
    
    def gHv(self, w):
        raise NotImplementedError
        
    def fHv(self, w):
        raise NotImplementedError

    def __call__(self, x, order):
        return self._funcs[order](x)
    
class nnWrapper(Wrapper):
    
    def __init__(self, func, loss):
        super().__init__()
        self.func, self.loss = func, loss
    
    def _toModule_toFunctional(self, w):
        if w.requires_grad:
            w = w.detach().requires_grad_(True)
        else:
            w = w.requires_grad_(True)
        
        nn.utils.vector_to_parameters(w, self.func.parameters())
        return make_functional(self.func, disable_autograd_tracking = False)
    
    def f(self, x, X, Y):
        device = x.device
        functional, w = self._toModule_toFunctional(x)
        with torch.no_grad():
            return self.loss(functional(w, X.to(device)), Y.to(device))
            
    def g(self, x, X, Y):
        device = x.device
        functional, w = self._toModule_toFunctional(x)
        val = self.loss(functional(w, X.to(device)), Y.to(device))
        g = torch.autograd.grad(val, w)
        g = nn.utils.parameters_to_vector(g)
        return g.detach()
    
    def fg(self, x, X, Y):
        device = x.device
        functional, w = self._toModule_toFunctional(x)
        val = self.loss(functional(w, X.to(device)), Y.to(device))
        g = torch.autograd.grad(val, w)
        g = nn.utils.parameters_to_vector(g)
        return val.detach(), g.detach()
    
    def fgHv(self, x, X, Y):
        device = x.device
        functional, x = self._toModule_toFunctional(x)
        val = self.loss(functional(x, X.to(device)), Y.to(device))
        g = torch.autograd.grad(val, x, create_graph = True)
        g = nn.utils.parameters_to_vector(g)
        Hv = lambda v : nn.utils.parameters_to_vector(
            torch.autograd.grad(g, x, grad_outputs = v, create_graph = False, retain_graph = True)
            ).detach()
        return val.detach(), g.detach(), Hv
    
    def Hv(self, x, X, Y):
        device = x.device
        functional, x = self._toModule_toFunctional(x)
        val = self.loss(functional(x, X.to(device)), Y.to(device))
        g = torch.autograd.grad(val, x, create_graph = True)
        g = nn.utils.parameters_to_vector(g)
        return lambda v : nn.utils.parameters_to_vector(torch.autograd.grad(g, x, v, create_graph = False, retain_graph = True)
                                                        ).detach()
    
    def _HvSingle(self, v, x, X, Y):
        device = x.device
        functional, x = self._toModule_toFunctional(x)
        val = self.loss(functional(x, X.to(device)), Y.to(device))
        g = torch.autograd.grad(val, x, create_graph = True)
        g = nn.utils.parameters_to_vector(g)
        return nn.utils.parameters_to_vector(torch.autograd.grad(g, x, v)).detach()
    
    def __call__(self, x, order, X, Y):
        fgHv = self._funcs[order](x, X, Y)
        x.requires_grad_(False)
        return fgHv
    
def Avec(A, x):
    if A is callable:
        return A(x)
    return torch.mv(A, x)
