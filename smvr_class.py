# -*- coding: utf-8 -*-
"""
Simultaneous Mean Variance Regression

Implementation by Lluc Puig Codina

Original paper: arXiv:1804.01631 [econ.EM];
                referenced in the code as SR-SS 
"""
import numpy as np
import nlopt
from functools import partial

class smvr:
    """
    Y (np.array): column vector of size n containing the regressand.
    
    X (np.array): n by k matrix containing regressors.
    
    S (function): Sd function. See SR-SS Assumptions 2 and 3 that it must
                        fulfil.
    """            
    
    def __init__(self, y, X, S):
        
        self.y = y
        self.X = X
        self.S = S        
        self.n, self.k = X.shape
        self.gamma = None

        if Y.shape != (self.n,):
            raise ValueError('Y size is not n by 1 or X size is not n by k.')
        elif hasattr(S, '__call__'):
            raise ValueError('S is not a function')
        
    def Omega(self, gamma):
        return self.S(self.X@gamma)*np.identity(self.n)
    
    def beta(self, gamma):
        y = self.y
        X = self.X
        Omega_inv = np.linalg.inv(self.Omega(gamma)) 
        return np.linalg.inv(X.T@Omega_inv@X)@X.T@Omega_inv@y
    
    def loss(self, gamma, grad):
        err = self.y - (self.X@self.beta(gamma))
        A = np.full(self.n,(1/2))*(np.square(err)/self.S(self.X@gamma))
        B = np.full(self.n,(1/2))*self.S(self.X@gamma)
        return sum(A) + sum(B)
    
    def constr(self, result, gamma, grad):
        result = -self.S(self.X@gamma) + np.full(self.n,np.finfo(float).eps) 
        return result
        
    def fit(self, gamma0):
        n, k = self.n, self.k
        
        if gamma0.shape != (k,):
            raise ValueError('Initial guess of gamma is not of size k by 1')
        
        opt = nlopt.opt(nlopt.LN_COBYLA, k)
        opt.set_min_objective(loss)
        opt.add_inequality_mconstraint(constr, np.full(n,0))
        opt.set_xtol_rel(1e-8)
        self.gamma = opt.optimize(gamma0)
        return None
        