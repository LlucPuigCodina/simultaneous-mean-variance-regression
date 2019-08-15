# -*- coding: utf-8 -*-
"""
Simultaneous Mean-Variance Regression - 11/08/2019

Implementation by Lluc Puig Codina

Original paper: arXiv:1804.01631 [econ.EM];
                referenced in the code as SR-SS 
"""

import numpy as np
import nlopt
from functools import partial


#Mean-Variance Regression Estimation

def Omega(X, S, gamma):
    """Obtain estimated Omega_n (np.array): a diagonal matrix of size n with 
    it's  non-zero entries being the estimated variance of each observation, 
    s(x_i'*gamma). Page 15 in SR-SS.

    Args:
        X (np.array): n by k matrix containing regressands.
        gamma (np.array): column vector of size k containing the guess for the
                            unknown conditionalvariance parameters.
        S (function): Sd function. See SR-SS Assumptions 2 and 3 that it must 
                        fulfil.
    """
    n = X.shape[0]
    return S(X@gamma)*np.identity(n)
    
def beta(Y, X, S, gamma):
    """Obtain estimated Beta (np.array): a column vector of size n containing 
    the unknown conditional mean parameters. Page 18 in SR-SS.

    Args:
        Y (np.array): column vector of size n containing the regressand.
        X (np.array): n by k matrix containing regressors.
        gamma (np.array): column vector of size k containing the guess for the
                            unknown conditional variance parameters.
        S (function): Sd function. See SR-SS Assumptions 2 and 3 that it must
                        fulfil.
    """
    Omega_inv = np.linalg.inv(Omega(X, S, gamma))    
    return np.linalg.inv(X.T@Omega_inv@X)@X.T@Omega_inv@Y
       
def loss(Y, X, S, gamma, grad):
    """Value of the objective function evaluated at the chosen parameters, 
    standard deviation function and data. Page 18 in in SR-SS.

    Args:
        Y (np.array): column vector of size n containing the regressand.
        X (np.array): n by k matrix containing regressors.
        gamma (np.array): column vector of size k containing the guess for the
                            unknown conditionalvariance parameters.
        S (function): Sd function. See SR-SS Assumptions 2 and 3 that it must 
                        fulfil.
    """ 
    n = X.shape[0]
    sd = S(X@gamma)
    beta_ = beta(Y, X, S, gamma)
    err = Y - (X@beta_)
    return sum(((1/2)*np.ones(n))*(np.square(err/sd) + np.ones(n))*sd)
    
def constr(X, S, result, gamma, grad):  
     """Value of the constraint (np.array). All standard deviation estimates
         should be non-zero.
 
     Args:
         X (np.array): n by k matrix containing regressors.
         gamma (np.array):  column vector of size k containing the guess for the 
                             unknown conditional variance parameters.
         S (function): Sd function. See SR-SS Assumptions 2 and 3 that it must 
                         fulfil.
     """
     n = X.shape[0]
     #constraint set up is <= 0, so use negative sign. 
     result = -S(X@gamma) + np.full(n,np.finfo(float).eps) 
     return result

def fit(Y, X, S, gamma0):
    """Optimization via [inser algorithm].

    Args:
        Y (np.array): column vector of size n containing the regressand.
        X (np.array): n by k matrix containing regressors.
        gamma_0 (np.array): column vector of size k containing the initial 
                                guess for the unknown conditional variance 
                                parameters.
        S (function): Sd function. See SR-SS Assumptions 2 and 3 that it must
                        fulfil.
                        
    Return:
        gamma (np.array): column vector of size k containing the estimated
                            unknown conditional variance parameters.
    """
    n, k = X.shape    
    opt_loss = partial(loss, Y, X, S)
    opt_constr = partial(constr, X, S)
    opt = nlopt.opt(nlopt.LN_COBYLA, k)
    opt.set_min_objective(opt_loss)
    opt.add_inequality_mconstraint(opt_constr, np.full(n,0))
    opt.set_xtol_rel(1e-8)
    gamma = opt.optimize(gamma0)
    return gamma   
     