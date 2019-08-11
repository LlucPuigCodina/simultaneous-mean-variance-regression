# -*- coding: utf-8 -*-
"""
Simultaneous Mean-Variance Regression - 11/08/2019

Implementation by Lluc Puig Codina

Original paper: arXiv:1804.01631 [econ.EM];
                referenced in the code as SR-SS 
"""

import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import nlopt


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
                        fulfill.
    """
    return(np.diag(S(X @ gamma)))
    
def beta(Y, X, S, gamma):
    """Obtain estimated Beta (np.array): a column vector of size n containing 
    the unknown conditional mean parameters. Page 18 in SR-SS.

    Args:
        Y (np.array): column vector of size n containing the regressand.
        X (np.array): n by k matrix containing regressors.
        gamma (np.array): column vector of size k containing the guess for the
                            unknown conditional variance parameters.
        S (function): Sd function. See SR-SS Assumptions 2 and 3 that it must
                        fulfill.
    """
    Omega_inv = np.linalg.inv(Omega(X, S, gamma))    
    return(np.linalg.inv(X.T@Omega_inv@X)@X.T@Omega_inv@Y)
       
def loss(Y, X, S, gamma):
    """Value of the objective function evaluated at the chosen parameters, 
    standard deviation function and data. Page 18 in in SR-SS.

    Args:
        Y (np.array): column vector of size n containing the regressand.
        X (np.array): n by k matrix containing regressors.
        gamma (np.array): column vector of size k containing the guess for the
                            unknown conditionalvariance parameters.
        S (function): Sd function. See SR-SS Assumptions 2 and 3 that it must 
                        fulfill.
    """ 
    n = np.size(Y)  
    sd = S(X@gamma)
    err = Y - (X@beta(Y, X, S, gamma))                  
    return((1/n)*sum(((1/2)*np.ones(n))*(np.square(err/sd) + np.ones(n))*sd))
    
def constr(X, S, gamma):
    """Value of the constraint (np.array).

    Args:
        X (np.array): n by k matrix containing regressors.
        gamma(np.array):  column vector of size k containing the guess for the 
                            unknown conditional variance parameters.
        S (function): Sd function. See SR-SS Assumptions 2 and 3 that it must 
                        fulfill.
    """
    return( S(X @ gamma) )    

def fit(Y, X, S, gamma0):
    """Optimization via Constrained Optimization BY Linear Approximation.

    Args:
        Y (np.array): column vector of size n containing the regressand.
        X (np.array): n by k matrix containing regressors.
        gamma_0 (np.array): column vector of size k containing the initial 
                                guess for the unknown conditional variance 
                                parameters.
        S (function): Sd function. See SR-SS Assumptions 2 and 3 that it must
                        fulfill.
                        
    Return:
        gamma (np.array): column vector of size k containing the estimated
                            unknown conditional variance parameters.
    """
    n = np.size(X,0)
    k = np.size(X,1)
    opt_loss = partial(loss, Y, X, S)
    opt_constr = partial(constr, X, S)
    opt = nlopt.opt(nlopt.LN_COBYLA, k)
    opt.set_min_objective(opt_loss)
    opt.add_inequality_mconstraint(opt_constr, np.zeros(n))
    gamma = opt.optimize(gamma0)
    return(gamma)
    
    
    
#Toy example 

n = 500
beta_true  = np.array([10,3])
gamma_true = np.array([1,1]) 
X = np.array([np.ones(n),
               np.random.uniform(0,50,n)]).T
## Linear conditional standard deviation, S, function.
def S(i): return(i)

Y = np.full(n, np.nan)
for i in range(n):
    Y[i] = np.random.normal(loc = X[i].T@beta_true, scale = S(X[i]@gamma_true))

fig, ax = plt.subplots()
ax.plot(X[:,1],Y, 'ro')
ax.set(ylabel = 'Y', xlabel = 'X',
         title = 'Toy dataset.  Linear conditional Sd function.')
plt.show()

## Test of beta function.
## From the true gamma we can directly obtain the true beta parameters.
print('True beta: ', beta_true)
print('OLS beta: ', np.linalg.inv(X.T@X)@X.T@Y)
print('MVR beta: ', beta(Y, X, S, gamma_true))

## Test of fit function
## Full estimation of both beta and gamma 
gamma0 = [0,2]
gamma_hat = fit(Y, X, S, gamma0)
beta_hat = beta(Y,X,gamma_hat, S)
print('True beta: ', beta_true) 
print('OLS beta: ', np.linalg.inv(X.T@X)@X.T@Y) 
print('MVR beta: ', beta_hat)
print('True gamma: ', gamma_true) 
print('MVR gamma: ', gamma_hat)
