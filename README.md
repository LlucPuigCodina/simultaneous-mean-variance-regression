
# Simultaneous-Mean-Variance-Regression

Python implementation of [arXiv:1804.01631](https://arxiv.org/abs/1804.01631) **[econ.EM]** by Lluc Puig Codina

CAUTION: The optimization might converge to a wild value depending on the chosen initial value which might violate the non-linear constraint. To see this, set `gamma_0` = [1,0]  in the linear case. The linear case seems to be quite unstable, and requiring much more optimization time than the exponential case.

---

### Simulation Tests


```python
import smvr
import numpy as np
```


```python
# Exponential case
np.random.seed(777)
n = 300

beta_true  = np.array([10, 3])
gamma_true = np.array([3, -1]) 

X = np.array([np.ones(n),
               np.random.uniform(0, 50, n)]).T # n by k matrix

def S(t): return np.exp(t)

Y = np.full(n, np.nan)
for i in range(n):
    Y[i] = np.random.normal(loc = X[i,:]@beta_true, scale = S(X[i]@gamma_true))

## Test of beta function.
## From the true gamma we can directly obtain the true beta parameters.
print('\n\n------------------------------------------------------')
print('Exponential case: Find beta_hat knowing the true gamma')
print('------------------------------------------------------')
print('True beta: ', beta_true)
print('OLS beta: ', (np.linalg.inv(X.T@X)@(X.T)@Y))
print('SMVR beta: ', smvr.beta(Y, X, S, gamma_true))


## Test of fit function
## Full estimation of both beta and gamma
gamma0 = np.array([-1, 5]) #initial guess
gamma_hat = smvr.fit(Y, X, S, gamma0)
beta_hat = smvr.beta(Y, X, S, gamma_hat)
print('\n\n---------------------------------------------')
print('Exponential case: Find beta_hat and gamma_hat')
print('---------------------------------------------')
print('True beta: ', beta_true)
print('True gamma: ', gamma_true) 
print('OLS beta: ', np.linalg.inv(X.T@X)@X.T@Y) 
print('SMVR beta: ', beta_hat)
print('SMVR gamma: ', gamma_hat)
```

    
    
    ------------------------------------------------------
    Exponential case: Find beta_hat knowing the true gamma
    ------------------------------------------------------
    True beta:  [10  3]
    OLS beta:  [10.58268607  2.98178095]
    SMVR beta:  [10.  3.]
    
    
    ---------------------------------------------
    Exponential case: Find beta_hat and gamma_hat
    ---------------------------------------------
    True beta:  [10  3]
    True gamma:  [ 3 -1]
    OLS beta:  [10.58268607  2.98178095]
    SMVR beta:  [10.  3.]
    SMVR gamma:  [ 2.74382346 -0.9053656 ]
    


```python
# Linear case
beta_true  = np.array([2, 0.5])
gamma_true = np.array([0.5 , 0.2]) 

X = np.array([np.ones(n),
               np.random.uniform(-1, 4, n)]).T

def S(t): return t

Y = np.full(n, np.nan)
for i in range(n):
    Y[i] = np.random.normal(loc = X[i,:]@beta_true, scale = S(X[i]@gamma_true))

## Test of beta function.
## From the true gamma we can directly obtain the true beta parameters.
print('\n\n-------------------------------------------------')
print('Linear case: Find beta_hat knowing the true gamma')
print('-------------------------------------------------')
print('True beta: ', beta_true)
print('OLS beta: ', (np.linalg.inv(X.T@X)@(X.T)@Y))
print('SMVR beta: ', smvr.beta(Y, X, S, gamma_true))

## Test of fit function
## Full estimation of both beta and gamma
gamma0 = np.array([1,1]) #initial guess
gamma_hat = smvr.fit(Y, X, S, gamma0)
beta_hat = smvr.beta(Y, X, S, gamma_hat)
print('\n\n----------------------------------------')
print('Linear Case: Find beta_hat and gamma_hat')
print('----------------------------------------')
print('True beta: ', beta_true)
print('True gamma: ', gamma_true) 
print('OLS beta: ', np.linalg.inv(X.T@X)@X.T@Y) 
print('SMVR beta: ', beta_hat)
print('SMVR gamma: ', gamma_hat)
```

    
    
    -------------------------------------------------
    Linear case: Find beta_hat knowing the true gamma
    -------------------------------------------------
    True beta:  [2.  0.5]
    OLS beta:  [1.90964476 0.55846447]
    SMVR beta:  [1.91575771 0.55455647]
    
    
    ----------------------------------------
    Linear Case: Find beta_hat and gamma_hat
    ----------------------------------------
    True beta:  [2.  0.5]
    True gamma:  [0.5 0.2]
    OLS beta:  [1.90964476 0.55846447]
    SMVR beta:  [1.9159158  0.55445541]
    SMVR gamma:  [0.50984555 0.20833171]
    
