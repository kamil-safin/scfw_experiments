import numpy as np


def portfolio(R, x, Rx=None):
    """
        R -- object matrix (N x n)
        x -- weights (n)
    """
    if Rx is None:
        Rx = R @ x
    return -np.sum(np.log(Rx)), Rx

def grad_portfolio(R, x, Rx):
    """
        R -- object matrix (N x n)
        x -- weights (n)
    """
    if Rx is None:
        Rx = R.dot(x)
    return -R.T.dot(1 / Rx)

def hess_portfolio(R, d, Rx):
    """
        R -- object matrix (N x n)
        x -- weights (n)
    """
    dtype = R.dtype
    Z = R / Rx.reshape(-1, 1)
    return np.einsum('ij,ik->jk', Z, Z, dtype=dtype)

def hess_mult_vec(R, d, Rx):
    """
        W -- object matrix (N x n)
        y -- labels (N)
        x -- weights (n)
    """
    Rd = R @ d
    return R.T.dot(Rd / Rx ** 2)

def hess_mult_portfolio(R, d, Rx):
    """
        R -- object matrix (N x n)
        x -- weights (n)
    """
    Rd = (R @ d)
    Z = (Rd / Rx) ** 2
    return np.sum(Z)

def linear_oracle_simplex(grad):
    grad_min = np.min(grad)
    s = np.array([el == grad_min for el in grad])
    s = s / sum(s)
    return s

def proj_simplex(y):
    ind = np.argsort(y)
    sum_y = sum(y)
    origin_y = sum_y
    n = len(y)
    Py = y
    for i in range(n):
        t = (sum_y - 1) / (n - i)
        if (origin_y > 1 and t < 0): #for numerical errors
            sum_y = sum(y[ind[i : n - 1]])
            t = (sum_y - 1) / (n - i)
        if i > 0:
            if t <= y[ind[i]] and t >= y[ind[i - 1]]:
                break
        elif t <= y[ind[i]]:
            break
        sum_y -= y[ind[i]]
        Py[ind[i]] = 0
    Py = np.maximum(y - t, np.zeros(n))
    return Py
