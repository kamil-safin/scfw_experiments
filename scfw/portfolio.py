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
    if Rx is None:
        Rx = R.dot(d)
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
    Py = y.copy()
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

def llo_oracle(x, r, grad, rho):
    n = len(x)
    d = rho * r
    sum_threshold = min(d / 2, 1)
    min_index = np.argmin(grad)
    p_pos = np.zeros(n)
    p_pos[min_index] = sum_threshold
    p_neg = np.zeros(n)
    sorted_indexes = (-grad).argsort()
    k = 0
    tmp_sum = 0
    for k in range(len(sorted_indexes)):
        tmp_sum += x[sorted_indexes[k]]
        if tmp_sum >= sum_threshold:
            break
    for j in range(k):
        index = sorted_indexes[j]
        p_neg[index] = x[index]
    p_neg[sorted_indexes[k]] = sum_threshold - (tmp_sum - x[sorted_indexes[k]])
    return x + p_pos - p_neg
