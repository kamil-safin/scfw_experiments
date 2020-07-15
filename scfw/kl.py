import sys
import numpy as np
from numpy import matlib

        
def val(W, y, lam, x, dot_product=None):
    t = x[-1]
    x = x[:-1]
    if dot_product is None:
        dot_product = W.dot(x)
    first_term = dot_product * np.log(dot_product / y)
    return np.sum(first_term - dot_product) + lam * t, dot_product

def grad(W, y, lam, x, dot_product=None):
    if dot_product is None:
        dot_product = W.dot(x)
    if min(x) < 0:
        sys.exit('x is not nonnegative')
    t = x[-1]
    x = x[:-1]
    first_part = W.T.dot(np.log(dot_product / y))
    return np.hstack((first_part, lam))

def hess_mult(W, y, lam, s, dot_product):
    s = s[:-1]
    num = W.dot(s) ** 2
    return np.sum(num / dot_product)

def hess_mult_vec(W, y, lam, s, dot_product):
    hess = 0
    for w, denom in zip(W, dot_product):
        hess += np.tensordot(w, w, axes=0) / denom
    return np.hstack((hess.dot(s[:-1]), 0))

def linear_oracle(grad, x):
    t = x[-1]
    s = np.zeros(len(grad)) + 1e-10
    i_max = np.argmax(-grad[:-1])
    if grad[i_max] < 0:
        s[i_max] = t # 1 x n
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

def projection(y):
    t = y[-1]
    y = y[:-1]
    P_y = proj_simplex(y)
    P_y = P_y * abs(t)
    return np.hstack((P_y, np.max((t, 1e-10))))