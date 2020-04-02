import numpy as np


def poisson(W, y, lam, x):
    """
        W -- object matrix (N x n)
        y -- labels (N)
        x -- weights (n)
        lam -- regularization param
    """ 
    dot_product = W @ x 
    fst_term = np.sum(dot_product)
    snd_term = y.dot(np.log(dot_product))
    return fst_term - snd_term + lam * sum(x), dot_product


def grad_poisson(W, y, lam, x, dot_product=None):
    """
        W -- object matrix (N x n)
        y -- labels (N)
        x -- weights (n)
        lam -- regularization param
        btm -- W @ x (N)
    """
    n = len(x)
    N = len(y)
    if min(x) < 0:
        print("fail")
    if dot_product is None:
        dot_product = np.squeeze(W @ x) # N
    e = np.ones(N)    
    mult = (e - (y / dot_product))
    x_term = (W.T @ mult) # n
    return x_term.T + lam * np.ones(n)   


def hess_poisson(W, y, x, lam, Btm):
    """
        W -- object matrix (N x n)
        y -- labels (N)
        x -- weights (n)
    """
    denom = 1 / (Btm) # N
    snd_einsum = np.multiply(W, denom.reshape(-1, 1))
    fst_einsum = y.reshape(-1, 1) * snd_einsum
    return np.einsum('ij,ik->jk', fst_einsum, snd_einsum)

def hess_mult_vec(W, y, s, Btm):
    """
        W -- object matrix (N x n)
        y -- labels (N)
        x -- weights (n)
    """
    return (((W @ s) * y)/((Btm) ** 2)).dot(W)


def hess_mult(W, y, x, Btm):
    """
        W -- object matrix (N x n)
        y -- labels (N)
        x -- weights (n)
    """
    num = y.dot(((W @ x) / Btm) ** 2)
    return num

def linear_oracle_full_simplex(grad, M):
    n = len(grad)
    s = np.zeros(n)
    i_max = np.argmax(-grad)
    if grad[i_max] < 0:
        s[i_max] = M # 1 x n
    return s    
