import numpy as np
import sys, traceback

import numpy as np
import scipy
from scipy.fftpack import idct, dct, dctn, idctn


#
#   For regular matrix dataset
#


def poisson_matr(W, y, lam, x, dot_product=None):
    """
        W -- object matrix (N x n)
        y -- labels (N)
        x -- weights (n)
        lam -- regularization param
    """
    if dot_product is None:
        dot_product = W @ x
    fst_term = np.sum(dot_product)
    snd_term = y.dot(np.log(dot_product))
    return fst_term - snd_term + lam * sum(x), dot_product


def grad_poisson_matr(W, y, lam, x, dot_product=None):
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
        sys.exit('x is not nonnegative')
    if dot_product is None:
        dot_product = np.squeeze(W @ x) # N
    e = np.ones(N)
    mult = (e - (y / dot_product))
    x_term = (W.T @ mult) # n
    return x_term.T + lam * np.ones(n)


def hess_poisson_matr(W, y, x, lam, Btm):
    """
        W -- object matrix (N x n)
        y -- labels (N)
        x -- weights (n)
    """
    denom = 1 / (Btm) # N
    snd_einsum = np.multiply(W, denom.reshape(-1, 1))
    fst_einsum = y.reshape(-1, 1) * snd_einsum
    return np.einsum('ij,ik->jk', fst_einsum, snd_einsum)

def hess_mult_vec_matr(W, y, s, Btm):
    """
        W -- object matrix (N x n)
        y -- labels (N)
        x -- weights (n)
    """
    return (((W @ s) * y)/((Btm) ** 2)).dot(W)


def hess_mult_matr(W, y, x, Btm):
    """
        W -- object matrix (N x n)
        y -- labels (N)
        x -- weights (n)
    """
    num = y.dot(((W @ x) / Btm) ** 2)
    return num


#
#   For pictures
#


def A_opr(im, h):
    return scipy.ndimage.convolve(im.astype(float), h, mode='wrap') # wrap -- circular convolution


def AT_opr(im, h):
    return A_opr(im.astype(float), h)


def A_opr_blur(im, h):
    return A_opr(idctn(im.astype(float), norm='ortho'), h)


def AT_opr_blur(im, h):
    return dctn(AT_opr(im.astype(float), h), norm='ortho')


def poisson_pict(Y, X, h, mu, Ax=None):
    """
        Y -- reconstructed picture (M x N)
        X -- picture (M x N)
        h -- convolution filter
        mu -- additional factor (const)
        Ax -- A_opr(X) (M x N)
    """
    if Ax is None:
        Ax = A_opr_blur(X, h)
    fst_term = np.sum(Ax) + mu
    snd_term = Y.dot(np.log(Ax + mu))
    return fst_term - snd_term, Ax


def grad_pict(Y, X, h, mu, Ax=None):
    """
        Y -- reconstructed picture (M x N)
        X -- picture (M x N)
        h -- convolution filter
        mu -- additional factor (const)
        Ax -- A_opr(X) (M x N)
    """
    if Ax is None:
        Ax = A_opr_blur(X, h)
    denom = Ax + mu
    par = 1 - Y / denom
    return AT_opr_blur(par, h)


def hess_mult_vec_pict(Y, X, h, mu, Ax=None):
    """ H x
        Y -- reconstructed picture (M x N)
        X -- matrix (M x N)
        h -- convolution filter
        mu -- additional factor (const)
        Ax -- A_opr(X) (M x N)
    """
    if Ax is None:
        Ax = A_opr_blur(X, h)
    denom = Ax + mu
    fst = Y / (denom**2)
    snd = A_opr_blur(X, h)
    return AT_opr_blur(fst * snd, h)


def hess_mult_pict(Y, X, h, mu, Ax=None):
    """ x^T H x
        Y -- reconstructed picture (M x N)
        X -- matrix (M x N)
        h -- convolution filter
        mu -- additional factor (const)
        Ax -- A_opr(X) (M x N)
    """
    if Ax is None:
        Ax = A_opr_blur(X, h)
    denom = Ax + mu
    fst = Y / (denom**2)
    snd = Ax**2
    return (fst * snd).sum()


def linear_oracle_full_simplex(grad, M):
    n = len(grad)
    s = np.zeros(n)
    i_max = np.argmax(-grad)
    if grad[i_max] < 0:
        s[i_max] = M # 1 x n
    return s
