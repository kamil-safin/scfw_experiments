import numpy as np
import scipy.linalg as sc

from scfw.portfolio import proj_simplex


def cov_val(A, X):
    inv_X = np.linalg.inv(X)
    val = -np.sum(np.log(np.linalg.eigvalsh(X))) + np.trace(A.dot(X))
    return val, inv_X

def cov_grad(A, X, inv_X=None):
    if inv_X is None:
        inv_X = np.linalg.inv(X)
    return A - inv_X

def cov_hess(inv_X):
    return np.tensordot(inv_X, inv_X)

#def cov_hess_mult_vec(S, inv_X):
#    hess = cov_hess(inv_X)
#    return hess.dot(S)


def cov_hess_mult_vec(S, param):
    return param.dot(S.dot(param))

def cov_hess_mult(S, inv_X):
    #hess_mult_vec = cov_hess_mult_vec(S, inv_X)
    #return hess_mult_vec.dot(S)
    temp = inv_X.dot(S)
    return np.trace(temp.dot(temp))

def linear_oracle(grad, r=1):
    i_max, j_max = np.unravel_index(np.argmax(np.abs(grad)), grad.shape)
    S = np.zeros(grad.shape)
    if i_max == j_max:
        S[j_max, i_max] = -r * np.sign(grad[i_max, j_max])
    else:
        S[j_max, i_max] = -r/2 * np.sign(grad[i_max, j_max])
        S[i_max, j_max] = S[j_max, i_max]
    return S

def projection(y, r):
    shape = y.shape
    y = y.flatten()
    if np.linalg.norm(y,1) <= r:
        return np.reshape(y, newshape=shape)
    else:
        y_abs = np.abs(y / r)
        P_y_abs = proj_simplex(y_abs)
        P_y = P_y_abs * np.sign(y) * r
    return np.reshape(P_y, newshape=shape)