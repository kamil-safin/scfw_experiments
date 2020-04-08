import numpy as np
from portfolio import proj_simplex


def norm(x):
    return np.imag(x)**2 + np.real(x)**2

def phase_val(A, X, y):
    sum_val = 0
    for y_i, A_i in zip(y, A):
        trace = np.trace(A_i.dot(X)).real
        sum_val += -y_i * np.log(trace) + trace
    return sum_val, X

def phase_gradient(A, X, y, trace_sum):
    sum_val = 0
#    A_T = np.conjugate(np.transpose(A, axes=(0, 2, 1)))
#    return np.sum((-y / np.trace(A.dot(X), axis1=1, axis2=2).real)[:, None, None] * A_T + A_T, axis=0)
    for y_i, A_i in zip(y, A):
        sum_val += -y_i * np.conjugate(A_i).T / np.trace(A_i.dot(X)).real + np.conjugate(A_i).T
    return sum_val

def hess_mult(A, X, y, S):
    sum_val = 0
    for y_i, A_i in zip(y, A):
        sum_val += y_i * np.trace(A_i.dot(S)).real**2 / np.trace(A_i.dot(X)).real**2
    return sum_val

def hess_mult_vec(A, X, y, S):
    sum_val = 0
    for y_i, A_i in zip(y, A):
        sum_val += y_i * np.conjugate(A_i).T * np.trace(A_i.dot(S)).real / np.trace(A_i.dot(X)).real**2
    return sum_val

def proj_map(X, c):
    U, v, UH = np.linalg.svd(X, hermitian=True)
    v_proj = proj_simplex(v)
    return U.dot(np.diag(c * v_proj)).dot(UH)