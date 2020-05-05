import sys

import numpy as np
import scipy.linalg as sc

from scfw.portfolio import proj_simplex


def norm(x):
    return np.imag(x)**2 + np.real(x)**2

def phase_val(A, X, y, trace_sum=None):
    sum_val = 0
    if trace_sum is None:
        trace_sum=[]
        for y_i, A_i in zip(y, A):
            trace = np.trace(A_i.dot(X)).real
            if trace < 0:
                print(trace)
                sys.exit('Error!')
            trace_sum.append(trace)
            sum_val += -y_i * np.log(trace) + trace
    else:
        sum_val=-y.dot(np.log(trace_sum))+sum(trace_sum)
    return sum_val, np.array(trace_sum)

def phase_gradient(A, X, y, trace_sum=None):
    sum_val = 0
#    A_T = np.conjugate(np.transpose(A, axes=(0, 2, 1)))
#    return np.sum((-y / np.trace(A.dot(X), axis1=1, axis2=2).real)[:, None, None] * A_T + A_T, axis=0)
    if trace_sum is None:
        for y_i, A_i in zip(y, A):
            trace_sum_i = np.trace(A_i.dot(X)).real
            sum_val += -y_i * A_i / trace_sum_i + A_i
    else:
        for y_i, A_i, trace_sum_i in zip(y, A,trace_sum):
            #sum_val += -y_i * np.conjugate(A_i).T / trace_sum_i + np.conjugate(A_i).T
            sum_val += -y_i * A_i / trace_sum_i + A_i #A_i is hermitian
    return sum_val

def hess_mult(A, y, S,trace_sum):
    sum_val = 0
    for y_i, A_i, trace_sum_i in zip(y, A,trace_sum):
        sum_val += y_i * np.trace(A_i.dot(S)).real**2 / (trace_sum_i**2)
    return sum_val

def hess_mult_vec(A, y, S, trace_sum):
    sum_val = 0
    for y_i, A_i, trace_sum_i in zip(y, A, trace_sum):
        sum_val += y_i * np.conjugate(A_i).T * np.trace(A_i.dot(S)).real / trace_sum_i**2
    return sum_val

def linear_oracle(grad, c):
    eig_vals, eig_vecs = sc.eigh(grad, eigvals=(0, 0)) # eigh ?
    u_t = eig_vecs[:, 0]
    V_t = c * np.dot(u_t.reshape(-1, 1), np.conj(u_t).reshape(1, -1))
    return V_t

def proj_map(X, c):
    #U, v, UH = np.linalg.svd(X, hermitian=True)
    v, U = sc.eigh(X) #this should not be projection on the simplex but projection on the simplex with radius c
    #check for numerical stability
    n1 = sc.norm(X - U @ np.diag(v) @ np.conjugate(U).T)
    #print(X@U-np.diag(v)*U)
    if n1>1e-10:
        print(n1)
        sys.exit('eigen value decomposition not accurate!')
    v_plus=np.maximum(v,0)
    if sum(v_plus)<=c:
        v_proj=v_plus
    else:
        v_proj = c*proj_simplex(v/c)
    #print(sum(v_proj))
    return U @ np.diag(v_proj) @ np.conjugate(U).T #U.dot(np.diag(c * v_proj)).dot(UH)
