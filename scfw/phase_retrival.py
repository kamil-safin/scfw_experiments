import numpy as np
from portfolio import proj_simplex


def norm(x):
    return np.imag(x)**2 + np.real(x)**2

def phase_val(A, X, y, trace_sum):
    sum_val = 0
    if trace_sum==None:
        trace_sum=[]
        for y_i, A_i in zip(y, A):
            trace = np.trace(A_i.dot(X)).real
            trace_sum.append(trace)
            sum_val += -y_i * np.log(trace) + trace
    else:
        sum_val=-y.dot(np.log(trace_sum))+sum(trace_sum)        
    return sum_val, trace_sum

def phase_gradient(A, X, y, trace_sum):
    sum_val = 0
#    A_T = np.conjugate(np.transpose(A, axes=(0, 2, 1)))
#    return np.sum((-y / np.trace(A.dot(X), axis1=1, axis2=2).real)[:, None, None] * A_T + A_T, axis=0)
    for y_i, A_i,trace_sum_i in zip(y, A,trace_sum):
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

def proj_map(X, c):
    #U, v, UH = np.linalg.svd(X, hermitian=True)
    v, U=np.linalg.eigh(X)
    v_proj = proj_simplex(v)
    #print(sum(v_proj))
    return np.conjugate(U).T @ np.diag(c* v_proj) @ U #U.dot(np.diag(c * v_proj)).dot(UH)
