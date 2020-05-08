import numpy as np
import scipy.linalg as sc


def dot_product(x, y):
    '''
    dot product for vector or matrix
    '''
    if x.ndim == 1:
        return np.dot(x, y)
    if x.ndim == 2:
        # for positive semi-definite matrices
        return np.trace(np.conjugate(x).T.dot(y)).real
    else:
        print('Invalid dimension')
        return None

def norm(x):
    '''
    norm for vector or matrix
    '''
    if x.ndim == 1:
        return sc.norm(x)
    if x.ndim == 2:
        return np.sqrt(dot_product(x, x))
    else:
        print('Invalid dimension')
        return None

def alpha_standard(k):
    return 2 / (k + 2)

def alpha_icml(Gap, hess_mult_v, d, Mf, nu):
    e = hess_mult_v ** 0.5
    beta = norm(d)
    if nu == 2:
        delta = Mf * beta
        t = 1 / delta * np.log(1 + Gap / (delta * e ** 2))
    elif nu == 3:
        delta = 1 / 2 * Mf * e
        t = Gap / (Gap * delta + e ** 2)
    else:
        delta = (nu - 2) / 2 * Mf * (beta ** (3 - nu)) * e ** (nu - 2)
        if nu == 4:
            t = 1 / delta * (1 - np.exp(-delta * Gap / (e ** 2)))
        elif nu < 4 and nu > 2:
            const = (4 - nu) / (nu - 2)
            t = 1 / delta * (1 - (1 + (-delta * Gap * const / (e ** 2))) ** (-1 / const))
    return min(1, t)

def alpha_lloo(k, hess, alpha_k, h_k, r_k, sigma_f, Mf, diam_X,rho):
    eigs = sc.eigh(hess)[0]
    L_k = max(eigs)
    sigma_f=min(min(eigs),sigma_f)
    if sigma_f<0: #numerical issues
        sigma_f = 1e-10
    if k==1:
        r_k = np.sqrt( 6 * h_k / sigma_f)
    else:
        r_k = r_k * np.sqrt(np.exp(-alpha_k /2))
    c_k=1 + Mf*diam_X*np.sqrt(L_k)/2
    alpha_k = sigma_f / (6 * c_k * L_k * rho**2)
    h_k = h_k * np.exp(-alpha_k/2)
    return alpha_k, h_k, r_k, sigma_f

def alpha_new_lloo(hess_mult, h_k, r_k, Mf):
    e=np.sqrt(hess_mult)*Mf/2
    alpha_k = min(h_k*Mf**2 /(4*e**2),1)*(1/(1+e))
    h_k = h_k * np.exp(-alpha_k/2)
    r_k = r_k * np.sqrt(np.exp(-alpha_k /2))
    return alpha_k, h_k, r_k

def alpha_line_search(grad_function, delta_x, beta, accuracy):
    t_lb = 0
    ub = dot_product(grad_function(beta), delta_x)
    #ub = grad_function(beta).T.dot(delta_x)
    t_ub = beta
    t = t_ub
    while t_ub < 1 and ub < 0:
        t_ub = 1 - (1 - t_ub) / 2
        ub = dot_product(grad_function(t_ub), delta_x)
        #ub = grad_function(t_ub).T.dot(delta_x)
    while t_ub - t_lb > accuracy:
        t = (t_lb + t_ub) / 2
        val = dot_product(grad_function(t), delta_x)
        #val = grad_function(t).T.dot(delta_x)
        if val > 0:
            t_ub = t
        else:
            t_lb = t
    return t

def alpha_L_backtrack(func_gamma,fx,gx,delta_x,L_last,t_max):
    tau=2
    nu=0.25
    L=nu*L_last
    qx = dot_product(gx, delta_x)
    qqx=L/2*norm(delta_x)**2
    t=min(-qx/(L*norm(delta_x)**2),t_max)
    while func_gamma(t)>fx+t*qx+t**2*qqx:
        L=tau*L
        qqx=qqx*tau
        t=min(-qx/(2*qqx),t_max)
    return t, L
