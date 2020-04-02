import numpy as np
from scipy.linalg import norm


def alpha_standard(k):
    return 2 / (k + 2)

def alpha_icml(Gap, hess_mult_v, d, Mf, nu):
    e = hess_mult_v ** 0.5
    bet = norm(d)
    if nu == 2:
        delta = Mf * bet
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


def alpha_line_search(k, grad_function, delta_x, beta, accuracy):
    lb = grad_function(0).T.dot(delta_x);
    t_lb = 0
    ub = grad_function(beta).T.dot(delta_x);
    t_ub = beta
    t = t_ub
    while t_ub < 1 and ub < 0:
        t_ub = 1 - (1 - t_ub) / 2
        ub = grad_function(t_ub).T.dot(delta_x);
    while t_ub - t_lb > accuracy:
        t = (t_lb + t_ub) / 2
        val = grad_function(t).T.dot(delta_x);
        if val > 0:
            t_ub = t
        else:
            t_lb = t 
    return t
