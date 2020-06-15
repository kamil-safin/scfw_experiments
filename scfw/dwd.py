import numpy as np


def dwd_val(A, y, c, x, n, d, p, q, denom=None):
    # x is flatten vector of (w, mu, xi)
    w = x[:d]
    mu = x[d]
    xi = x[(d + 1):]
    if denom is None:
        denom = A @ w + mu * y + xi
    val_1 = 1 / denom**q
    val_1 = np.sum(val_1) * (1 / n)
    val_2 = c.dot(xi)
    return val_1 + val_2, denom

def dwd_grad(A, y, c, x, n, d, p, q, denom=None):
    if denom is None:
        w = x[:d]
        mu = x[d]
        xi = x[(d + 1):]
        denom = A @ w + mu * y + xi
    denom = 1 / denom**(q + 1)
    w_grad = (-q / n) * (A.T @ denom)
    mu_grad = (-q / n) * (y.dot(denom))
    xi_grad = (-q / n) * denom + c
    return np.hstack((w_grad, mu_grad, xi_grad))

def hess():
    pass

def hess_mult(A, y, c, s, n, d, p, q, x=None, denom=None):
    if denom is None:
        w = x[:d]
        mu = x[d]
        xi = x[(d + 1):]
        denom = A @ w + mu * y + xi
    denom = 1 / (denom**(q + 2) + 1e-10)
    s_w = s[:d]
    s_mu = s[d]
    s_xi = s[(d + 1):]
    numer = (A @ s_w + s_mu * y + s_xi)**2
    val = q * (q + 1) / n * np.sum(numer * denom)
    return  val

def hess_mult_vec():
    pass


def l1_oracle(grad, n):
    s = np.zeros(n)
    i_max = np.argmax(np.abs(grad))
    s[i_max] = -1 * np.sign(grad[i_max])
    return s

def l2_oracle(grad, n):
    s = -1 * grad
    s = s / np.linalg.norm(s)
    return s

def linear_oracle(grad, d, p, R, u):
    w_grad = grad[:d]
    mu_grad = grad[d]
    xi_grad = grad[(d + 1):]
    w_s = l2_oracle(w_grad, d)
    xi_s = l2_oracle(xi_grad, p)
    mu_s = -1 * u * np.sign(mu_grad)
    return np.hstack((w_s, mu_s, xi_s))

def projection():
    pass