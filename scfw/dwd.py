import numpy as np


def dwd_val(A, y, c, x, n, d, p, q, denom=None):
    # x is flatten vector of (w, mu, xi)
    w = x[:d]
    mu = x[d]
    xi = x[(d + 1):]
    if denom is None:
        denom = A @ w + mu * y + xi
    val_1 = 1 / denom**q
    val_1 = np.sum(val_1) * (1 / p)
    val_2 = c.dot(xi)
    return val_1 + val_2, denom

def dwd_grad(A, y, c, x, n, d, p, q, denom=None):
    if denom is None:
        w = x[:d]
        mu = x[d]
        xi = x[(d + 1):]
        denom = A @ w + mu * y + xi
    denom = 1 / denom**(q + 1)
    w_grad = (-q / p) * (A.T @ denom)
    mu_grad = (-q / p) * (y.dot(denom))
    xi_grad = (-q / p) * denom + c
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
    val = q * (q + 1) / p * np.sum(numer * denom)
    return  val

def hess_mult_vec(A, y, c, s, n, d, p, q, x=None, denom=None):
    if denom is None:
        w = x[:d]
        mu = x[d]
        xi = x[(d + 1):]
        denom = A @ w + mu * y + xi
    denom = 1 / denom**(q + 2)
    s_w = s[:d]
    new_vec=A @ s_w + y * s[d] + s[d+1:n] #(A,y,I).T @ ((A,y,I)@ s *denom)
    w_hess = (q * (q + 1)/ p) * (A.T @ (new_vec * denom))
    mu_hess = (q * (q + 1)/ p) * ((y*new_vec).dot(denom))
    xi_hess = (q * (q + 1)/ p) * new_vec*denom
    return np.hstack((w_hess, mu_hess, xi_hess))

def l2_oracle(grad, R=1):
    s = -1 * grad
    s = s * R / np.linalg.norm(s)
    return s

def linear_oracle(grad, d, p, R_w, R, u):
    w_grad = grad[:d]
    mu_grad = grad[d]
    xi_grad = grad[(d + 1):]
    w_s = l2_oracle(w_grad, R=R_w)
    xi_s = l2_oracle(xi_grad, R=R)
    mu_s = -1 * u * np.sign(mu_grad)
    return np.hstack((w_s, mu_s, xi_s))

def l2_projection(x, n, R=1):
    x = x * R / np.linalg.norm(x)
    return x

def projection(x, n, d, p, R_w, R, u):
    w = x[:d]
    mu = x[d]
    xi = x[(d + 1):]
    w_pr = l2_projection(w, d, R=R_w)
    xi_pr = l2_projection(xi, p, R=R)
    mu_pr = u * np.sign(mu)
    return np.hstack((w_pr, mu_pr, xi_pr))
