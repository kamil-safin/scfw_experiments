import numpy as np
import scipy.linalg as sc
from time import time


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

def estimate_lipschitz(hess_mult_vec, n, ndim):
    Lest = 1
    if ndim == 1:
        dirr = np.ones(n)
    elif ndim == 2:
        dirr = np.eye(n)
    if Lest == 1:
        # Estimate Lipschitz Constant
        for _ in range(1, 16):
            Dir = hess_mult_vec(dirr)
            dirr = Dir / norm(Dir)
        Hd = hess_mult_vec(dirr)
        dHd = dot_product(dirr, Hd)
        L = dHd / (dot_product(dirr, dirr))
    return L

def estimate_lipschitz_bb(x, x_old, grad, grad_old, bb_type=2):
    s = x - x_old
    y = grad - grad_old
    if bb_type == 2:
        est = norm(y) / norm(s)
    elif bb_type == 3:
        est = abs(dot_product(y, s)) / norm(s)
    else:
        est = np.sqrt(norm(y)) / norm(s)
    return est

def fista(func, Grad_func, prox_func, Hopr, x, sc_params, print_fista=False):
    y = x.copy()
    n = len(y)
    Lest = sc_params['Lest']
    fista_type = sc_params['fista_type']
    if Lest == 'estimate':
        L = estimate_lipschitz(Hopr, n, ndim=x.ndim)
    elif Lest == 'backtracking':
        L = 1
    x_cur = y.copy()
    f_cur = func(x_cur)
    fista_iter = sc_params['fista_iter']
    tol = sc_params['fista_tol']
    t = 1
    beta = 2
    for k in range(1, fista_iter + 1):
        grad_y = Grad_func(y)
        f_y = func(y)
        if Lest == 'estimate':
            x_tmp = y - 1 / L * grad_y
            # x_tmps.append(x_tmp)
            z = prox_func(x_tmp, L)
            f_z = func(z)
            diff_yz = z - y
        elif Lest == 'backtracking':
            z = y
            L = L / beta
            diff_yz = z - y
            # grad_y.T -> grad_y
            f_z = f_y+1
        while f_z > f_y + dot_product(grad_y, diff_yz) + (L / 2) * norm(diff_yz) ** 2 or f_z > f_y:
            L = L * beta
            x_tmp = y - 1 / L * grad_y
            z = prox_func(x_tmp, L)
            f_z =  func(z)
            diff_yz = z - y
            if L>1e+20:
                #print('L too big')
                z=prox_func(y,L)
                f_z=func(z)
                diff_yz=z-y
                L=L/beta
                break
        f_nxt = f_z
        #print(L,f_y,f_y + dot_product(grad_y, diff_yz) + (L / 2) * norm(diff_yz) ** 2, f_cur,f_nxt)
        if f_nxt > f_cur and fista_type == 'mfista':
            x_nxt = x_cur
            f_nxt = f_cur
        else:
            x_nxt = z
        zdiff = z - x_cur
        ndiff = norm(zdiff)
        if (ndiff < tol) and (k > 1):
            if print_fista:
                print('Fista err = %3.3e; Subiter = %3d; subproblem converged!' % (ndiff, k))
            break
        #if (k % 100 == 0) or k==1:
        #    print('Fista err = %3.3e; Subiter = %3d; \n' % (ndiff, k))
        xdiff = x_nxt - x_cur
        t_nxt = 0.5 * (1 + np.sqrt(1 + 4 * (t ** 2)))
        y = x_nxt + (t - 1) / t_nxt * xdiff + t / t_nxt * (z-x_nxt)
        t = t_nxt
        x_cur = x_nxt
        f_cur = f_nxt
    return x_nxt

def prox_grad(func_x,
          grad_x,
          hess_mult_vec,
          prox_func,
          Mf,
          x0,
          prox_params,
          eps=0.001,
          print_every=100):

    max_iter = prox_params['iter_prox']
    bb_type = prox_params['bb_type']
    backtracking = prox_params['backtracking']
    btk_iters = prox_params['btk_iters']
    n = x0.shape[0]
    ndim = x0.ndim
    x_cur = x0
    x_old = 0
    grad_old = 0
    f_hist, time_hist, alpha_hist = [], [], []
    int_start = time()
    time_hist.append(0)
    for k in range(1, max_iter + 1):
        f, extra_param = func_x(x_cur)
        grad_cur = grad_x(x_cur, extra_param)
        hess_mult_vec_x = lambda x: hess_mult_vec(x, extra_param)
        #Lips_cur = estimate_lipschitz(hess_mult_vec_x, n=n, ndim=ndim)
        Lips_cur = estimate_lipschitz_bb(x_cur, x_old, grad_cur, grad_old, bb_type=bb_type)
        H = Lips_cur * np.eye(n)
        #def Hopr(s): return H.dot(s)
        #def grad_func(xx): return Hopr(xx - x_cur) + grad_cur
        #def Quad(xx): return ((H.dot(xx - x_cur)).dot(xx - x_cur))*0.5 + dot_product(grad_cur, xx - x_cur)
        #x_nxt = fista(Quad, grad_func, prox_func, Hopr, x_cur, prox_params) #we can do this in closed form
        x_nxt = prox_func(x_cur - 1/Lips_cur * grad_cur, Lips_cur)
        diffx = x_nxt - x_cur
        nrm_dx = norm(diffx)
        lam_k = np.sqrt((H.dot(diffx)).dot(diffx))
        beta_k = Mf * norm(diffx)
        if backtracking:
            for _ in range(btk_iters):
                if Lips_cur <= ((lam_k * lam_k) / (nrm_dx * nrm_dx)):
                    break
                else:
                    Lips_cur = Lips_cur / 2
                    x_nxt = prox_func(x_cur - 1/Lips_cur * grad_cur, Lips_cur)
            
        diffx = x_nxt - x_cur
        nrm_dx = norm(diffx)
        lam_k = np.sqrt((H.dot(diffx)).dot(diffx))
        beta_k = Mf * norm(diffx)
        alpha = min(beta_k / (lam_k * (lam_k + beta_k)), 1.)
        alpha_hist.append(alpha)
        x_old = x_cur
        grad_old = grad_cur
        x_cur  = x_cur + alpha * diffx
        time_hist.append(time() - int_start)
        rdiff = nrm_dx / max(1.0, norm(x_cur))
        f_hist.append(f)

        if (rdiff <= eps) and (k > 1):
            print('Convergence achieved!')
            print('iter = %4d, stepsize = %3.3e, rdiff = %3.3e,value=%g' % (k, alpha, rdiff, f))
            break

        if (k % print_every == 0) or (k == 1):
            print('iter = %4d, stepsize = %3.3e, rdiff = %3.3e , f = %g' % (k, alpha, rdiff, f))
    f_hist.append(f)
    int_end = time()
    print(int_end - int_start)
    return x_nxt, alpha_hist, f_hist, time_hist
