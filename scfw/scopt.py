import time
import numpy as np
from scipy.linalg import norm


def estimate_lipschitz(hess_mult_vec, n):
    Lest = 1
    dirr = np.ones(n)
    if Lest == 1:
        # Estimate Lipschitz Constant
        for _ in range(1, 16):
            Dir = hess_mult_vec(dirr)
            dirr = Dir / norm(Dir)
        Hd = hess_mult_vec(dirr)
        dHd = dirr.dot(Hd)
        L = dHd / (dirr.dot(dirr))
    return L


def conj_grad(Grad, Hopr, x, sc_params):
    """
        Computing Newton direction by conjugate gradient method
    """
    r = -Grad - Hopr(x)
    x_new = x
    k = 0
    conj_iter = sc_params['conj_grad_iter']
    eps = sc_params['conj_grad_tol']
    while norm(r) > eps and k < conj_iter:
        p = r
        Hp = Hopr(p)
        alph = r.dot(r) / (p.dot(Hp))
        x_new = x_new + alph * p
        r_new = r - alph * Hp
        bet = r_new.dot(r_new) / r.dot(r)
        p = p + bet * p
        r = r_new
        k = k + 1

    return x_new


def fista(func, Grad_func, prox_func, Hopr, x, sc_params):
    y = x.copy()
    n = len(y)
    Lest = sc_params['Lest']
    fista_type = sc_params['fista_type']
    if Lest == 'estimate':
        L = estimate_lipschitz(Hopr, n)
    elif Lest == 'backtracking':
        L = 1
    x_cur = y.copy()
    f_cur = func(x_cur)
    fista_iter = sc_params['fista_iter']
    tol = sc_params['fista_tol']
    t = 1
    for k in range(1, fista_iter + 1):
        grad_y = Grad_func(y)
        if Lest == 'estimate':
            x_tmp = y - 1 / L * grad_y
            # x_tmps.append(x_tmp)
            z = prox_func(x_tmp, L)
            f_nxt = func(z)
        elif Lest == 'backtracking':
            f_y = func(y)
            beta = 2
            z = y
            L = L / beta
            diff_yz = z - y
            f_z = f_y + grad_y.T.dot(diff_yz) + (L / 2) * norm(diff_yz) ** 2 + 1
            while f_z > f_y + grad_y.T.dot(diff_yz) + (L / 2) * norm(diff_yz) ** 2:
                L = L * beta
                x_tmp = y - 1 / L * Grad_func(y)
                z = prox_func(x_tmp, L)
                f_z = func(z)
                diff_yz = z - y
            f_nxt = func(z)

        if f_nxt > f_cur and fista_type == 'mfista':
            x_nxt = x_cur
            f_nxt = f_cur
        else:
            x_nxt = z
        zdiff = z - x_cur
        ndiff = norm(zdiff)
        if (ndiff < tol) and (k > 1):
            print('Fista err = %3.3e; Subiter = %3d; subproblem converged!\n' % (ndiff, k))
            break
        xdiff = x_nxt - x_cur
        t_nxt = 0.5 * (1 + np.sqrt(1 + 4 * t ** 2))
        y = x_nxt + (t - 1) / t_nxt * xdiff + t / t_nxt * zdiff
        t = t_nxt
        x_cur = x_nxt
        f_cur = f_nxt
    return x_nxt


def scopt(func_x,
          grad_x,
          hess_mult,
          hess_mult_vec,
          Mf,
          nu,
          prox_func,
          x0,
          sc_params,
          eps=0.001,
          print_every=100):

    x = x0
    x_hist = []
    alpha_hist = []
    Q_hist = []
    time_hist = []
    grad_hist = []
    err_hist = []
    int_start = time.time()
    time_hist.append(0)
    max_iter = sc_params['iter_SC']
    def func(xx): return (func_x(xx))[0]
    bPhase2 = False
    use_two_phase = sc_params['use_two_phase']
    for i in range(1, max_iter + 1):

        start = time.time()

        Q, extra_param = func_x(x)
        Grad = grad_x(x, extra_param)
        def Hopr(s): return hess_mult_vec(s, extra_param)
        # compute local Lipschitz constant

        Newton_dir = conj_grad(Grad, Hopr, x, sc_params)
        def grad_func(xx): return Hopr(xx - x - Newton_dir)
        x_nxt = fista(func, grad_func, prox_func, Hopr, x, sc_params)
        diffx = x_nxt - x

        lam_k = np.sqrt(hess_mult(diffx, extra_param))
        beta_k = Mf * norm(diffx)
        # solution value stop-criterion
        nrm_dx = norm(diffx)
        rdiff = nrm_dx / max(1.0, norm(x))
        if use_two_phase and not bPhase2:
            if nu == 2:  # conditions to go to phase 2
                # sigma_k= #still need to add something to compute sigma
                if lam_k * Mf / np.sqrt(sigma_k) < 0.12964:
                    bPhase2 = True
            elif nu < 3:
                d_nu = 1  # too complicated to implement
                if lam_k * Mf / (sigma_k)**((3 - nu) / 2) < min(2 * d_nu / (nu - 2), 1 / 2):
                    bPhase2 = True
            elif nu == 3:
                if lam_k * 2 * Mf < 1:
                    bPhase2 = True
        if not bPhase2:  # if we are not in phase 2
            if beta_k == 0:
                tau_k = 0
            else:
                if nu == 2:
                    tau_k = 1 / beta_k * np.log(1 + beta_k)
                elif nu == 3:
                    d_k = 0.5 * Mf * lam_k
                    tau_k = 1 / (1 + d_k)
                elif nu < 3:
                    d_k = (nu / 2 - 1) * (Mf * lam_k)**(nu - 2) * beta_k**(3 - nu)
                    nu_param = (nu - 2) / (4 - nu)
                    tau_k = (1 - (1 + d_k / nu_param)**(-nu_param)) / d_k
                else:
                    print('The value of nu is not valid')
                    return None
        else:  # if we are in phase 2
            tau_k = 1

        end = time.time()

        alpha_hist.append(tau_k)
        x_hist.append(x)
        Q_hist.append(Q)
        grad_hist.append(Grad)
        err_hist.append(rdiff)
        time_hist.append(end - start)

        x = x + tau_k * diffx

        # Check the stopping criterion.
        if (rdiff <= eps) and i > 1:
            print('Convergence achieved!')
            print('iter = %4d, stepsize = %3.3e, rdiff = %3.3e,value=%g\n' % (i, tau_k, rdiff, Q))
            x_hist.append(x)
            Q_hist.append(Q)
            break

        if (i % print_every == 0) or (i == 1):
            print('iter = %4d, stepsize = %3.3e, rdiff = %3.3e , f = %g\n' % (i, tau_k, rdiff, Q))

        # if mod(iter, options.printst) ~= 0
        #     fprintf('iter = %4d, stepsize = %3.3e, rdiff = %3.3e\n', iter, s, rdiff);
        # end

    int_end = time.time()
    if i >= max_iter:
        x_hist.append(x)
        Q_hist.append(Q)
        print('Exceed the maximum number of iterations')
    print(int_end - int_start)
    return x, alpha_hist, Q_hist, time_hist, grad_hist
