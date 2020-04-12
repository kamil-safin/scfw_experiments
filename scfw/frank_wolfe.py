import time

import numpy as np

from alpha_policies import *


def frank_wolfe(fun_x,
                grad_x,
                grad_beta,
                hess_mult_x,
                extra_fun,
                Mf,
                nu,
                linear_oracle,
                x_0,
                FW_params,
                hess=None,
                sigma_f=None,
                diam_X=None,
                rho=None,
                alpha_policy='standard',
                eps=0.001,
                print_every=100,
                debug_info=False):
    """
        fun_x -- function, outputs function value and extra parameters
        grad_x -- grad value
        grad_beta -- grad velue for convex combination
        hess_mult_x -- Hessian multiplies by x on both sides
        hess -- Hessian computation (for lloo)
        extra_fun -- extra function used to transfer to other methods
        Mf -- parameter for self concordence
        nu -- ?
        linear_oracle -- ?
        x_0 -- starting point (n)
        alpha_policy -- name of alpha changing policy function
        eps -- epsilon
        print_every -- print results every print_every steps
    """
    lower_bound = float("-inf")
    upper_bound = float("inf")
    criterion = 1e10 * eps
    x = x_0

    x_hist = []
    alpha_hist = []
    Gap_hist = []
    Q_hist = []
    time_hist = [0]
    grad_hist = []

    print('********* Algorithm starts *********')
    int_start = time.time()
    max_iter = FW_params['iter_FW']
    line_search_tol = FW_params['line_search_tol']
    L_last=1
    for k in range(1, max_iter + 1):
        start_time = time.time()
        Q, extra_params = fun_x(x)
        if min(extra_params) < -1e-10: #this is a way to know if the gradient is defined on x
            print("gradient is not defined")
            break

        #find optimal
        grad = grad_x(x, extra_params)
        s = linear_oracle(grad)

        delta_x = x - s
        Gap = grad @ delta_x
        lower_bound = max(lower_bound, Q - Gap)
        upper_bound = min(upper_bound, Q)

        if alpha_policy == 'standard':
            alpha = alpha_standard(k)
        elif alpha_policy == 'backtracking':
            extra_param_s = extra_fun(s) #this is a way to know if the gradient is defined on s
            my_func_beta = lambda beta: fun_x(beta*s+(1-beta)*x,beta*extra_param_s*(1-beta)*extra_param_s)[0]
            alpha, L_last = alpha_L_backtrack(my_func_beta,Q,grad,-delta_x,L_last)
        elif alpha_policy == 'line_search':
            extra_param_s = extra_fun(s) #this is a way to know if the gradient is defined on s
            if min(extra_param_s) == 0: #if 0 it is not defines and beta is adjusted
                beta = 0.5
            else:
                beta = 1
            my_grad_beta = lambda beta: grad_beta(x, s, beta, extra_params, extra_param_s)
            alpha = alpha_line_search(my_grad_beta, -delta_x, beta, line_search_tol)
        elif alpha_policy == 'icml':
            hess_mult = hess_mult_x(s - x, extra_params)
            alpha = alpha_icml(Gap, hess_mult, -delta_x, Mf, nu)
        elif alpha_policy == 'lloo':
            if k == 1:
                hess_val = hess(x, extra_params)
                L = max(np.linalg.eigvalsh(hess_val))
                c = 1 + Mf * diam_X * np.sqrt(L) / 2
                r = 1
            hess_func = lambda x: hess(x, extra_params)
            alpha, r, L, c = alpha_lloo(x, hess_func, r, L, c, Mf, sigma_f, diam_X, rho)

        # filling history
        x_hist.append(x)
        alpha_hist.append(alpha)
        Gap_hist.append(Gap)
        Q_hist.append(Q)
        grad_hist.append(grad)
        time_hist.append(time.time() - start_time)

        x = x + alpha * (s - x)

        criterion = min(criterion, norm(x - x_hist[-1]) / max(1, norm(x_hist[-1])))
        #criterion = Gap
        #print(upper_bound)
        #print(lower_bound)
        #criterion=(upper_bound-lower_bound)/abs(lower_bound)
        if criterion <= eps:

            x_hist.append(x)
            Q_hist.append(Q)
            Q, _ = fun_x(x)
            print('Convergence achieved!')
            #print(f'x = {x}')
            #print(f'v = {v}')
            print(f'iter = {k}, stepsize = {alpha}, crit = {criterion}, upper_bound={upper_bound}, lower_bound={lower_bound}')
            return x, alpha_hist, Gap_hist, Q_hist, time_hist, grad_hist


        if k % print_every == 0 or k == 1:
            if not debug_info:
                print(f'iter = {k}, stepsize = {alpha}, criterion = {criterion}, upper_bound={upper_bound}, lower_bound={lower_bound}')
            else:
                print(k)
                print(f'Q = {Q}')
                print(f's = {np.nonzero(s)}')
                print(f'Gap = {Gap}')
                print(f'alpha = {alpha}')
                print(f'criterion = {criterion}')
                #print(f'grad = {grad}')
                print(f'grad norm = {norm(grad)}')
                #print(f'min abs dot = {min(abs(dot_product))}')
                #print(f'x = {x}')
                #x_nz = x[np.nonzero(x)[0]]
                #print(f'x non zero: {list(zip(x_nz, np.nonzero(x)[0]))}\n')

    x_hist.append(x)
    Q_hist.append(Q)
    int_end = time.time()
    print(int_end - int_start)
    return x, alpha_hist, Gap_hist, Q_hist, time_hist, grad_hist
