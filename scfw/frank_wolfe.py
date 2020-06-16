import time
import sys

import numpy as np
import scipy as sc
import scipy.linalg


from .alpha_policies import *


def adjust_beta(extra_param, extra_param_s, x, s, case='general'):
    if extra_param_s.ndim==1:
        if case == 'line_search':
            if min(extra_param_s) == 0: #if 0 it is not defines and beta is adjusted
                beta = 0.5
            else:
                beta = 1
            return beta
        else:
            if min(extra_param_s) < 0: #if 0 it is not defines and beta is adjusted
                indexes = np.where(extra_param_s <= 0)
                beta_max = min(extra_param[indexes]/(extra_param[indexes] - extra_param_s[indexes]))
            else:
                beta_max=1
    else:
        L = np.linalg.cholesky(x)
        invL = np.linalg.inv(L)
        temp = invL @ (s - x)
        min_eig, _ = scipy.linalg.eigh(temp@(invL.transpose()))
        min_eig = min(min_eig)
        if case == 'line_search':
            if min_eig == 0: #if 0 it is not defines and beta is adjusted
                beta = 0.5
            else:
                beta = 1
            print('line_search beta_max: %f' % beta)
            return beta
        else:
            if min_eig < 0:
                #(1-beta)*1+beta min_eig>0 => beta<=1/(1-min_eig)
                beta_max = min(1, 1 / abs(min_eig))
            else:
                beta_max = 1
        print('beta_max: %f' % beta_max)
    return beta_max

def frank_wolfe(fun_x,
                func_beta,
                grad_x,
                grad_beta,
                hess_mult_x,
                extra_fun,
                Mf,
                nu,
                linear_oracle,
                x_0,
                FW_params,
                lloo_oracle=None,
                hess=None,
                alpha_policy='standard',
                eps=0.001,
                print_every=100,
                debug_info=False):
    """
        fun_x -- function, outputs function value and extra parameters
        grad_x -- grad value
        grad_beta -- grad value for convex combination
        hess_mult_x -- Hessian multiplies by x on both sides
        hess -- Hessian computation (for lloo)
        extra_fun -- extra function used to transfer to other methods
        Mf -- parameter for self concordence
        nu -- ?
        linear_oracle -- ?
        x_0 -- starting point (n)
        fw_params -- ?
        alpha_policy -- name of alpha changing policy function
        eps -- epsilon
        print_every -- print results every print_every steps
    """
    lower_bound = float("-inf")
    upper_bound = float("inf")
    real_Gap = upper_bound - lower_bound
    criterion = 1e10 * eps
    x = x_0

    alpha_hist = []
    Gap_hist = []
    f_hist = []
    time_hist = [0]
    print('********* Algorithm starts *********')
    int_start = time.time()
    max_iter = FW_params['iter_FW']
    line_search_tol = FW_params['line_search_tol']
    if alpha_policy == 'lloo' or alpha_policy == 'new_lloo':
        rho=FW_params['rho']
        diam_X=FW_params['diam_X']
        sigma_f=FW_params['sigma_f']

    for k in range(1, max_iter + 1):
        start_time = time.time()
        f, extra_param = fun_x(x)
#        if (extra_param.ndim != 2) and (min(extra_param) < -1e-10): #this is a way to know if the gradient is defined on x
#            print(extra_param)
#            print("gradient is not defined")
#            break

        #find optimal
        grad = grad_x(x, extra_param)
        s = linear_oracle(grad)
        delta_x = x - s
        if x.ndim == 1:
            Gap = grad @ delta_x
        elif x.ndim == 2:
            # matrix case
            Gap = np.trace(np.conjugate(grad).T.dot(delta_x)).real
        else:
            print('Invalid dimension')

        if alpha_policy == 'standard':
            alpha = alpha_standard(k)
            # phase-retrival case
            if x.ndim == 2:
                alpha = 2 / (k + 3)
        elif alpha_policy == 'backtracking':
            if k==1:
                L_last=1
            extra_param_s = extra_fun(s) #this is a way to know if the gradient is defined on s
            #if extra_param_s.ndim==1:
            #    if min(extra_param_s) < 0: #if 0 it is not defines and beta is adjusted
            #        indexes=np.where(extra_param_s<=0)
            #        beta_max=min(extra_param(indexes)/(extra_param(indexes)-extra_param_s(indexes)))
            #    else:
            #        beta_max=1
            #else:
            #    L=np.linalg.cholesky(x)
            #    invL=np.linalg.inv(L)
            #    temp=invL@s
            #    min_eig,_=scipy.linalg.eigh(temp@(invL.transpose()))
            #    min_eig=min(min_eig)
            #    if min_eig<0:
            #        #(1-beta)*1+beta min_eig>0 => beta<=1/(1-min_eig)
            #        beta_max=1/(1-min_eig)-1e-10
            #    else:
            #        beta_max=1
            beta_max = adjust_beta(extra_param, extra_param_s, x, s)
            my_func_beta = lambda beta: func_beta(x,s,beta,extra_param,extra_param_s)[0]
            alpha, L_last = alpha_L_backtrack(my_func_beta, f, grad, -delta_x,L_last,beta_max)
        elif alpha_policy == 'line_search':
            extra_param_s = extra_fun(s) #this is a way to know if the gradient is defined on s
            #if min(extra_param_s) == 0: #if 0 it is not defines and beta is adjusted
            #    beta = 0.5
            #else:
            #    beta = 1
            beta = adjust_beta(extra_param, extra_param_s, x, s)
            my_grad_beta = lambda beta: grad_beta(x, s, beta, extra_param, extra_param_s)          
            alpha = alpha_line_search(my_grad_beta, -delta_x, beta, line_search_tol)
        elif alpha_policy == 'sc':
            hess_mult = hess_mult_x(s - x, extra_param)
            alpha = alpha_sc(Gap, hess_mult, -delta_x, Mf, nu)           
        elif alpha_policy == 'sc_backtracking':
            hess_mult = hess_mult_x(s - x, extra_param)
            if k==1:
                Mf_last=Mf/1e+8
            extra_param_s = extra_fun(s) #this is a way to know if the gradient is defined on s
            #if min(extra_param_s) < 0: #if 0 it is not defines and beta is adjusted
            #    indexes=np.where(extra_param_s<=0)
            #    beta_max=min(extra_param(indexes)/(extra_param(indexes)-extra_param_s(indexes)))
            #else:
            #    beta_max=1
            beta_max = adjust_beta(extra_param, extra_param_s, x, s)
            my_func_beta = lambda beta: func_beta(x,s,beta,extra_param,extra_param_s)[0]
            alpha, Mf_last = alpha_M_backtrack(my_func_beta, f, Gap, hess_mult,-delta_x,Mf_last,beta_max,nu)
        elif alpha_policy == 'sc_hybrid':
            hess_mult = hess_mult_x(s - x, extra_param)
            if k==1:
                Mf_last=Mf
                L_last=1
            extra_param_s = extra_fun(s) #this is a way to know if the gradient is defined on s
            #if min(extra_param_s) < 0: #if 0 it is not defines and beta is adjusted
            #    indexes=np.where(extra_param_s<=0)
            #    beta_max=min(extra_param(indexes)/(extra_param(indexes)-extra_param_s(indexes)))
            #else:
            #    beta_max=1
            beta_max = adjust_beta(extra_param, extra_param_s, x, s)
            my_func_beta = lambda beta: func_beta(x,s,beta,extra_param,extra_param_s)[0]
            alpha,L_last, Mf_last=alpha_sc_hybrid(my_func_beta,f, grad, Gap, hess_mult, -delta_x, L_last, Mf_last, beta_max,nu)
        elif alpha_policy == 'lloo':
            s2 = s.copy()
            delta_x2 = delta_x.copy()
            hess_val = hess(x, extra_param)
            if k==1:
                r_k=1
                alpha=1
            h_k = Gap
            alpha, h_k, r_k, sigma_f = alpha_lloo(k, hess_val, alpha, h_k, r_k, sigma_f, diam_X, Mf, rho)
            s = lloo_oracle(x, r_k, grad,rho)
        elif alpha_policy == 'new_lloo':
            s2 = s.copy()
            delta_x2 = delta_x.copy()
            if k==1 :
                #hess_val = hess(x, extra_param)
                #eigs=(np.linalg.eigvalsh(hess_val))
                #sigma_f=min(eigs)
                #if sigma_f<0: #treat numerical issues
                #    sigma_f=1e-10
                #print(Gap)
                h_k = Gap
                r_k = np.sqrt( 6 * h_k / sigma_f)
            s = lloo_oracle(x, r_k, grad,rho)
            delta_x = x - s
            hess_mult = hess_mult_x(delta_x, extra_param)
            alpha , h_k, r_k = alpha_new_lloo(hess_mult, h_k, r_k, Mf)
        x_nxt = x + alpha * (s - x)
        time_hist.append(time.time() - start_time)
        x_last = x.copy()
        alpha_hist.append(alpha)
        Gap_hist.append(Gap)
        f_hist.append(f)
        x = x_nxt
        if f < upper_bound:
            upper_bound = f
            x_best=x.copy()
        lower_bound = max(lower_bound, f - Gap)
        if (lower_bound - upper_bound) / abs(lower_bound) > 1e-10:
        #    print(lower_bound)
        #    print(upper_bound)
            temp = x + alpha * (s - x)
            #print(x)
            #print(np.linalg.norm(x,1),sum(abs(x)))
            #print(np.linalg.norm(s,1),sum(abs(s)))
            #print(np.linalg.norm(temp,1),sum(abs(temp)))
            #print(f,fun_x(x)[0],fun_x(s)[0],fun_x(temp)[0])
            #print(Gap,grad.T.dot(x-s))
            #print(grad)
            #print(s)
            #print(x-s)
            sys.exit("lower bound bigger than upper bound")
        real_Gap=upper_bound-lower_bound
        # filling history
        # x_hist.append(x)
        #grad_hist.append(grad)

        criterion = min(criterion, norm(x - x_last) / max(1, norm(x_last)))
        #criterion = Gap
        #print(upper_bound)
        #print(lower_bound)
        #criterion=(upper_bound-lower_bound)/abs(lower_bound)
        if criterion <= eps and upper_bound-lower_bound/np.abs(lower_bound)<=eps:

            f_hist.append(f)
            f, _ = fun_x(x_best)
            print('Convergence achieved!')
            #print(f'x = {x}')
            #print(f'v = {v}')
            print(f'iter={k}, stepsize={alpha:.2e}, criterion={criterion:.2e}, upper_bound={upper_bound}, lower_bound={lower_bound}, real_Gap={real_Gap:.2e}, f_val={f}')
            return x_best, alpha_hist, Gap_hist, f_hist, time_hist


        if k % print_every == 0 or k == 1:
            if not debug_info:
                print(f'iter={k}, stepsize={alpha:.2e}, criterion={criterion:.2e}, upper_bound={upper_bound}, lower_bound={lower_bound}, real_Gap={real_Gap:.2e}, f_val={f}')
            else:
                print(k)
                print(f'f = {f}')
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

    #x_hist.append(x)
    f_hist.append(f)
    int_end = time.time()
    print(int_end - int_start)
    return x_best, alpha_hist, Gap_hist, f_hist, time_hist
