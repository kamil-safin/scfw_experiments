import os
import time
import pickle
from multiprocessing import Pool

import numpy as np
from sklearn.datasets import load_svmlight_file
from scipy.linalg import norm

import scfw.log_reg as lr
from scfw.frank_wolfe import frank_wolfe


def run_fw(problem_name):
    out_dir = 'results'
    results_file = os.path.join(out_dir, problem_name)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    results = {problem_name: {}}
    Phi, y = load_svmlight_file(os.path.join('data', problem_name))

    # fix classes
    if max(y) == 2:
        y = 2 * y - 3

    N, n = Phi.shape

    # normalize
    for i, row in enumerate(Phi):
        if np.sum(row.multiply(row)) != 0:
            Phi[i] = row.multiply(1 / np.sqrt(np.sum(row.multiply(row))))

    gamma = 1 / 2 * np.sqrt(N)
    Mf = 1/gamma*np.max(np.sqrt(np.sum(Phi.multiply(Phi),axis=1)))
    nu = 3
    mu = 0
    
    
    #running parameters
    x0 = np.zeros(n)
    r = n*0.05
    terminate_tol = 1e-20
    
    #parameters for FW
    FW_params={
        'iter_FW':50000,
        'line_search_tol':1e-10,
        'rho':np.sqrt(n), #parameters for ll00
        'diam_X':2,
        'sigma_f':1,                   
    }
    
    
    sc_params={
        #parameters for SCOPT
        'iter_SC': 1000,
        'Lest': 'backtracking',#,'estimate', #estimate L
        'use_two_phase':False,
        #FISTA parameters
        'fista_type': 'mfista',
        'fista_tol': 1e-5,
        'fista_iter': 1000,
        #Conjugate Gradient Parameters
        'conj_grad_tol':1e-5,
        'conj_grad_iter':1000,
    }

    func_x = lambda x: lr.log_reg(Phi, y, x, mu, gamma)
    func_beta = lambda x, s, beta, exp_product, exp_product_s:lr.log_reg(Phi, y, (1 - beta) * x + beta * s, mu,gamma,np.exp(np.log(exp_product)*(1-beta)+np.log(exp_product_s)*beta))
    grad_x = lambda x, exp_product: lr.grad_log_reg(Phi, y, x,  mu, gamma, exp_product)
    grad_beta = lambda x, s, beta, exp_product, exp_product_s: lr.grad_log_reg(Phi, y, (1 - beta) * x + beta * s, mu, gamma, (1 - beta) * exp_product + beta * exp_product_s)
    hess_x = lambda s, exp_product: lr.hess(Phi, y, mu, gamma, exp_product,s)
    hess_mult_x = lambda s, exp_product: lr.hess_mult_log_reg(Phi, y, mu, gamma, exp_product,s)
    hess_mult_vec_x = lambda s, exp_product: lr.hess_mult_vec(Phi, y,mu, gamma, exp_product,s)
    extra_func = lambda x: np.exp(-y*(Phi @ x+mu))
    linear_oracle = lambda grad: lr.linear_oracle_l1(grad, r)
    # llo_oracle = lambda x, r, grad, rho: pr.llo_oracle(x, r, grad,rho)
    prox_func = lambda x, L: lr.projection_l1(x,r)

    run_alpha_policies = ["backtracking", "standard", "line_search", "icml"]


    for policy in run_alpha_policies:
        print(f'{policy} for {problem_name} started!')
        x, alpha_hist, Gap_hist, Q_hist, time_hist = frank_wolfe(func_x,
                           func_beta,                                      
                           grad_x,
                           grad_beta,
                           hess_mult_x,
                           extra_func,
                           Mf,
                           nu,
                           linear_oracle,                                                    
                           x0,
                           FW_params,
                           hess=None, 
                           lloo_oracle=None,                                                 
                           alpha_policy=policy,                                                    
                           eps=terminate_tol, 
                           print_every=50000, 
                           debug_info=False)
  
        results[problem_name][policy] = {
            'x': x,
            'alpha_hist': alpha_hist,
            'Gap_hist': Gap_hist,
            'Q_hist': Q_hist,
            'time_hist': time_hist,
        }

        with open(results_file, 'wb') as f:
            pickle.dump(results, f)    

        print(f'{policy} for {problem_name} finished!')
    return results


if __name__ == '__main__':
    start_time = time.time()

    problems = ['a4a','w4a','a1a','a2a','a3a','a5a','a6a','a7a','a8a','a9a','w1a','w2a','w3a','w5a','w6a','w7a','w8a']
    pool = Pool()
    foo = pool.map(run_fw, problems)

    total_time = time.time() - start_time
    hours = total_time // 3600                                                                                                                                                                                                             
    minutes = total_time // 60                                                                                                                                                                                                             
    seconds = total_time % 60                                                                                                                                                                                                              
    with open('time', 'w') as f:                                                                                                                                                                                                           
        f.write(f'{hours}h, {minutes}m, {seconds}s') 
    print(f'Total time is {hours}h, {minutes}m, {seconds}s')
