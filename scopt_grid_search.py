import os
import time
import pickle
from multiprocessing import Pool
from itertools import product

os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
from sklearn.datasets import load_svmlight_file
from scipy.linalg import norm

import scfw.log_reg as lr
from scfw.scopt import scopt


def run_scopt_gs(params):
    problem_name = 'a1a'
    out_dir = os.path.join('results', 'log_reg_gs')
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    Lest, use_two_phase, fista_iter, conj_grad_iter = params

    results_file = '_'.join([problem_name, str(Lest), str(use_two_phase), str(fista_iter), str(conj_grad_iter)])
    results_file = os.path.join(out_dir, results_file + '.pckl')
    if os.path.exists(results_file):
        with open(results_file, 'rb') as f:
            results = pickle.load(f)
    else:
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
    terminate_tol = 1e-5
    
    sc_params={
        #parameters for SCOPT
        'iter_SC': 5000,
        'Lest': Lest,
        'use_two_phase': use_two_phase,
        #FISTA parameters
        'fista_type': 'mfista',
        'fista_tol': 1e-5,
        'fista_iter': fista_iter,
        #Conjugate Gradient Parameters
        'conj_grad_tol':1e-5,
        'conj_grad_iter': conj_grad_iter
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


    print(f'scopt for {problem_name} started!')
    x, alpha_hist, Q_hist, time_hist = scopt(func_x,
            grad_x,
            hess_mult_x,
            hess_mult_vec_x,
            Mf,
            nu,
            prox_func,
            x0,  
            sc_params,                                              
            eps=terminate_tol,                                              
            print_every=50000)
        
    results[problem_name]['scopt'] = {
        'Lest': Lest,
        'use_two_phase': bool(use_two_phase),
        'fista_iter': fista_iter,
        'conj_grad_iter': conj_grad_iter,
        'x': x,
        'alpha_hist': alpha_hist,
        'Q_hist': Q_hist,
        'time_hist': time_hist,
    }

    with open(results_file, 'wb') as f:
        pickle.dump(results, f)    

    print(f'scopt for {problem_name} finished!')
    return results


if __name__ == '__main__':
    start_time = time.time()

    Lest_list = ['backtracking', 'estimate']
    use_two_phase_list = [1, 0]
    fista_iter_list = [500, 1000, 2000]
    conj_grad_iter = [500, 1000, 2000]
    grid_params = product(Lest_list, use_two_phase_list, fista_iter_list, conj_grad_iter)

    pool = Pool(3)
    foo = pool.map(run_scopt_gs, grid_params)

    total_time = time.time() - start_time
    hours = total_time // 3600 
    minutes = total_time // 60 - hours * 60
    seconds = total_time % 60
    with open('time', 'w') as f:
        f.write(f'{hours}h, {minutes}m, {seconds}s') 
    print(f'Total time is {hours}h, {minutes}m, {seconds}s')
