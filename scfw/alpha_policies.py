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

def alpha_sc(Gap, hess_mult_v, d, Mf, nu):
    e = hess_mult_v ** 0.5
    beta = norm(d)
    t, _=copute_t_nu(Mf,beta,e,Gap,nu)
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

def alpha_L_backtrack(func_t,fx,gx,delta_x,L_last,t_max):
    tau=2
    nu=0.25
    L=nu*L_last
    qx = dot_product(gx, delta_x)
    qqx=L/2*norm(delta_x)**2
    t=min(-qx/(L*norm(delta_x)**2),t_max)
    while func_t(t)>fx+t*qx+t**2*qqx:
        L=tau*L
        qqx=qqx*tau
        t=min(-qx/(2*qqx),t_max)
    return t, L

def alpha_M_backtrack(func_t,fx, Gap, hess_mult_delta, delta_x, Mf_last, t_max,nu):
    tau=2
    mult=1/4
    Mf=mult*Mf_last
    e = hess_mult_delta ** 0.5
    beta = norm(delta_x)
    #decrease is -Gap*t+omega(t*dv)*hess_mult_delta*t^2
    #for numerical reasons, this term is simplified
    if nu == 2:
        decrease_nu=lambda t,dv: 1/(dv**2)*(np.exp(t*dv)-t*dv-1)*hess_mult_delta-t*Gap
    elif nu == 3:
        decrease_nu=lambda t,dv:(-np.log(1-t*dv))/(dv**2)*hess_mult_delta-t*(Gap+hess_mult_delta/dv)
    elif nu == 4:
        decrease_nu=lambda t,dv:-Gap*t+((1-t*dv)*np.log(1-t*dv)+t*dv)/(dv**2)*hess_mult_delta
    else:
        const =  (nu - 2) / (4 - nu)
        const2 = (nu-2) / (2*(3-nu))
        decrease_nu=lambda t,dv: -Gap*t+const*((const2/dv)*(((1-t*dv)**(1/const2))-1)+1)*hess_mult_delta*t
    t, delta_v = copute_t_nu(Mf,beta,e,Gap,nu) #for current f determine by theory
    t = min(t,t_max)
    decrease = decrease_nu(t,delta_v)
    while func_t(t)>fx+decrease: #checks if we exceed the upper bound
        if decrease>0: #if we did not obtain decrease there is a numerical error
            print('something is wrong')
            print(f'Mf = {Mf}, Gap = {Gap}, e = {e}, delta_v={delta_v}, t={t}, t_max={t_max}, omega_val={omega_val},decrease={deacrease}')
        Mf = tau*Mf #increase Mf
        t, delta_v = copute_t_nu(Mf, beta, e, Gap, nu) #compute new stepsize
        t = min(t,t_max)
        decrease=decrease_nu(t,delta_v) #compute the decrease
    #print(f'Mf = {Mf}, Gap = {Gap}, e = {e}, delta_v={delta_v}, t={t}, t_max={t_max}, omega_val={omega_val},decrease={deacrease}')
    return t, Mf

def alpha_sc_hybrid(func_t,fx, gx, Gap, hess_mult_delta, delta_x, L_last, Mf_last, t_max,nu):
    #alpha_M, Mf_last=alpha_M_backtrack(func_t,fx, Gap, hess_mult_delta, delta_x, Mf_last, t_max,nu) #this is using backtracking for Mf
    t_M=alpha_sc(Gap, hess_mult_delta, delta_x, Mf_last, nu) #this is using the theoretical Mf – no backtracking
    t_L,L_last=alpha_L_backtrack(func_t,fx,gx,delta_x,L_last,t_max)
    if t_M>=t_L: #pick the larger stepsize
        t=t_M
    else:
        t=t_L
    return t, L_last, Mf_last


def copute_t_nu(Mf,beta,e,Gap,nu):
    if nu == 2:
        delta_v = Mf * beta
        t = 1 / delta_v * np.log(1 + (Gap*delta_v) / ( e ** 2))
    elif nu == 3:
        delta_v =  Mf * e / 2
        t = Gap / (Gap * delta_v + e ** 2)
    else:
        delta_v = (nu - 2) / 2 * Mf * (beta ** (3 - nu)) * e ** (nu - 2)
        if nu == 4:
            t = 1 / delta * (1 - np.exp(-delta_v * Gap / (e ** 2)))
        elif nu < 4 and nu > 2:
            const = (4 - nu) / (nu - 2)
            t = 1 / delta * (1 - (1 + (-delta_v * Gap * const / (e ** 2))) ** (-1 / const))
    return t, delta_v
