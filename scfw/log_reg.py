import numpy as np


# def logistic_loss(Phi, y, x, mu):
#     """
#         Phi -- N x n
#         y -- N x 1
#         x -- 1 x n
#         mu -- 1 x 1
#     """
#     Phix = Phi * x # N x 1
#     return np.log(1 + np.exp(-y * (Phix + mu))) # N x 1

def log_reg(Phi, y, x, mu, gamma,mult, exp_product=None):
    """
        Phi -- N x n
        y -- N x 1
        x -- 1 x n
        mu -- 1 x 1
        gamma -- const
    """
    N, n = Phi.shape
    if exp_product is None:
        Phix = Phi @ x # N x 1
        exp_product= np.exp(-y * (Phix + mu))
    log_loss = np.log(1 + exp_product)
    return np.sum(log_loss)*mult/N + gamma*mult/2 * np.linalg.norm(x,2)**2, exp_product

def log_reg_expanded(Phi, y, x, mu, gamma,mult, exp_product=None):
    """
        Phi -- N x n
        y -- N x 1
        x -- מ+1 x 1
        mu -- 1 x 1
        gamma -- const
    """
    N, n = Phi.shape
    if len(x)<n+1:
        sys.error("x not long enough")
    if exp_product is None:
        Phix = Phi @ x[0:n] # N x 1
        exp_product= np.exp(-y * (Phix + mu))
    log_loss = np.log(1 + exp_product)
    return np.sum(log_loss)*mult/N+ gamma/2*mult * x[n], exp_product


def grad_log_reg(Phi, y, x, mu, gamma, exp_product,mult):
    """
        1 / N \sum_{i = 1}^N -y * Phi / (exp(y * (<Phi, x> + mu)) + 1) + gamma * x
        Phi -- N x n
        y -- N x 1
        x -- 1 x n
        mu -- 1 x 1
        gamma -- const
    """
    N = len(exp_product)
    tot=Phi.T@(-y*exp_product/(exp_product+1))*mult/N+gamma*x*mult # n x 1
    return tot

def grad_log_reg_extended(Phi, y, mu, gamma,exp_product,mult):
    """
        1 / N \sum_{i = 1}^N -y * Phi / (exp(y * (<Phi, x> + mu)) + 1) + gamma * x
        Phi -- N x n
        y -- N x 1
        x -- n +1x 1
        mu -- 1 x 1
        gamma -- const
    """
    grad_x=grad_log_reg(Phi, y, mu, gamma, exp_product,mult)
    grad=grad_x.append(gamma)
    return grad

def hess(Phi, y, mu, gamma, exp_product,s,mult):
    """
        Phi -- N x n
        y -- N x 1
        mu -- 1 x 1
        gamma -- const
        Phix -- N x 1
        s -- n x 1
    """
    n=len(s)
    #N = len(exp_product)
    frac = (-y*exp_product) / (1 + exp_product)  # N x 1
    Phi_prod=(Phi.T@frac)
    Mat=Phi_prod.T@Phi_prod
    if mat.shape[0]!=n:
        sys.exit()
    return Mat*mult/N+gamma*np.eye(n)*mult

def hess_mult_vec(Phi, y, mu, gamma, exp_product,s,mult):
    """
        Phi -- N x n
        y -- N x 1
        mu -- 1 x 1
        gamma -- const
        Phix -- N x 1
        s -- n x 1
    """
    N = len(exp_product)
    frac = (-y*exp_product) / (1 + exp_product)  # N x 1
    Phis=Phi@s
    return (Phi.T@frac)*(frac.dot(Phis))*mult/N+gamma*s*mult

def hess_mult_vec_extended(Phi, y, mu, gamma, exp_product,s,mult):
    """
        Phi -- N x n
        y -- N x 1
        mu -- 1 x 1
        gamma -- const
        Phix -- N x 1
        s -- n+1 x 1
    """
    n=Phi.shape[1]
    hess_vec=hess_mult_vec(Phi, y, mu, gamma, exp_product,s[0:n],mult)
    hess_vec=hess_vec.append(0)
    return hess_vec

def hess_mult_log_reg(Phi, y,mu, gamma, exp_product,s,mult):
    """
        Phi -- N x n
        mu -- 1 x 1
        gamma -- const
        Phix -- N x 1
        s -- n x 1
    """
    N = len(exp_product)
    frac = (y*exp_product) / (1 + exp_product)  # N x 1
    Phis=Phi@s
    return (Phis.T@frac)**2*mult/N+gamma*np.linalg.norm(s,2)**2*mult

def hess_mult_log_reg_extended(Phi, y,mu, gamma, exp_product,s,mult):
    """
        Phi -- N x n
        mu -- 1 x 1
        gamma -- const
        Phix -- N x 1
        s -- n+1 x 1
    """
    return hess_mult_log_reg(Phi, y,mu, gamma, exp_product,s[0:n],mult)


def linear_oracle_l1_level_set(c, M):
    n = len(grad)-1
    s=linear_oracle_l1(grad[0:n],1)
    if s.T.dot*c[0:n]+c[n]<0:
        s_whole=s.append(1)*M;
    else:
        s_whole=zeros(n+1)
    return s_whole

def linear_oracle_l1(c,r):
    n = len(c)
    s = np.zeros(n)
    i_max = np.argmax(np.abs(c))
    s[i_max] = -r*np.sign(c[i_max]) # 1 x n
    return s

def projection_l1(y,r):
    if np.linalg.norm(y,1)<=r:
        return y.copy()
    else:
        y_abs=np.abs(y/r)
        P_y_abs=proj_simplex(y_abs)
        P_y=P_y_abs*np.sign(y)*r
    return P_y

def proj_simplex(y):
    ind = np.argsort(y)
    sum_y = sum(y)
    origin_y = sum_y
    n = len(y)
    Py = y.copy()
    for i in range(n):
        t = (sum_y - 1) / (n - i)
        if (origin_y > 1 and t < 0): #for numerical errors
            sum_y = sum(y[ind[i : n - 1]])
            t = (sum_y - 1) / (n - i)
        if i > 0:
            if t <= y[ind[i]] and t >= y[ind[i - 1]]:
                break
        elif t <= y[ind[i]]:
            break
        sum_y -= y[ind[i]]
        Py[ind[i]] = 0
    Py = np.maximum(y - t, np.zeros(n))
    return Py
