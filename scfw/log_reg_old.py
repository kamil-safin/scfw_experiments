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

def log_reg(Phi, y, x, mu, gamma, Phix=None):
    """
        Phi -- N x n
        y -- N x 1
        x -- 1 x n
        mu -- 1 x 1
        gamma -- const
    """
    N, n = Phi.shape
    if Phix==None:
        Phix = Phi @ x # N x 1
    log_loss = np.log(1 + np.exp(-y * Phix + mu))
    return np.mean(log_loss) + gamma * np.norm(x,1), Phix

def log_reg_expanded(Phi, y, x, mu, gamma, Phix=None):
    """
        Phi -- N x n
        y -- N x 1
        x -- 1 x n
        mu -- 1 x 1
        gamma -- const
    """
    N, n = Phi.shape
    if len(x)<n+1:
        sys.error("x not long enough")
    if Phix==None:
        Phix = Phi @ x[0:n] # N x 1
    log_loss = np.log(1 + np.exp(-y * (Phix + mu)))
    return np.sum(log_loss) + gamma * x[n], Phix


def grad_log_reg(Phi, y, mu, gamma, Phix):
    """
        1 / N \sum_{i = 1}^N -y * Phi / (exp(y * (<Phi, x> + mu)) + 1) + gamma * x
        Phi -- N x n
        y -- N x 1
        x -- 1 x n
        mu -- 1 x 1
        gamma -- const
    """
    N = len(Phix)
    exp_product=np.exp(-y * (Phix + mu)) # N x 1
    tot=-Phi.T@(y*exp_product/(exp_product+1)) # n x 1
    return tot

def grad_log_reg_extended(Phi, y, mu, gamma, Phix):
    """
        1 / N \sum_{i = 1}^N -y * Phi / (exp(y * (<Phi, x> + mu)) + 1) + gamma * x
        Phi -- N x n
        y -- N x 1
        x -- n +1x 1
        mu -- 1 x 1
        gamma -- const
    """
        grad_x=grad_log_reg(Phi, y, mu, gamma, Phix)
        grad=grad_x.append(gamma)
    return grad

def hess_mult_vec(Phi, y, mu, gamma, Phix,s):
    """
        Phi -- N x n
        y -- N x 1
        mu -- 1 x 1
        gamma -- const
        Phix -- N x 1
        s -- n x 1
    """
    N = len(Phix)
    exp_product = np.exp(y * (Phix + mu))
    frac = y*exp_product / (1 + exp_product) ** 2 # N x 1
    Phis=Phix@s
    return Phis@(Phis.T@frac)

def hess_mult_vec_extended(Phi, y, mu, gamma, Phix,s):
    """
        Phi -- N x n
        y -- N x 1
        mu -- 1 x 1
        gamma -- const
        Phix -- N x 1
        s -- n+1 x 1
    """
    n=Phi.shape[1]
    hess_vec=hess_mult_vec(Phi, y, mu, gamma, Phix,s[0:n])
    hess_vec=hess_vec.append(0)
    return hess_vec

def hess_mult_log_reg(Phi, y,mu, gamma, Phix,s):
    """
        Phi -- N x n
        mu -- 1 x 1
        gamma -- const
        Phix -- N x 1
        s -- n x 1
    """
    N = len(Phix)
    exp_product = np.exp(y * (Phix + mu))
    frac = y*exp_product / (1 + exp_product) ** 2 # N x 1
    Phis=Phix@s
    return (Phis.T@frac)**2

def hess_mult_log_reg_extended(Phi, y,mu, gamma, Phix,s):
    """
        Phi -- N x n
        mu -- 1 x 1
        gamma -- const
        Phix -- N x 1
        s -- n+1 x 1
    """
    return hess_mult_log_reg(Phi, y,mu, gamma, Phix,s[0:n])


def linear_oracle_l1_level_set(c, M):
    n = len(grad)-1
    s=linear_oracle_l1(grad[0:n])
    if s.T.dot*c[0:n]+c[n]<0:
        s_whole=s.append(1)*M;
    else:
        s_whole=zeros(n+1)
    return s_whole

def linear_oracle_l1(c):
    n = len(c)
    s = np.zeros(n)
    i_max = np.argmax(np.abs(c))
    s[i_max] = -np.sign(c[index]) # 1 x n
    return s
