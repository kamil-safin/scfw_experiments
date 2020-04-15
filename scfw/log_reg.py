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

def log_reg(Phi, y, x, mu, gamma):
    """
        Phi -- N x n
        y -- N x 1
        x -- 1 x n
        mu -- 1 x 1
        gamma -- const
    """
    Phix = Phi @ x # N x 1
    log_loss = np.log(1 + np.exp(-y * Phix + mu))
    return np.mean(log_loss) + gamma / 2 * x.dot(x), Phix


def grad(Phi, y, x, mu, gamma, Phix):
    """
        1 / N \sum_{i = 1}^N -y * Phi / (exp(y * (<Phi, x> + mu)) + 1) + gamma * x
        Phi -- N x n
        y -- N x 1
        x -- 1 x n
        mu -- 1 x 1
        gamma -- const
    """
    N = len(Phix)
    Phiy = Phi.multiply(y.reshape(-1, 1)) # N x n
    frac = (1 / (np.exp(y * (Phix + mu)) + 1)).reshape(-1, 1) # N x 1
    return np.array(-1 / N * Phiy.T.dot(frac) + gamma * x.reshape(-1, 1)).flatten()

def hess_mult_vec(Phi, y, x, mu, gamma, Phix):
    """
        Phi -- N x n
        y -- N x 1
        x -- 1 x n
        mu -- 1 x 1
        gamma -- const
        Phix -- N x 1
    """
    N = len(Phix)
    exp_product = np.exp(y * (Phix + mu))
    frac = exp_product / (1 + exp_product) ** 2 # N x 1
    return 1 / N * Phi.T @ (Phix * frac) + gamma * x
#     fst = Phi.multiply(frac.reshape(-1, 1)) # N x n
#     snd = Phix # N x 1
#     return 1 / N * fst.T @ snd + gamma * x

def hess_mult_log_reg(Phi, y, x, mu, gamma, Phix):
    """
        Phi -- N x n
        y -- N x 1
        x -- 1 x n
        mu -- 1 x 1
        gamma -- const
        Phix -- N x 1
    """
    N = len(Phix)
    exp_product = np.exp(y * (Phix + mu))
    frac = exp_product / (1 + exp_product)**2 # N x 1
    Phix2 = Phix**2
    return 1 / N * Phix2.T @ (frac) + gamma * x.dot(x)
#     rhs = (Phix / (1 + exp_product))**2
#     return 1 / N * np.sum(exp_product * rhs) + gamma * x.dot(x)

