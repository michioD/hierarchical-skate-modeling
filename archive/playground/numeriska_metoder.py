import numpy as np
# from statinlprojekt import get_pooled_var
from scipy.special import polygamma, gammaln, psi
PsiPrime = lambda x: polygamma(1,x)

# global_svariance = 0.16403415647726363

pooled_var = 0.004165276081942748

def AlphaBeta_MoM_skattning(xdata):
    data = [x for x in xdata if x>0]
    if len(data)==1:
        svariance = pooled_var
    else:
        svariance = np.var(data,ddof=1)
    alpha_0 = np.mean(data)*((1-np.mean(data))/svariance-1)
    beta_0 = (1-np.mean(data))*((1-np.mean(data))/svariance-1)
    return np.array([alpha_0,beta_0])   

def beta_likelihood(alpha, beta, data):
    log_likelihood = gammaln(alpha + beta) - (gammaln(alpha) + gammaln(beta)) + np.sum((alpha - 1) * np.log(data) + (beta - 1) * np.log(1 - data))
    return log_likelihood

def grad_F(alpha: float,beta: float,data: np.array): #want this to be 0¨
    data = np.array(data)
    k = len(data)
    partial_alpha = psi(alpha+beta) -psi(alpha) + np.sum(np.log(data))/k
    partial_beta = psi(alpha+beta) - psi(beta) + np.sum(np.log(1-data))/k
    result = np.array([partial_alpha,partial_beta])
    return result

def Jac(alpha,beta):
    result = np.array(
        [
            [PsiPrime(alpha+beta) - PsiPrime(alpha), PsiPrime(alpha+beta)],
            [PsiPrime(alpha+beta), PsiPrime(alpha+beta)-PsiPrime(beta)]
        ])
    return result

def newton_raphson(data):
    data = np.array([x for x in data if x>0])
    alpha_beta = AlphaBeta_MoM_skattning(data)  # Initial guess using the method of moments
    tol = 1e-6  # Tolerance for convergence
    max_iter = 100  # Maximum number of iterations
    iter_count = 0
    while iter_count < max_iter:
        alpha, beta = alpha_beta
        likelihood_gradient = grad_F(alpha, beta, data)
        hessian = Jac(alpha, beta)
        alpha_beta -= np.linalg.solve(hessian, likelihood_gradient)
        if np.max(np.abs(likelihood_gradient)) < tol:
            break
        iter_count += 1

    return alpha_beta

