
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma, loggamma
from statinlprojekt import Theta_MoM_skattning, AlphaBeta_MoM_skattning, tricks_data, get_parameters_tricks, get_avg_alpha_beta_tricks


def log_beta_pdf(x, alpha, beta): return loggamma(alpha + beta) - loggamma(alpha) - \
    loggamma(beta) + (alpha - 1)*np.log(x) + (beta - 1)*np.log(1 - x)


def log_datafördelning(data, theta):
    DF = 0
    for x in data:
        if x == 0:
            DF += np.log(1-theta)
        else:
            DF += np.log(theta)
    return DF


avg_alpha_beta_tricks = get_avg_alpha_beta_tricks()


def log_prior(theta, chris, donald):
    return log_beta_pdf(theta, chris, donald) + (-5/2) * np.log(chris+donald)


def log_posterior(data, theta, chris, donald):
    log_p = log_prior(theta, chris, donald)
    return log_p+log_datafördelning(data, theta)


def method_moments(data):
    data = [x for x in data if x > 0]
    m1 = np.mean(data)
    m2 = np.mean([x**2 for x in data])
    alpha = (m1*(m1-m2))/(m2-m1**2)
    beta = ((m1-m2)*(1-m1))/(m2-m1**2)
    return([alpha, beta])


def metropolis_theta(data, sample_size=1e4):
    def genY(X):
        X = np.array(X)
        proposal_chris = np.exp(0.5 * stats.norm.rvs())
        proposal_donald = np.exp(0.5 * stats.norm.rvs())
        proposal_theta = stats.beta.rvs(X[0], X[1])
        return np.array([proposal_chris, proposal_donald, proposal_theta])

    n_samples = int(sample_size)
    n_chains = 1
    delta = 0.5

    chriss = np.zeros((n_samples, n_chains))
    donalds = np.zeros((n_samples, n_chains))
    thetas = np.zeros((n_samples, n_chains))

    chriss[0] = 1 
    donalds[0] = 1  
    thetas[0] = Theta_MoM_skattning(data)

    for i in range(n_samples - 1):
        for j in range(n_chains):
            last_chris = chriss[i, j]
            last_donald = donalds[i, j]
            last_theta = thetas[i, j]
            Y = genY([last_chris, last_donald, last_theta])

            proposal_chris = Y[0]
            proposal_donald = Y[1]
            proposal_theta = Y[2]
            rho = np.exp(log_posterior(data, proposal_theta, proposal_chris, proposal_donald)) / \
                np.exp(log_posterior(data, last_theta,
                       last_chris, last_donald))  # f(Y)/f(Xi-1)
            u = stats.uniform.rvs()
            if u <= rho:
                chriss[i + 1, j] = proposal_chris
                donalds[i + 1, j] = proposal_donald
                thetas[i+1, j] = proposal_theta
            else:
                chriss[i + 1, j] = last_chris
                donalds[i + 1, j] = last_donald
                thetas[i+1, j] = last_theta

    def make_thetas_hist():
        plt.figure(figsize=(8, 4))
        for j in range(n_chains):
            plt.hist(thetas[:, j], bins=20, alpha=0.5,
                     density=True, label="chain "+str(j+1))
            plt.scatter(thetas[0, j], 0, label="MoM skattning")
            plt.ylabel(r'$f(\theta|x)$')
            plt.xlabel(r'$\theta$')
            plt.legend()
        plt.show()

    def make_alpha_beta_scatter():
        plt.figure(figsize=(8, 4))
        for j in range(n_chains):
            plt.plot(chris[:, j], donald[:, j], '.-',
                     markersize=10, alpha=0.5, label="chain"+str(j+1))
        plt.legend()
        plt.xlabel('alpha', fontsize=12)
        plt.ylabel('beta', fontsize=12)
        plt.show()

    return thetas, chriss, donalds


if __name__ == "__main__":
    Example_skateboarder = "Gustavo"
    data = np.array(tricks_data[Example_skateboarder])
    thetas, chris, donald, = metropolis_theta(data, 1e4)

    def make_thetas_hist():
        plt.figure(figsize=(8, 4))
        for j in range(1):
            plt.hist(thetas[:, j], bins=20, alpha=0.5,
                     density=True, label="chain "+str(j+1), color='red')
            plt.scatter(thetas[0, j], 0, label="MoM skattning", color='blue')
            plt.ylabel(r'$f(\theta|x)$')
            plt.xlabel(r'$\theta$')
            plt.legend()
        plt.show()

    make_thetas_hist()

    def make_alpha_beta_scatter():
        plt.figure(figsize=(8, 4))
        for j in range(1):
            plt.plot(chris[:, j], donald[:, j], '.-',
                     markersize=10, alpha=0.5, label="chain"+str(j+1), color='red')
        plt.legend()
        plt.xlabel('c', fontsize=12)
        plt.ylabel('d', fontsize=12)
        plt.show()

    make_alpha_beta_scatter()
    print(get_parameters_tricks()["Gustavo"])
