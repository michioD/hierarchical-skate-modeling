
from scipy import stats
import numpy as np 
import matplotlib.pyplot as plt
from scipy.special import loggamma
from statinlprojekt import AlphaBeta_MoM_skattning, run_data, get_parameters_runs, get_avg_alpha_beta_runs

def log_datafördelning(data,alpha,beta):
    log_beta_pdf = lambda x, alpha, beta: loggamma(alpha + beta) - loggamma(alpha) - loggamma(beta) + (alpha - 1)*np.log(x) + (beta - 1)*np.log(1 - x)
    DF = 0
    for x in data:
        DF+= log_beta_pdf(x,alpha,beta)
    return DF


avg_alpha_beta_runs = get_avg_alpha_beta_runs()

def log_prior(alpha, beta):
    lambd = 1
    thet = lambd*(sum(avg_alpha_beta_runs)+1)
    # thet = lambd*(11)
    return np.log(thet**lambd) - loggamma(thet) + (thet - 1)*np.log(alpha + beta + 1) - lambd*(alpha + beta + 1) - np.log(alpha + beta)

def log_posterior(data,alpha, beta):
    log_p = log_prior(alpha, beta)
    return log_p+log_datafördelning(data,alpha,beta)


def make_contour_plot(data):
    alpha_grid = np.linspace(0.1, 6, 100)
    beta_grid = np.linspace(0.1, 6, 100)
    log_posterior_grid = [[log_posterior(data,alpha, beta)for alpha in alpha_grid] for beta in beta_grid]
    posterior_grid = np.exp(log_posterior_grid - np.max(log_posterior_grid))
    plt.figure(figsize=(10, 6))
    plt.contour(alpha_grid, beta_grid, posterior_grid)
    plt.xlabel(r"$\alpha$", fontsize=12)
    plt.ylabel(r"$\beta$", fontsize=12)
    plt.show()


def method_moments(data):
    data = [x for x in data if x>0]
    m1 = np.mean(data)
    m2 = np.mean([x**2 for x in data])
    alpha = (m1*(m1-m2))/(m2-m1**2)
    beta = ((m1-m2)*(1-m1))/(m2-m1**2)
    return([alpha,beta])

def metropolisY(data,sample_size=1e4):
    def genY(X):
        X = np.array(X)
        proposal_alpha = np.exp(np.log(X[0]) + delta * stats.norm.rvs())
        proposal_beta = np.exp(np.log(X[1]) + delta * stats.norm.rvs())
        return np.array([proposal_alpha,proposal_beta])

    n_samples = int(sample_size)
    n_chains = 1
    delta = 0.5

    alphas = np.zeros((n_samples, n_chains))
    betas = np.zeros((n_samples, n_chains))

    init_guess = method_moments(data)
    alphas[0] = init_guess[0] + np.exp(stats.cauchy.rvs(size=n_chains, scale=0.05)) 
    betas[0] = init_guess[1] + np.exp(stats.cauchy.rvs(size=n_chains, scale=0.05))

    for i in range(n_samples - 1):
        for j in range(n_chains):
            last_alpha = alphas[i, j]
            last_beta = betas[i, j] 
            Y = genY([last_alpha,last_beta])
            proposal_alpha = Y[0]
            proposal_beta = Y[1]
            rho = np.exp(log_posterior(data,proposal_alpha, proposal_beta)) / np.exp(log_posterior(data,last_alpha, last_beta)) # f(Y)/f(Xi-1)
            u = stats.uniform.rvs()
            if u <= rho:
                alphas[i + 1, j] = proposal_alpha
                betas[i + 1, j] = proposal_beta
            else:
                alphas[i + 1, j] = last_alpha
                betas[i + 1, j] = last_beta    
    return alphas, betas

if __name__=="__main__":
    Example_skateboarder = "Gustavo"
    data = np.array(run_data[Example_skateboarder])
    alphas, betas, = metropolisY(data,1e4)
    def disp_means():
        print("Sticksprovsmedelvärdet för: " )
        print("alpha: ", np.mean(alphas[:,0]))
        print("beta: ", np.mean(betas[:,0]))
    
    def make_alpha_beta_scatter():
        plt.figure(figsize=(8, 4))
        for j in range(1):
            plt.plot(alphas[:, j], betas[:, j], '.-', markersize=10, alpha=0.5, label="chain"+str(j+1))
        plt.legend()
        plt.title(f"Example skateboarde: {Example_skateboarder}",)
        plt.xlabel('alpha', fontsize=12)
        plt.ylabel('beta', fontsize=12)
        plt.show()
    make_contour_plot(data)
    make_alpha_beta_scatter()
    print(get_parameters_runs()["Gustavo"])
