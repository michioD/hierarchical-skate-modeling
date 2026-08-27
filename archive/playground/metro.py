
from scipy import stats
import numpy as np 
import matplotlib.pyplot as plt
from scipy.special import gamma, loggamma
from statinlprojekt import Theta_MoM_skattning, AlphaBeta_MoM_skattning, tricks_data, get_parameters_tricks, get_avg_alpha_beta_tricks

# data = np.array([0.16, 0.32, 0, 0, 0, 0.16, 0.36, 0.41, 0.48, 0.28])



def log_datafördelning(data,theta, alpha,beta):
    log_beta_pdf = lambda x, alpha, beta: loggamma(alpha + beta) - loggamma(alpha) - loggamma(beta) + (alpha - 1)*np.log(x) + (beta - 1)*np.log(1 - x)
    DF = 0
    for x in data:
        if x==0:
            DF+= np.log(1-theta)
        else:
            DF+= log_beta_pdf(x,alpha,beta) + np.log(theta)
    return DF

avg_alpha_beta_tricks = get_avg_alpha_beta_tricks()
def log_prior(alpha, beta):
    lambd = 1
    thet = lambd*(sum(avg_alpha_beta_tricks)+1)
    return np.log(thet**lambd) - loggamma(thet) + (thet - 1)*np.log(alpha + beta + 1) - lambd*(alpha + beta + 1) - np.log(alpha + beta)

def log_posterior(data,theta, alpha, beta):
    log_p = log_prior(alpha, beta)
    return log_p+log_datafördelning(data,theta,alpha,beta)


def make_contour_plot(data):
    theta = Theta_MoM_skattning(data)
    alpha_grid = np.linspace(0.1, 100, 100)
    beta_grid = np.linspace(0.1, 60, 100)
    log_posterior_grid = [[log_posterior(data,theta, alpha, beta)for alpha in alpha_grid] for beta in beta_grid]
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

def metropolis(data,sample_size=1e4):
    def genY(X):
        X = np.array(X)
        proposal_alpha = np.exp(np.log(X[0]) + delta * stats.norm.rvs())
        proposal_beta = np.exp(np.log(X[1]) + delta * stats.norm.rvs())
        proposal_theta = stats.uniform.rvs()
        return np.array([proposal_alpha,proposal_beta,proposal_theta])

    n_samples = int(sample_size)
    n_chains = 1
    delta = 0.5

    alphas = np.zeros((n_samples, n_chains))
    betas = np.zeros((n_samples, n_chains))
    thetas = np.zeros((n_samples,n_chains))


    init_guess = AlphaBeta_MoM_skattning(data)
    alphas[0] = init_guess[0] + np.exp(stats.cauchy.rvs(size=n_chains, scale=0.05)) 
    betas[0] = init_guess[1] + np.exp(stats.cauchy.rvs(size=n_chains, scale=0.05))
    thetas[0] = Theta_MoM_skattning(data)

    for i in range(n_samples - 1):
        # print(i)
        for j in range(n_chains):
            last_alpha = alphas[i, j]
            last_beta = betas[i, j] 
            last_theta = thetas[i,j]
            Y = genY([last_alpha,last_beta, last_theta])
            proposal_alpha = Y[0]
            proposal_beta = Y[1]
            proposal_theta = Y[2]
            rho = np.exp(log_posterior(data,proposal_theta, proposal_alpha, proposal_beta)) / np.exp(log_posterior(data,last_theta, last_alpha, last_beta)) # f(Y)/f(Xi-1)
            u = stats.uniform.rvs()
            if u <= rho:
                alphas[i + 1, j] = proposal_alpha
                betas[i + 1, j] = proposal_beta
                thetas[i+1, j] = proposal_theta
            else:
                alphas[i + 1, j] = last_alpha
                betas[i + 1, j] = last_beta
                thetas[i+1, j] = last_theta
    

    def make_thetas_hist():
        plt.figure(figsize=(8, 4))
        for j in range(n_chains):
            plt.hist(thetas[:, j],bins=20, alpha=0.5, density=True, label="chain "+str(j+1))
            plt.scatter(thetas[0,j],0, label="MoM skattning")
            plt.ylabel(r'$f(\theta|x)$')
            plt.xlabel(r'$\theta$')
            plt.legend()
        plt.show()

    def make_alpha_beta_scatter():
        plt.figure(figsize=(8, 4))
        for j in range(n_chains):
            plt.plot(alphas[:, j], betas[:, j], '.-', markersize=10, alpha=0.5, label="chain"+str(j+1))
        plt.legend()
        plt.xlabel('alpha', fontsize=12)
        plt.ylabel('beta', fontsize=12)
        plt.show()

    def Plot_3D():
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(thetas[:,0],alphas[:,0],betas[:,0], c = 'b',marker='o')
        ax.set_xlabel(r"$theta$")
        ax.set_ylabel(r"$alpha$")
        ax.set_zlabel(r"$beta$")
        plt.show()

    # Plot_3D()
    def disp_means():
        print("Sticksprovsmedelvärdet för: " )
        print("theta: ", np.mean(thetas[:,0]))
        print("alpha: ", np.mean(alphas[:,0]))
        print("beta: ", np.mean(betas[:,0]))
    return thetas, alphas, betas

if __name__=="__main__":
    data = np.array(tricks_data["Gustavo"])
    make_contour_plot(data)
    # thetas, alphas, betas, = metropolis(data,1e3)
    print(get_parameters_tricks()["Gustavo"])