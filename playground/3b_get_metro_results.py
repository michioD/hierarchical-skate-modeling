import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from statinlprojekt import tricks_data, Lcq_ids
from metro import metropolis


def make_metro_file():
    thetas_16 = []
    alphas_16 = []
    betas_16 = []
    theta_means = []
    alpha_means = []
    beta_means = []
    theta_variances = []
    alpha_variances = []
    beta_variances = []
    for id in Lcq_ids:
        print(id)
        ids_data = tricks_data[id]
        thetas, alphas, betas = metropolis(ids_data, 1e4)
        
        mean_theta = np.mean(thetas[:, 0])
        mean_alpha = np.mean(alphas[:, 0])
        mean_beta = np.mean(betas[:, 0])
        
        var_theta = np.var(thetas[:, 0], ddof=1)
        var_alpha = np.var(alphas[:, 0], ddof=1)
        var_beta = np.var(betas[:, 0], ddof=1)
        
        theta_means.append(mean_theta)
        alpha_means.append(mean_alpha)
        beta_means.append(mean_beta)
        theta_variances.append(var_theta)
        alpha_variances.append(var_alpha)
        beta_variances.append(var_beta)
        
        thetas_16.append(thetas[:, 0])
        alphas_16.append(alphas[:, 0])
        betas_16.append(betas[:, 0])

    metro_results = {'ids': Lcq_ids,
                    'thetas': thetas_16,
                    'alphas': alphas_16,
                    'betas': betas_16,
                    'theta mean': theta_means,
                    'alpha mean': alpha_means,
                    'beta mean': beta_means,
                    'theta variance': theta_variances,
                    'alpha variance': alpha_variances,
                    'beta variance': beta_variances}

    metro_df = pd.DataFrame(metro_results)
    metro_df.to_json('metro_results.json')

metro_df = pd.read_json('metro_results.json')
metro_df.set_index('ids',inplace=True)


def make_hists():
    fig, axes = plt.subplots(4, 4, figsize=(12, 8))
    fig.suptitle(r"$\theta$ Histograms")
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    c = 0
    for i in range(4):
        for j in range(4):
            thetas = metro_df.loc[Lcq_ids[c]]['thetas']
            axes[i, j].hist(thetas, density=True)
            axes[i, j].set_title(Lcq_ids[c])
            c += 1
    plt.show()


def make_scatters():
    fig, axes = plt.subplots(4, 4, figsize=(12, 8))
    fig.suptitle(r"$\alpha$,$\beta$ scatters")
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    c = 0
    for i in range(4):
        for j in range(4):
            alphas = metro_df.loc[Lcq_ids[c]]['alphas']
            betas = metro_df.loc[Lcq_ids[c]]['betas']
            axes[i, j].scatter(alphas[0], betas[0], marker='o',
                               s=50, alpha=1, color='orange')
            axes[i, j].plot(alphas, betas, '.-', markersize=5, alpha=0.5)
            axes[i, j].set_title(Lcq_ids[c])
            c += 1
    plt.show()

def moving_avg_plot():
    fig, axes = plt.subplots(4, 4, figsize=(12, 8))
    fig.suptitle(r"moving avg")
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    c = 0
    for i in range(4):
        for j in range(4):
            alphas = metro_df.loc[Lcq_ids[c]]['alphas']
            betas = metro_df.loc[Lcq_ids[c]]['betas']
            thetas = metro_df.loc[Lcq_ids[c]]['thetas']
            iters = len(alphas)
            moving_avg_alpha = [np.mean(alphas[:k]) for k in range(1,iters)]
            moving_avg_beta = [np.mean(betas[:k]) for k in range(1,iters)]
            moving_avg_theta = [np.mean(thetas[:k]) for k in range(1,iters)]
            axes[i,j].plot(list(range(1,iters)),moving_avg_alpha, label=r"$\bar{\alpha}$")
            axes[i,j].legend()
            axes[i,j].plot(list(range(1,iters)),moving_avg_beta, label=r"$\bar{\beta}$")
            axes[i,j].plot(list(range(1,iters)),moving_avg_theta, label=r"$\bar{\theta}$")
            axes[i,j].set_title(Lcq_ids[c])
            axes[i,j].legend()
            c+=1
    plt.show()

if __name__ == '__main__':
    # make_hists()
    # make_scatters()
    # moving_avg_plot()
    print(metro_df.drop(['thetas','alphas','betas'],axis=1))
