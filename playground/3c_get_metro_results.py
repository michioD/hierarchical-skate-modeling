import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from statinlprojekt import run_data, Lcq_ids
from simulate import print_dic
from metro_for_Y import metropolisY


def make_metro_file():
    alphas_16 = []
    betas_16 = []
    alpha_means = []
    beta_means = []
    alpha_variances = []
    beta_variances = []
    for id in Lcq_ids:
        print(id)
        ids_data = run_data[id]
        alphas, betas = metropolisY(ids_data, 1e4)
        
        mean_alpha = np.mean(alphas[:, 0])
        mean_beta = np.mean(betas[:, 0])
        
        var_alpha = np.var(alphas[:, 0], ddof=1)
        var_beta = np.var(betas[:, 0], ddof=1)
        
        alpha_means.append(mean_alpha)
        beta_means.append(mean_beta)

        alpha_variances.append(var_alpha)
        beta_variances.append(var_beta)
        
        alphas_16.append(alphas[:, 0])
        betas_16.append(betas[:, 0])

    metro_results = {'ids': Lcq_ids,
                    'alphas': alphas_16,
                    'betas': betas_16,
                    'alpha mean': alpha_means,
                    'beta mean': beta_means,
                    'alpha variance': alpha_variances,
                    'beta variance': beta_variances}

    metro_df = pd.DataFrame(metro_results)
    metro_df.to_json('for_Y_metro_results.json')

# make_metro_file()

metro_df = pd.read_json('for_Y_metro_results.json')
metro_df.set_index('ids',inplace=True)

def make_scatters():
    fig, axes = plt.subplots(4, 4, figsize=(12, 8))
    fig.suptitle(r"$\alpha$,$\beta$ scatters")
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    c = 0
    for i in range(4):
        for j in range(4):
            alphas = metro_df.loc[Lcq_ids[c]]['alphas']
            betas = metro_df.loc[Lcq_ids[c]]['betas']
            axes[i,j].scatter(alphas[0],betas[0], marker='o',s=50,alpha=1, color='orange')
            axes[i,j].plot(alphas, betas, '.-', markersize=5, alpha=0.5, color = 'green')
            axes[i,j].set_title(Lcq_ids[c])
            c+=1
    plt.show()

if __name__=="__main__":
    make_scatters()
