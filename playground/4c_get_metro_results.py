import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from statinlprojekt import tricks_data, Lcq_ids
from metro_for_theta import metropolis_theta


def make_metro_file():
    thetas_16 = []
    chriss_16 = []
    donalds_16 = []
    theta_means = []
    chris_means = []
    donald_means = []
    theta_variances = []
    chris_variances = []
    donald_variances = []
    for id in Lcq_ids:
        print(id)
        ids_data = tricks_data[id]
        thetas, chriss, donalds = metropolis_theta(ids_data, 1e4)
        
        mean_theta = np.mean(thetas[:, 0])
        mean_chris = np.mean(chriss[:, 0])
        mean_donald = np.mean(donalds[:, 0])
    
        var_theta = np.var(thetas[:, 0], ddof=1)
        var_chris = np.var(chriss[:, 0], ddof=1)
        var_donald = np.var(donalds[:, 0], ddof=1)
        
        theta_means.append(mean_theta)
        chris_means.append(mean_chris)
        donald_means.append(mean_donald)
        theta_variances.append(var_theta)
        chris_variances.append(var_chris)
        donald_variances.append(var_donald)
        
        thetas_16.append(thetas[:, 0])
        chriss_16.append(chriss[:, 0])
        donalds_16.append(donalds[:, 0])

    metro_results = {'ids': Lcq_ids,
                    'thetas': thetas_16,
                    'chriss': chriss_16,
                    'donalds': donalds_16,
                    'theta mean': theta_means,
                    'chris mean': chris_means,
                    'donald mean': donald_means,
                    'theta variance': theta_variances,
                    'chris variance': chris_variances,
                    'donald variance': donald_variances}

    metro_df = pd.DataFrame(metro_results)
    metro_df.to_json('for_theta_metro_results.json')
# make_metro_file()

metro_df = pd.read_json('for_theta_metro_results.json')
metro_df.set_index('ids',inplace=True)




def make_hists():
    fig, axes = plt.subplots(4, 4, figsize=(12, 8))
    fig.suptitle(r"$\theta$ Histograms")
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    c = 0
    for i in range(4):
        for j in range(4):
            thetas = metro_df.loc[Lcq_ids[c]]['thetas']
            axes[i,j].hist(thetas, density=True, color='red')
            axes[i,j].set_title(Lcq_ids[c])
            c+=1
    plt.show()

def make_scatters():
    fig, axes = plt.subplots(4, 4, figsize=(12, 8))
    fig.suptitle(r"$C_i$,$D_i$ scatters")
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    c = 0
    for i in range(4):
        for j in range(4):
            chriss = metro_df.loc[Lcq_ids[c]]['chriss']
            donalds = metro_df.loc[Lcq_ids[c]]['donalds']
            axes[i,j].scatter(chriss[0],donalds[0], marker='o',s=50,alpha=1, color='orange')
            axes[i,j].plot(chriss, donalds, '.-', markersize=5, alpha=0.5, color= 'red')
            axes[i,j].set_title(Lcq_ids[c])
            c+=1
    plt.show()

def moving_avg_plot():
    fig, axes = plt.subplots(4, 4, figsize=(12, 8))
    fig.suptitle(r"moving avg")
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    c = 0
    for i in range(4):
        for j in range(4):
            chriss = metro_df.loc[Lcq_ids[c]]['chriss']
            donalds = metro_df.loc[Lcq_ids[c]]['donalds']
            thetas = metro_df.loc[Lcq_ids[c]]['thetas']
            iters = len(chriss)
            moving_avg_chris = [np.mean(chriss[:k]) for k in range(1,iters)]
            moving_avg_donald = [np.mean(donalds[:k]) for k in range(1,iters)]
            moving_avg_theta = [np.mean(thetas[:k]) for k in range(1,iters)]
            axes[i,j].plot(list(range(1,iters)),moving_avg_chris, label=r"$\bar{\chris}$")
            axes[i,j].legend()
            axes[i,j].plot(list(range(1,iters)),moving_avg_donald, label=r"$\bar{\donald}$")
            axes[i,j].plot(list(range(1,iters)),moving_avg_theta, label=r"$\bar{\theta}$")
            axes[i,j].set_title(Lcq_ids[c])
            axes[i,j].legend()
            c+=1
    plt.show()

if __name__=="__main__":
    make_hists()
    make_scatters()
    # moving_avg_plot()
    print(metro_df.drop(['thetas','chriss','donalds'],axis=1))