from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import scipy.special
import pandas as pd

df = pd.read_csv("SLS22.csv") #data frame 
Lcq_ids = ["Majerus", "Oliveira","Decenzo","Santiago", "Papa", "Eaton", "Mota", "Shirai", 
           "Jordan", "Hoefler", "Hoban", "Gustavo", "Ribeiro C", "O’neill", "Foy", "Midler"]

def init_normal_dataframe():
    ndf = df
    ndf["run 1"] = [x/10 for x in df["run 1"]]
    ndf["run 2"] = [x/10 for x in df["run 2"]]
    ndf["trick 1"] = [x/10 for x in df["trick 1"]]
    ndf["trick 2"] = [x/10 for x in df["trick 2"]]
    ndf["trick 3"] = [x/10 for x in df["trick 3"]]
    ndf["trick 4"] = [x/10 for x in df["trick 4"]]
    ndf["trick 5"] = [x/10 for x in df["trick 5"]]
    ndf["trick 6"] = [x/10 for x in df["trick 6"]]
    ndf["make 1"] = [int(bool(x)) for x in df["trick 1"].values.tolist()]
    ndf["make 2"] = [int(bool(x)) for x in df["trick 2"].values.tolist()]
    ndf["make 3"] = [int(bool(x)) for x in df["trick 3"].values.tolist()]
    ndf["make 4"] = [int(bool(x)) for x in df["trick 4"].values.tolist()]
    return ndf
ndf = init_normal_dataframe()


ids = []
for name in ndf['id']:
    if name not in ids:
        ids.append(name)

def make_histogram():
    # fig, ax = plt.subplots(1, 1)
    plt.hist(ndf["trick 1"], density=True, histtype='stepfilled', alpha=0.5, label = "Trick 1")
    plt.hist(ndf["trick 2"], density=True, histtype='stepfilled', alpha=0.5, label = "Trick 2")
    plt.hist(ndf["trick 3"], density=True, histtype='stepfilled', alpha=0.5, label = "Trick 3")
    plt.hist(ndf["trick 4"], density=True, histtype='stepfilled', alpha=0.5, label = "Trick 4")
    plt.legend()
    plt.show()


def make_histogram_runs():
    # fig, ax = plt.subplots(1, 1)
    plt.hist(ndf["run 1"], density=True,
             histtype='stepfilled', alpha=0.5, label="run 1")
    plt.hist(ndf["run 2"], density=True,
             histtype='stepfilled', alpha=0.5, label="run 2")
    plt.legend()
    plt.show()

def calculate_Q1_partd():
    count_lands = ndf["make 1"].values.tolist().count(1) + ndf["make 2"].values.tolist().count(1) +ndf["make 3"].values.tolist().count(1) + ndf["make 4"].values.tolist().count(1) 
    count_bigger_than6 = 0;
    alltricks = ndf["trick 1"].values.tolist() + ndf["trick 2"].values.tolist() + ndf["trick 3"].values.tolist() + ndf["trick 4"].values.tolist()
    for x in alltricks:
        if x>0.6:
            count_bigger_than6+=1
    return float(count_bigger_than6)/float(count_lands)

def plot_run1_run2():
    plt.scatter(ndf["run 1"],ndf["run 2"],color='blue', marker='o')
    plt.xlabel('run 1')
    plt.ylabel('run 2')
    plt.legend()
    plt.show()

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


def print_dic(dic):
    for keys,values in dic.items():
        print(keys,values)

### extracting trick and run data
def init_trick_data():
    ndf.set_index('id',inplace=True)
    tricks_data = {}
    n = len(ndf)
    for name in list(ids):
        namesdata = ndf.loc[name]
        if isinstance(namesdata['trick 1'],float):
            tricks_data[name] = np.array([namesdata['trick 1']]+[namesdata['trick 2']]+[namesdata['trick 3']]+[namesdata['trick 4']])
        else:
            tricks_data[name] = np.array(list(namesdata['trick 1'])+list(namesdata['trick 2'])+list(namesdata['trick 3'])+list(namesdata['trick 4']))
    return tricks_data

def init_run_data():
    tricks_data = {}
    n = len(ndf)
    for name in list(ids):
        namesdata = ndf.loc[name]
        if isinstance(namesdata['run 1'],float):
            tricks_data[name] = np.array([namesdata['run 1']]+[namesdata['run 2']])
        else:
            tricks_data[name] = np.array(list(namesdata['run 1'])+list(namesdata['run 2']))
    return tricks_data

tricks_data = init_trick_data()
run_data = init_run_data()



def calc_q1d():
    def display(input):
        df = pd.DataFrame(input)
        df = df.transpose()
        df.columns = ["Trick betyg större än 0.6", "Inte landar tricket"]
        print(df)

    result = {}
    for name in list(ids):
        xdata = tricks_data[name]
        count_not_land = list(xdata).count(0.0)
        count_land = len(xdata)-count_not_land
        count_bigger_than_6=0
        for x in xdata:
            if x>0.6:
                count_bigger_than_6+=1
        
        result[name] = [count_bigger_than_6/count_land, count_not_land/len(xdata)]
    display(result)

### Calculations needed to estimate theta, alpha, beta
average_svariance = 0
for eachdata in tricks_data.values():
    a = np.array([k for k in eachdata if k>0])
    average_svariance+=np.var(a)
average_svariance/=len(ids)

def Theta_MoM_skattning(xdata):
    data = xdata>0.0
    return np.mean(data)

def AlphaBeta_MoM_skattning(xdata):
    data = [x for x in xdata if x>0]
    if len(data)==1:
        svariance = average_svariance
    else:
        svariance = np.var(data, ddof=1)
    m = np.mean(data)
    alpha_0 = m*(m*(1-m)/svariance-1)
    beta_0 = (1-m)*(m*(1-m)/svariance-1)
    return np.array([alpha_0,beta_0])   

### Storing the parameter estimations above to each player in dictionary form
def get_parameters_tricks():
    result = {}
    for name in ids:
        theta = Theta_MoM_skattning(np.array(tricks_data[name]))
        alpha = AlphaBeta_MoM_skattning(np.array(tricks_data[name]))[0]
        beta = AlphaBeta_MoM_skattning(np.array(tricks_data[name]))[1]
        result[name] = [theta,alpha,beta]
    return result

def get_parameters_runs():
    result = {}
    for name in ids:
        alpha = AlphaBeta_MoM_skattning(np.array(run_data[name]))[0]
        beta = AlphaBeta_MoM_skattning(np.array(run_data[name]))[1]
        result[name] = [alpha,beta]
    return result

### Arbitrary metric: calculate the mean of the alpha,beta parameters. USED IN METROPOLIS
def get_avg_alpha_beta_tricks():
    params = list(get_parameters_tricks().values())
    alphas = [params[i][1] for i in range(len(params))]
    betas = [params[i][2] for i in range(len(params))]
    medel_alpha, medel_beta = np.mean(alphas), np.mean(betas)
    return [medel_alpha,medel_beta]

def get_avg_alpha_beta_runs():
    params = get_parameters_runs()
    alphas = [params[name][0] for name in Lcq_ids]
    betas = [params[name][1] for name in Lcq_ids]
    medel_alpha, medel_beta = np.mean(alphas), np.mean(betas)
    return [medel_alpha,medel_beta]

### Debugging functions
def create_table(dict):
    result = {
        'ids': ids,
        'Alpha, Beta':[betyg_arr for betyg_arr in list(dict.values())]
    }

    return pd.DataFrame(result)
def display_params(params):
    df = pd.DataFrame(params)
    df = df.transpose()
    df.columns = ["Theta", "Alpha", "Beta"]
    print(df)

### make histograms of runs and trick data


# def make_hists_trickbetyg():
#     fig, axes = plt.subplots(4, 4, figsize=(12, 8))
#     fig.suptitle(r"trick betyg histograms")
#     plt.subplots_adjust(hspace=0.5, wspace=0.5)
#     c = 0
#     for i in range(4):
#         for j in range(4):
#             skateboarder_id = Lcq_ids[c]
#             xdata = tricks_data[skateboarder_id]            
#             axes[i, j].hist(xdata, density=True)
#             axes[i, j].set_title(Lcq_ids[c])
#             c += 1
#     plt.show()

# if __name__=="__main__":
#     # make_hists_trickbetyg()
#     calc_q1d()

