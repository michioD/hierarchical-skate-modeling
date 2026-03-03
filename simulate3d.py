from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
from statinlprojekt import Lcq_ids, ids, print_dic
#["Majerus", "Oliveira","Decenzo","Santiago", "Papa", "Eaton", "Mota", "Shirai","Jordan", "Hoefler", "Hoban", "Gustavo", "Ribeiro C", "O’neill", "Foy", "Midler"]

actual_finalists = ["Gustavo", "Hoban", "Eaton", "Decenzo"]

Lcq_ids_index = {Lcq_ids[e]:e for e in range(len(Lcq_ids))}


def get_bayes_params_tricks():
    xparams = {}
    metro_df = pd.read_json('metro_results.json')
    metro_df.set_index('ids',inplace=True)
    for id in Lcq_ids:
        xparams[id]= [metro_df.loc[id]['theta mean'],metro_df.loc[id]['alpha mean'], metro_df.loc[id]['beta mean']]
    return xparams

def get_bayes_params_runs():
    yparams = {}
    metro_df_Y = pd.read_json('for_Y_metro_results.json')
    metro_df_Y.set_index('ids',inplace=True)
    for id in Lcq_ids:
        yparams[id]= [metro_df_Y.loc[id]['alpha mean'], metro_df_Y.loc[id]['beta mean']]
    return yparams


xparams, yparams = get_bayes_params_tricks(), get_bayes_params_runs()

def sort_with_indices(arr):
    indexed_arr = list(enumerate(arr))
    sorted_arr = sorted(indexed_arr, key=lambda x: x[1])
    indices = [index for index, _ in sorted_arr]
    return indices

def total_betyg(tricks, runs):
    sorted_trick = sorted(tricks)
    return (sorted_trick[-2]+sorted_trick[-1]) + max(runs)


def sim_trick_betyg(theta,alpha,beta):
    V = np.array(np.random.choice([0,1], 4, p = [1-theta,theta]))
    Z = np.array(np.random.beta(alpha,beta,4))
    X = [0,0,0,0]
    for i in range(len(X)):
        X[i] = Z[i]*V[i]
    return X


def get_rand_LCQ(): # returns an array of n LCQs. An LCQ is an array representing 
    new_LCQ = {}
    for name in Lcq_ids:
        x_theta,x_alpha,x_beta = xparams[name][0],xparams[name][1],xparams[name][2]
        y_alpha,y_beta = yparams[name][0],yparams[name][1]
        skaters_tricks = sim_trick_betyg(x_theta,x_alpha,x_beta)
        skaters_runs = np.random.beta(y_alpha,y_beta,2)
        new_LCQ[name] = total_betyg(skaters_tricks, skaters_runs)
    return new_LCQ


def get_rand_finalists():
    rand_LCQ = get_rand_LCQ()
    sorted_players = sorted(rand_LCQ.items(), key = lambda x:x[1])
    return [sorted_players[-4][0],sorted_players[-3][0],sorted_players[-2][0],sorted_players[-1][0]]

def most_common_set(set_list):
    set_counts = Counter(frozenset(s) for s in set_list)
    most_common_set, count = set_counts.most_common(1)[0]
    return set(most_common_set), count

def most_common_array(arrays):
    tuple_arrays = [tuple(inner_list) for inner_list in arrays]
    array_counts = Counter(tuple_arrays)
    most_common_tuple, count = array_counts.most_common(1)[0]
    most_common_array = list(most_common_tuple)
    return most_common_array, count


def make_finalist_histogram():
    winner_count = {player : 0 for player in Lcq_ids}
    finalist_array = []
    for i in range(5000):
        new_W = get_rand_finalists()
        finalist_array.append(new_W)
        for e in new_W:
            winner_count[e] += 1

    def disp_most_common_finalists():
        finalist_array_sets = [set(finalists) for finalists in finalist_array]
        actual_finalists_freq = finalist_array_sets.count(set(actual_finalists))
        print("The actual finalists are\n", actual_finalists, " has freq: ", actual_finalists_freq, " which approx equals: ",100*actual_finalists_freq/5000,"%")
        most_common_finalists, freq = most_common_set(finalist_array_sets)
        print("The mode is\n", most_common_finalists, " has freq: ", freq, " which approx equals: ", 100*freq/5000,"%")
    disp_most_common_finalists()

    
    # plt.figure(figsize=(20, 5))
    colors = ['orange' if x in actual_finalists else 'blue' for x in Lcq_ids]
    plt.bar(Lcq_ids, [winner_count[name] for name in Lcq_ids], color=colors, alpha=0.8)
    plt.xlabel("Skaters ids")
    plt.ylabel("# times appeared as finalist")
    plt.title("Simulation using bayes MCMC estimation")
    plt.show()

if __name__=="__main__":
    make_finalist_histogram()
