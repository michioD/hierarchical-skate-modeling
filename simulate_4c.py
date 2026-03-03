from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
from statinlprojekt import Lcq_ids, ids, print_dic, tricks_data, run_data, AlphaBeta_MoM_skattning, get_parameters_runs
from metro_for_theta import metropolis_theta


actual_finalists = ["Gustavo", "Hoban", "Eaton", "Decenzo"]
Lcq_ids_index = {Lcq_ids[e]: e for e in range(len(Lcq_ids))}


def get_heir_params_tricks():
    xparams = {}
    metro_df = pd.read_json('for_theta_metro_results.json')
    metro_df.set_index('ids', inplace=True)
    for id in Lcq_ids:

        result_theta = metro_df.loc[id]['theta mean']
        xdata = tricks_data[id]
        result_alpha, result_beta = AlphaBeta_MoM_skattning(xdata)
        xparams[id] = [result_theta, result_alpha, result_beta]
    return xparams


def get_heir_params_runs():
    yparams = get_parameters_runs()
    for id in ids:
        if id not in Lcq_ids:
            yparams.pop(id)
    return get_parameters_runs()


xparams, yparams = get_heir_params_tricks(), get_heir_params_runs()


def total_betyg(tricks, runs):
    sorted_trick = sorted(tricks)
    return (sorted_trick[-2]+sorted_trick[-1]) + max(runs)


def sim_trick_betyg(theta, alpha, beta):
    V = np.array(np.random.choice([0, 1], 4, p=[1-theta, theta]))
    Z = np.array(np.random.beta(alpha, beta, 4))
    X = [0, 0, 0, 0]
    for i in range(len(X)):
        X[i] = Z[i]*V[i]
    return X


def get_rand_LCQ():  # returns an array of n LCQs. An LCQ is an array representing
    new_LCQ = {}
    for name in Lcq_ids:
        x_theta, x_alpha, x_beta = xparams[name][0], xparams[name][1], xparams[name][2]
        y_alpha, y_beta = yparams[name][0], yparams[name][1]
        skaters_tricks = sim_trick_betyg(x_theta, x_alpha, x_beta)
        skaters_runs = np.random.beta(y_alpha, y_beta, 2)
        new_LCQ[name] = total_betyg(skaters_tricks, skaters_runs)
    return new_LCQ


def get_rand_finalists():
    rand_LCQ = get_rand_LCQ()
    sorted_players = sorted(rand_LCQ.items(), key=lambda x: x[1])
    return [sorted_players[-4][0], sorted_players[-3][0], sorted_players[-2][0], sorted_players[-1][0]]


def most_common_set(set_list):
    set_counts = Counter(frozenset(s) for s in set_list)
    most_common_set, count = set_counts.most_common(1)[0]
    return set(most_common_set), count


def make_finalist_histogram():
    winner_count = {player: 0 for player in Lcq_ids}
    finalist_array = []
    for i in range(5000):
        new_W = get_rand_finalists()
        finalist_array.append(new_W)
        for e in new_W:
            winner_count[e] += 1

    def disp_most_common_finalists():
        finalist_array_sets = [set(finalists) for finalists in finalist_array]
        actual_finalists_freq = finalist_array_sets.count(
            set(actual_finalists))
        print("The actual finalists are\n", actual_finalists, " has freq: ",
              actual_finalists_freq, " which approx equals: ", 100*actual_finalists_freq/5000, "%")
        most_common_finalists, freq = most_common_set(finalist_array_sets)
        print("The mode is\n", most_common_finalists, " has freq: ",
              freq, " which approx equals: ", 100*freq/5000, "%")
    disp_most_common_finalists()

    plt.figure(figsize=(20, 5))
    colors = ['orange' if x in actual_finalists else 'blue' for x in Lcq_ids]
    plt.bar(Lcq_ids, [winner_count[name]
            for name in Lcq_ids], color=colors, alpha=0.8)
    plt.xlabel("Skaters ids")
    plt.ylabel("# times appeared as finalist")
    plt.title("Simulation using new theta estimation")
    plt.show()


if __name__ == "__main__":
    make_finalist_histogram()
