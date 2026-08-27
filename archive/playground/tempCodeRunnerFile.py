from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
from statinlprojekt import Lcq_ids, ids

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

get_bayes_params_tricks()