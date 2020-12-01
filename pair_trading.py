# Importing external libraries
import pandas as pd
import numpy as np
import yfinance as yf
from statsmodels.tsa.stattools import coint


def find_strong_pairs(data):
    """
        Find strongest paired stocks
    """
    n = data.shape[1]
    score_matrix = np.zeros((n, n))
    pvalue_matrix = np.ones((n, n))
    keys = data.keys()
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            S1 = data[keys[i]]
            S2 = data[keys[j]]
            result = coint(S1, S2)
            score = result[0]
            pvalue = result[1]
            score_matrix[i, j] = score
            pvalue_matrix[i, j] = pvalue
            if pvalue < 0.05:
                pairs.append((keys[i], keys[j]))
    return score_matrix, pvalue_matrix, pairs


# read data from csv
df1 = pd.read_csv('data.csv')

# convert date to proper format
df1['date'] = pd.to_datetime(df1['date'])
df1['year'] = df1['date'].dt.year

# convert dataframe into proper for function
ticker_list = list(df1['Name'].value_counts().index.values)
df_tmp = yf.download(ticker_list, start=min(df1['date']), end=max(df1['date']))
df_tmp = df_tmp['Close']
df_tmp.fillna(0, inplace=True)

# NOTE: this function takes a lot of time to run considering we have over 500 different stocks
score_matrix, pvalue_matrix, pairs = find_strong_pairs(df_tmp)

# show all strong pairs
for pair in pairs:
    print(pair)
