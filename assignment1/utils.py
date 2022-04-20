import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import os

"""
implement the following functions: 
* train-test-split
* grid-search-cv
"""


def train_test_split(df, test_size = 10, shuffle = True):
    """
    df = pandas dataframe
    test_size in percentage
    return train_df, test_df
    """
    num_samples = df.shape[0]
    num_test_samples = round(test_size * num_samples / 100)
    num_training_samples = num_samples - num_test_samples
    if shuffle:
        sampled_indexes = np.random.randint(0,num_samples, num_test_samples)
        test_df = df.loc[sampled_indexes]
        test_df = test_df.sample(frac = 1)
        train_df = df.drop(sampled_indexes)
        train_df = train_df.sample(frac = 1)
        return train_df, test_df

    else:
        train_df = df.loc[:num_training_samples]
        test_df = df.loc[num_training_samples:]
        return train_df, test_df
    
    
def R2_score(predicted_values, actual_values):
        correlation_matrix = np.corrcoef(predicted_values, actual_values)
        correlation_xy = correlation_matrix[0,1]
        r_squared = correlation_xy**2
        return r_squared  

if __name__ == "__main__":
    df = pd.read_csv("./data/covid_data_india.csv")
    train_df, test_df = train_test_split(df)
    print(train_df.head(), test_df.head())