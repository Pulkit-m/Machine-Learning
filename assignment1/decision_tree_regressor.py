import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import os

class DecisionTreeRegressor:
    def __init__(self, df, min_samples = 2, max_depth = None):
        self.df = df 
        self.max_depth = max_depth 
        self.min_samples = min_samples
        self.tree_depth = 0 
        self.tree_structure = None

    def get_potential_splits(self, data):
        """
        take data in form of pandas.DataFrame.values
        returns a dictionary where keys are column indexes, and values is the list of all the possible potential splits for that respective key
        """
        potential_splits = dict()
        _, num_cols = data.shape
        for col in range(num_cols - 1): 
            potential_splits[col] = list()
            values = np.unique(data[:,col])
            for i in range(len(values)-1):
                value = np.mean([values[i], values[i+1]])
                potential_splits[col].append(value)
        
        return potential_splits

    def split_data(self, data, split_col, split_val):
        """
        data is of form pandas.DataFrame.values
        split_col is the (proposed)optimal col and split_val is the (proposed)value for the split
        returns data_above, data_below
        """
        data_above = data[data[:,split_col] <= split_val]
        data_below = data[data[:,split_col] > split_val]
        return data_above, data_below

    def calculate_std(self, data):
        """
        data is of form pandas.DataFrame.values. Also data is inclusive of target column
        returns variance and covariance based on the target column
        """
        target = data[:,-1]
        N = len(target)
        average = np.mean(target)
        std_dev = np.std(target)
        covariance = (std_dev/average)*100
        return std_dev, covariance  

    def calculate_overall_std(self, data_above, data_below):
        """
        input data of form pandas.DataFrame.values
        returns weighted average of entropies of the input tables
        """
        num_samples_above, num_samples_below = len(data_above), len(data_below)
        total_samples = num_samples_above + num_samples_below
        std_above, _ = self.calculate_std(data_above)
        std_below, _ = self.calculate_std(data_below)
        overall_std = (num_samples_above/total_samples)*std_above + (num_samples_below/total_samples)*std_below
        return overall_std

    def determine_best_split(self, data, potential_splits):
        """
        for all columns and for all values of potential splits in every column evaluate 
        the best possible reduction in std_deviation
        """
        best_split_col = None
        best_split_value = None
        std_pre_split, _ = self.calculate_std(data)
        best_std = std_pre_split
        for col in potential_splits.keys():
            for value in potential_splits[col]:
                data_above, data_below = self.split_data(data, split_col= col, split_val = value)
                std_post_split = self.calculate_overall_std(data_above, data_below)
                if(std_post_split <= best_std):
                    best_split_col = col
                    best_split_value = value
                    best_std = std_post_split
        # print("Splitting wrt column {} and value {}; Std Dev reduced from {} to {}".format(best_split_col, best_split_value, std_pre_split, best_std))
        return best_split_col, best_split_value 

    def average_of_target_col(self, data):
        """
        returns the average of the target values of samples in the dataframe
        """
        target = data[:,-1]
        if(len(target)>0):
            return np.mean(target)
        else:
            return 0


    def build_tree(self,df,df_type = 'pandas', depth = 0):
        """
        df_type: (string) either "pandas" or "numpy"
        do not touch the depth parameter. It is being used as a counter within the function.
        when max_depth is an integer then pruning comes into action.
        Otherwise leaf node or not is decided by the minimum number of samples in the data passed
        """
        if df_type == 'pandas':
            global COLUMN_NAMES
            COLUMN_NAMES = df.columns 
            data = df.values 
        elif df_type == 'numpy':
            data = df 
        
        # base cases
        if(self.min_samples < 2):
            self.min_samples = 2
        # if no limit is set on max_depth then min_samples will serve as the termination criterion
        if(self.max_depth == None and len(data) < self.min_samples):
            print("returning a leaf of depth {}".format(depth))
            return self.average_of_target_col(data)
        # when max depth is provided, then both max_depth and min_samples serve as the termination criterion
        # to help max_depth functionality perform better set min_samples to 2
        elif(depth == self.max_depth or len(data) < self.min_samples):
            print("returning a leaf of depth {}".format(depth))
            return self.average_of_target_col(data)
        
        # recursive cases
        else:
            depth += 1
            self.tree_depth = max(self.tree_depth, depth)
            potential_splits = self.get_potential_splits(data)
            split_col, split_val = self.determine_best_split(data, potential_splits)
            data_above, data_below = self.split_data(data, split_col, split_val)

            nodeCondition = "{} <= {}".format(COLUMN_NAMES[split_col], split_val)
            left_child = self.build_tree(data_above,'numpy', depth)
            right_child = self.build_tree(data_below,'numpy', depth) 

            if(left_child == right_child):
                tree = left_child 
            else: 
                tree = {nodeCondition: [left_child, right_child]}
            
            return tree 


    def predict_sample(self, sample, tree):
        nodeCondition = list(tree.keys())[0]
        attribute, comparator, value = nodeCondition.split(" ")
        if(sample[attribute] <= float(value)):
            answer = tree[nodeCondition][0]
        else:
            answer = tree[nodeCondition][1]

        if isinstance(answer, dict):
            subtree = answer
            return self.predict_sample(sample, subtree)

        else: 
            return answer    


    def rmse(self, df, tree):
        """data can be both train or test"""
        predicted = df.apply(self.predict_sample, axis = 1, args = (tree,))
        actual_values = df.loc[:,"new_cases"]
        rmse = np.sqrt(np.mean((predicted - actual_values)**2))
        return rmse

    def r_squared(self, df, tree):
        predicted_values = df.apply(self.predict_sample, axis = 1, args = (tree,))
        actual_values = df.loc[:,"new_cases"]
        correlation_matrix = np.corrcoef(predicted_values, actual_values)
        correlation_xy = correlation_matrix[0,1]
        r_squared = correlation_xy**2
        return r_squared


if __name__ == "__main__":
    df = pd.read_csv('./data/covid_data_india.csv' ).iloc[:,1:]
