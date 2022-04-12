import pandas as pd
import numpy as np 
import scipy 
import matplotlib.pyplot as plt 
import seaborn as sns 
from tqdm import tqdm
import os
# no need to implement normalization or standardization here itself. will 
# put it in the main.py itself.
class KMeansClustering:
    def __init__(self, K, initializer = 'random'):
        """
        in model.fit() give the dataframe object directly
        K (int)
        intializer = 'random' or 'KMeans++'
        """
        self.K = K


    def random_initializer(self, shape):
        # shape should be a numpy array of shape [471, 6]
        return np.random.rand(self.K,shape[1])
        # returning K random centroids

    def fit(self, X, max_iter = 20):
        """
        for our purpose X consists of only the mobility points.
        """
        try: 
            X = X.to_numpy()
        except Exception as e:
            print("could not convert input data to numpy array")
        centroids = self.random_initializer(X.shape)
        class_index = None
        log_centroids = np.zeros(shape = [self.K, max_iter, X.shape[1]])
        for iter in range(max_iter):

            distances = self.euc_dist(centroids, X)
            class_index = np.argmin(distances, axis = 1, keepdims=True)
            for j in range(self.K):
                centroids[j] = np.sum(X*(class_index==j), axis= 0)/X.shape[0]
                log_centroids[j][iter] = centroids[j]

        self.plot_transition(log_centroids)
        return class_index 


    def euc_dist(self, centroids, samples):
        """
        centroids: shape [K, 6]
        dataset: shape [N, 6]
        distances : shape [N, K]
        """
        dist_vec = np.zeros(shape = [self.K, samples.shape[0]])
        for i in range(self.K):
            dist_vec[i] = np.sqrt(np.sum(np.square(centroids[i] - samples), axis = 1)).reshape(1,samples.shape[0])     
        return dist_vec.T
        
    def plot_transition(self, log_centroids):
        fig, ax = plt.subplots(self.K, 1)
        for i in range(self.K):
            ax[i].plot(log_centroids[i])
            ax[i].set_title(f"K = {i + 1}")
        
        if not os.path.exists("./plots"):
            os.makedirs("./plots")
        plt.savefig(f"./plots/Transitions_in_K={self.K}.png")



if __name__ == "__main__":
    KMeans = KMeansClustering(K = 3)
    df = pd.read_csv('./data/covid_data_india.csv', index_col = None)
    df = df.iloc[:, 1:7]
    classes = KMeans.fit(df)
    print(pd.Series(classes.flatten()).value_counts())