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
    def __init__(self, K, initializer = 'KMeans++'):
        """
        in model.fit() give the dataframe object directly
        K (int)
        intializer = 'random' or 'KMeans++'
        """
        self.K = K
        self.initr = initializer
        self.centroids = None 


    def random_initializer(self, shape):
        # shape should be a numpy array of shape [471, 6]
        return 100*np.random.rand(self.K,shape[1])
        # returning K random centroids

    def KMeansPlusPlus(self, X, K):
        centroids = []
        num_samples = X.shape[0]
        features = X.shape[1]
        ind = np.random.randint(0, num_samples, 1)
        centroids.append(X[ind].flatten())
        for i in range(self.K - 1):
            distances = np.zeros(shape = [num_samples, len(centroids)])
            for j in range(len(centroids)):
                cen = np.array(centroids[i]).reshape(1, features)
                distances[:,i] = np.sqrt(np.sum((X-cen)**2, axis = 1))
            min_dist_from_nearest_cen = np.min(distances, axis = 1)
            next_centroid = X[np.argmax(min_dist_from_nearest_cen)]
            centroids.append(next_centroid)
        
        return np.array(centroids)

    def fit(self, X, max_iter = 20):
        """
        for our purpose X consists of only the mobility points.
        returns classes and inertia value for this fit
        """
        try: 
            X = X.to_numpy()
        except Exception as e:
            print("could not convert input data to numpy array")

        if(self.initr == 'KMeans++'):
            centroids = self.KMeansPlusPlus(X, self.K)
        else:
            centroids = self.random_initializer(X.shape)
        class_index = None
        log_centroids = np.zeros(shape = [self.K, max_iter, X.shape[1]])
        num_samples = X.shape[0]
        features = X.shape[1]
        for iter in range(max_iter):
            distances = np.zeros(shape = [num_samples, self.K])
            for i in range(self.K):
                cen = centroids[i].reshape(1,features)
                distances[:,i] = np.sqrt(np.sum((X-cen)**2, axis=1))

            class_index = np.argmin(distances, axis = 1, keepdims=True)
            
            for j in range(self.K):
                num_samples_in_class_j = np.sum(class_index==j)
                if(num_samples_in_class_j == 0):
                    continue
                      
                centroids[j] = np.sum(X*(class_index==j), axis= 0)/num_samples_in_class_j
                log_centroids[j][iter] = centroids[j]

        # self.plot_transition(log_centroids)
        self.centroids = centroids
        inertia = self.calculate_inertia(X, centroids, class_index)
        print(f"Inertia for K = {self.K} = {inertia}")
        return class_index , inertia

    def calculate_inertia(self,data, centroids, classes):
        inertia = 0
        classes = classes.reshape(-1)
        for i in  range(self.K):
            cen = centroids[i]
            samples = data[np.where(classes==i)]
            dist = np.sqrt(np.sum((samples - cen)**2, axis = 1))
            inertia += np.sum(dist)
        return inertia

    def predict(self, point):
        point = point.reshape(-1,self.K)
        centroids = self.centroids.reshape(self.K, -1)
        return np.argmin(np.sum((centroids - point)**2, axis = 1, keepdims = 1))
    

    def plot_transition(self, log_centroids):
        fig, ax = plt.subplots(self.K, 1)
        for i in range(self.K):
            ax[i].plot(log_centroids[i])
            ax[i].set_title(f"K = {i + 1}")
        
        if not os.path.exists("./plots"):
            os.makedirs("./plots")
        plt.savefig(f"./plots/Transitions_in_K={self.K}.png")
        # plt.close()



if __name__ == "__main__":
    df = pd.read_csv('./data/covid_data_india.csv', index_col = None)
    df = df.iloc[:, 1:7]
    inertia_log = []
    for i in range(1,11):
        KMeans = KMeansClustering(K = i)
        classes, inertia = KMeans.fit(df)
        print(pd.Series(classes.flatten()).value_counts())
        inertia_log.append(inertia)
    print(inertia_log)
    
    # classes = KMeans.fit(df)