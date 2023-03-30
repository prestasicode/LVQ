
from matplotlib import pyplot as plt
from matplotlib import patches as patches

import numpy as np


class LVQ:
    
    def __init__(self, net_x_dim, net_y_dim, num_features):
        self.num_features = num_features
        self.net_x_dim = net_x_dim
        self.net_y_dim = net_y_dim
        
    def train(self, data, ref_vec, num_epochs=100, init_learning_rate=0.01):
        
        num_rows = data.shape[0]
        indices = np.arange(num_rows)
        
        # visualization
        if self.num_features == 3:
            fig = plt.figure()
        else:
            fig = None
        # for (epoch = 1,..., Nepochs)
        for i in range(1, num_epochs + 1):
            
            # interpolate new values for α(t) and σ (t)
            learning_rate = self.decay_learning_rate(init_learning_rate, i, num_epochs)
            
            for record in indices:
                row_t = data[record, :]
                # find its Best Matching Unit
                bmu, bmu_idx = self.find_bmu(row_t, ref_vec)
                # for (k = 1,..., K)
                
                weight = ref_vec[bmu_idx, :-1]
                if bmu[-1] == row_t[-1]:
                    new_w = weight + (learning_rate * (row_t[:-1] - weight))
                else:
                    new_w = weight - (learning_rate * (row_t[:-1] - weight))
                ref_vec[bmu_idx, :-1] = new_w
            
            # visualization
            vis_interval = int(num_epochs/10)
            if i % vis_interval == 0:
                print("LVQ training epoches %d" % i)
                print("learning rate ", learning_rate)
                print("weights: ", ref_vec)
                print("-------------------------------------")
        if fig is not None:
            plt.show()
    
    def find_bmu(self, row_t, ref_vec):
        
        bmu_idx = 0
        # set the initial minimum distance to a huge number
        min_dist = np.iinfo(np.int).max
        # calculate the high-dimensional distance between each neuron and the input
        # for (k = 1,..., K)
        for x in range(ref_vec.shape[0]):
            weight_k = ref_vec[x, :-1]
            sq_dist = np.sum((weight_k - row_t[:-1]) ** 2)
            
            # compute winning node c using Eq. (2)
            if sq_dist < min_dist:
                min_dist = sq_dist
                bmu_idx = x
        # get vector corresponding to bmu_idx
        bmu = ref_vec[bmu_idx, :]
        return bmu, bmu_idx

    def predict(self, data, ref_vec):
        # find its Best Matching Unit
        bmu, bmu_idx = self.find_bmu(data, ref_vec)
        return bmu[-1]

    def decay_learning_rate(self, initial_learning_rate, iteration, num_iterations):
        return initial_learning_rate * np.exp(-iteration / num_iterations)
