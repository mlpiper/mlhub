"""
This code contains functions to calculate univariate statistics for categorical features, given a dataset.

"""

import numpy as np

def cal_multinomial_prob(sample, prob_dist):
    """Calculate the average probability, given the multinomial distribution parameters of each feature column.

    Usage:
        :param: sample: A dataset that is a 2D numpy array
                prob_dist: A list containing the probability distribution of categories in each column of the sample.
                           Order of arrays in the list is same as the order of columns in sample data
        :rtype: prob_sample: A matrix, containing the probability of each sample.
                avg_prob: An array, average probability of the dataset per feature.
    """
    prob_sample = np.zeros((sample.shape[0],sample.shape[1]))
    for a in range(0,sample.shape[0]):
        # Calculate probability of each sample and save in a vector
        for b in range(0,sample.shape[1]):
            probability_map = prob_dist[b]
            for c in range(0,len(probability_map)):
                if(probability_map[c][0] == sample[a,b]):
                    prob_sample[a,b] = probability_map[c][1]
    avg_prob = np.mean(prob_sample, axis=0)
    return prob_sample, avg_prob

def cal_multinomial_params(sample):
    """Calculate the probability of each category in each column, assuming multi-nomial distribution.

    Usage:
        :param: sample: A dataset that is a 2D numpy array
        :rtype: prob_dist: A list containing the probability distribution of categories in each column of the sample.
                           Order of arrays in the list is same as the order of columns in sample data
    """
    prob_dist = []
    for a in range(0,sample.shape[1]):
        # Determine the frequency of unique values
        unique, counts = np.unique(sample[:,a], return_counts=True)
        # .T in the line below represents a transpose.
        prob_dist.append(np.asarray((unique, counts/sample.shape[0])).T)
    return prob_dist

def cal_multinomial_inference_params(sample, prob_dist):
    """Calculate the normalized histogram of each category in the same order as training.

    Usage:
        :param: sample: A dataset that is a 2D numpy array
                prob_dist: A list containing the probability distribution of categories in each column of the sample.
                           Order of arrays in the list is same as the order of columns in sample data
        :rtype: histogram: A list containing the probability distribution of categories in each column of the sample.
                           Order of arrays in the list is same as the order of columns in sample data
    """
    histogram = []
    for a in range(0,sample.shape[1]):
        # Determine the frequency of unique values
        unique, counts = np.unique(sample[:,a], return_counts=True)
        unique = unique.tolist()
        # Arrange them in the same order as training
        train_unique = [x[0] for x in prob_dist[a]]
        unique_infer = []
        counts_infer = []
        for a in range(0,len(train_unique)):
            # if it exists, append the value and count, else append the value as 0
            if(unique.count(train_unique[a])):
                idx = unique.index(train_unique[a])
                unique_infer.append(unique[idx])
                counts_infer.append(counts[idx])
            else:
                unique_infer.append(train_unique[a])
                counts_infer.append(0)

        histogram.append(np.asarray((np.array(unique_infer), np.array(counts_infer)/sample.shape[0])).T)
    return histogram

"""
#Test Code
sample = np.array([[1, 2], [2, 3], [3, 4], [3, 4], [4, 5]])
prob_dist = cal_multinomial_params(sample)
prob_sample, avg_prob = cal_multinomial_prob(sample, prob_dist)

produced the correct results.


prob_sample =
array([[ 0.2,  0.2],
       [ 0.2,  0.2],
       [ 0.4,  0.4],
       [ 0.4,  0.4],
       [ 0.2,  0.2]])


avg_prob = array([ 0.28,  0.28])

"""
