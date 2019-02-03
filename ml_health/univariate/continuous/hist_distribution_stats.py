"""
The Code contains functions to calcualte univariate statistics for categorical features, given a dataset.

"""

import numpy as np

def cal_hist_prob(sample, prob_dist):
    """Calculate the average probability, given the multinomial distribution parameters of each feature column.

    Usage:
        :param: sample: A dataset that is a 2D numpy array
                prob_dist: A list containing the frequency/probability distribution of values in each bin of the dataset.
                           Order of arrays in the list is same as the order of columns in sample data
        :rtype: prob_sample: A matrix, containing the probability of each sample.
                avg_prob: An array, average probability of the dataset per feature.
    """
    prob_sample = np.zeros((sample.shape[0],sample.shape[1]))
    for a in range(0,sample.shape[0]):
        # Calculate probability of each sample and save in a vector
        for b in range(0,sample.shape[1]):
            probability_map = prob_dist[b]
            for c in range(1,len(probability_map[0])):
                if( sample[a,b] >= probability_map[0][c-1] and sample[a,b] < probability_map[0][c]):
                    prob_sample[a,b] = probability_map[1][c-1]
    avg_prob = np.mean(prob_sample, axis=0)
    return prob_sample, avg_prob

def cal_hist_params(sample, num_bins):
    """Create a fixed number of bins and corresponding normalized frequency for a given dataset

    Usage:
        :param: sample: A dataset that is a 2D numpy array
                num_bins: Number of bins to create
        :rtype: prob_dist: A list containing the frequency/probability distribution of values in each bin for the dataset.
                           Order of arrays in the list is same as the order of columns in sample data
    """
    # Calculate the mean and std of the dataset
    mean = np.mean(sample, axis=0)
    standard_deviation = np.std(sample, axis=0)

    # Bin values per feature, that include +inf and -inf same as our system
    bins = np.zeros((sample.shape[1], num_bins + 1))
    bins_subset = np.zeros((sample.shape[1], num_bins-2))
    for a in range(0,sample.shape[1]):
        bins_subset[a,:] = (np.arange(mean[a] - 2*standard_deviation[a], mean[a] + 2*standard_deviation[a], (4*standard_deviation[a])/(num_bins-2)))[0:num_bins-2]
    bins[:,1:num_bins-1] = bins_subset
    bins[:,0] = -np.inf
    bins[:,num_bins-1] = mean + 2*standard_deviation
    bins[:,num_bins] = np.inf

    prob_dist = []
    for a in range(0,sample.shape[1]):
        # Determine the frequency of unique values
        counts, bins_feature = np.histogram(sample[:,a], bins[a,:])
        prob_dist.append(np.asarray((bins_feature, counts/sample.shape[0])).T)
    return prob_dist


"""
#Test Code
sample = np.array([[1, 2], [2, 3], [3, 4], [3, 4], [4, 5]])
prob_dist = cal_hist_params(sample, 4)
prob_sample, avg_prob = cal_hist_prob(sample, prob_dist)

"""
