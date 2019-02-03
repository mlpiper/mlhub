import numpy as np
from multi_variate_stats import multi_variate_statistics_create, multi_variate_statistics_inference, multi_variate_multinomial_stats_create, cal_prob_categorical, cal_multinomial_params

"""
This code simulates the new ml health case on Samsung and Yelp dataset.

"""

# Read the dataset
dataset_train = np.genfromtxt('/data-lake/poc/wix/datasets/Samsung/Wix_logistic_train_batch1_featurenames.csv', delimiter=",", skip_header=1)
dataset_test = np.genfromtxt('/data-lake/poc/wix/datasets/Samsung/Wix_logistic_test_batch1_featurenames.csv', delimiter=",", skip_header=1)

# calculate the health statistics on training side
mean_sample, var_sample, train_prob_sample, max_likelihood_sample, mean_prob_sample  = multi_variate_statistics_create(dataset_train[:,1:dataset_train.shape[1]])

# calculate the health statistics on inference side
prob_test_sample, mean_prob_test  = multi_variate_statistics_inference(dataset_test, mean_sample, var_sample, max_likelihood_sample)

print('Mean probability in training sampes is: ', mean_prob_sample, 'mean probability in test samples is: ', mean_prob_test)

# Read the dataset
dataset_train = np.genfromtxt('/data-lake/poc/wix/datasets/Yelp/Wix_rf_train_batch1_featurenames.csv', delimiter=",", skip_header=1)
dataset_test = np.genfromtxt('/data-lake/poc/wix/datasets/Yelp/Wix_rf_test_batch1_featurenames.csv', delimiter=",", skip_header=1)


hash_map,test_prob_sample,detect_train_prob_sample,nr_samples  = multi_variate_multinomial_stats_create(dataset_train[:,1:dataset_train.shape[1]])

test_prob = cal_prob_categorical(dataset_test, hash_map,dataset_train.shape[0])

print('Training probability is: ', np.mean(test_prob))

# Since the above did not work out as we intended, we will go ahead with univariate distribution for categorical features to begin with

prob_dist = cal_multinomial_params(dataset_train[:,1:dataset_train.shape[1]])

# The above gives us individual probabilities, we can calculate the average training probability for column.

prob_sample_train, avg_prob_train = cal_multinomial_prob(dataset_train[:,1:dataset_train.shape[1]],prob_dist)

prob_sample_test, avg_prob_test = cal_multinomial_prob(dataset_test,prob_dist)

