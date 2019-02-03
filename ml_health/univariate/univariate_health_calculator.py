"""
This code creates a class called ml_health similar to other python scikit-learn algorithms
It provides the following functions:
    fit: Fit the training data to generate the necessary parameters
    score: Generate a health score, based on the parameters
"""
import numpy as np
import sys
sys.path.insert(0,'../..')
import ml_health.univariate.continuous.hist_distribution_stats as hds
import ml_health.univariate.categorical.multinomial_distribution_stats as mds

from scipy.stats import entropy
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import pdist


class MlHealth:
    def __init__(self):
        self._prob_dist_categorical = []
        self._prob_dist_continuous = []
        self._average_prob = []
        self._cat_index = []
    """
    This function calculates the histogram/frequency for categorical and continuous features.
    It also calculated the average score per feature.
    """
    def fit(self,training_dataset):
        # Detect continuous and categorical features
        likely_cat = np.ones((training_dataset.shape[1],1))
        cat_features_subset = []
        con_features_subset = []
        for a in range(0,training_dataset.shape[1]):
            likely_cat[a] = len(np.unique(training_dataset[:,a])) < 30
            if(likely_cat[a] == 1):
                cat_features_subset.append(training_dataset[:,a])
            else:
                con_features_subset.append(training_dataset[:,a])
        # Save this index as the same subset should be made during inference
        self._cat_index = likely_cat

        # Create a numpy array categorical and continuous features
        cat_features_subset = np.array(cat_features_subset).T
        con_features_subset = np.array(con_features_subset).T

        avg_cat_prob = []
        avg_con_prob = []
        # Calculate the health parameters
        if(len(cat_features_subset)):
            prob_dist = mds.cal_multinomial_params(cat_features_subset)
            prob_sample, avg_cat_prob = mds.cal_multinomial_prob(cat_features_subset, prob_dist)
            self._prob_dist_categorical = prob_dist
        if(len(con_features_subset)):
            prob_dist = hds.cal_hist_params(con_features_subset, 13)
            prob_sample, avg_con_prob = hds.cal_hist_prob(con_features_subset, prob_dist)
            self._prob_dist_continuous = prob_dist

        # Save the average health score per feature
        if(len(avg_cat_prob) and len(avg_con_prob)):
            self._average_prob = np.concatenate((avg_cat_prob, avg_con_prob),axis=0)
        elif(len(avg_cat_prob)):
            self._average_prob = avg_cat_prob
        elif(len(avg_con_prob)):
            self._average_prob = avg_con_prob

    def score(self,inference_dataset,N=1):
        if(len(self._average_prob)==0):
            sys.exit("fit was not called and no model paramters were generated.")
        cat_features_subset = []
        con_features_subset = []

        # Create subsets of categorical and continuous features based on training informaiton
        for a in range(0,inference_dataset.shape[1]):
            if(self._cat_index[a]):
                cat_features_subset.append(inference_dataset[:,a])
            else:
                con_features_subset.append(inference_dataset[:,a])
        # Create a subset of categorical features
        cat_features_subset = np.array(cat_features_subset).T
        con_features_subset = np.array(con_features_subset).T
        avg_cat_prob = []
        avg_con_prob = []

        if(len(self._prob_dist_categorical)):
            prob_sample, avg_cat_prob = mds.cal_multinomial_prob(cat_features_subset, self._prob_dist_categorical)

        if(len(self._prob_dist_continuous)):
            prob_sample, avg_con_prob = hds.cal_hist_prob(con_features_subset, self._prob_dist_continuous)

        if(len(avg_cat_prob) and len(avg_con_prob)):
            average_prob = np.concatenate((avg_cat_prob, avg_con_prob),axis=0)
        elif(len(avg_cat_prob)):
            average_prob = avg_cat_prob
        elif(len(avg_con_prob)):
            average_prob = avg_con_prob

        scores = average_prob/self._average_prob
        idx = np.argpartition(scores, N)
        score = np.mean(scores[idx[:N]])
        return score


    def full_score(self,inference_dataset):
        if(len(self._average_prob)==0):
            sys.exit("fit was not called and no model paramters were generated.")
        cat_features_subset = []
        con_features_subset = []

        # Create subsets of categorical and continuous features based on training informaiton
        for a in range(0,inference_dataset.shape[1]):
            if(self._cat_index[a]):
                cat_features_subset.append(inference_dataset[:,a])
            else:
                con_features_subset.append(inference_dataset[:,a])
        # Create a subset of categorical features
        cat_features_subset = np.array(cat_features_subset).T
        con_features_subset = np.array(con_features_subset).T
        avg_cat_prob = []
        avg_con_prob = []

        if(len(self._prob_dist_categorical)):
            prob_cat_sample, avg_cat_prob = mds.cal_multinomial_prob(cat_features_subset, self._prob_dist_categorical)

        if(len(self._prob_dist_continuous)):
            prob_con_sample, avg_con_prob = hds.cal_hist_prob(con_features_subset, self._prob_dist_continuous)

        if(len(avg_cat_prob) and len(avg_con_prob)):
            prob = np.concatenate((prob_cat_sample, prob_con_sample),axis=1)
        elif(len(avg_cat_prob)):
            prob = prob_cat_sample
        elif(len(avg_con_prob)):
            prob = prob_con_sample

        score = prob/self._average_prob
        return score

    def other_scores(self,inference_dataset):
        """
        This function calculates and returns, RMSE, KL Divergence, Wasserstein score.
        """
        if(len(self._average_prob)==0):
            sys.exit("fit was not called and no model paramters were generated.")
        cat_features_subset = []
        con_features_subset = []

        # Create subsets of categorical and continuous features based on training informaiton
        for a in range(0,inference_dataset.shape[1]):
            if(self._cat_index[a]):
                cat_features_subset.append(inference_dataset[:,a])
            else:
                con_features_subset.append(inference_dataset[:,a])
        # Create a subset of categorical features
        cat_features_subset = np.array(cat_features_subset).T
        con_features_subset = np.array(con_features_subset).T

        if(len(self._prob_dist_categorical)):
            prob_dist_inference = mds.cal_multinomial_inference_params(cat_features_subset, self._prob_dist_categorical)
            temp = self._prob_dist_categorical
            RMSE_cat = np.zeros((len(self._prob_dist_categorical),1))
            kl_cat = np.zeros((len(self._prob_dist_categorical),1))
            wasserstein_cat = np.zeros((len(self._prob_dist_categorical),1))
            for a in range(0,len(self._prob_dist_categorical)):
                prob_training_sample = np.array([x[1] for x in temp[a]])
                prob_cat_sample = np.array([x[1] for x in prob_dist_inference[a]])
                RMSE_cat[a] = np.sqrt(((prob_training_sample - prob_cat_sample) ** 2).mean())
                kl_cat[a] = entropy(prob_training_sample, prob_cat_sample)
                wasserstein_cat[a] = wasserstein_distance(prob_training_sample, prob_cat_sample)

        if(len(self._prob_dist_continuous)):
            temp = self._prob_dist_continuous
            RMSE_con = np.zeros((len(self._prob_dist_continuous),1))
            kl_con = np.zeros((len(self._prob_dist_continuous),1))
            wasserstein_con = np.zeros((len(self._prob_dist_continuous),1))
            for a in range(0,len(self._prob_dist_continuous)):
                bins_training = temp[a][0]
                prob_training_sample = np.array(temp[a][1])
                prob_cont_sample, unused = np.histogram(con_features_subset[:,a], bins_training)
                prob_cont_sample = prob_cont_sample/len(con_features_subset)
                RMSE_con[a] = np.sqrt(((prob_training_sample - prob_cont_sample) ** 2).mean())
                kl_con[a] = entropy(prob_training_sample, prob_cont_sample)
                wasserstein_con[a] = wasserstein_distance(prob_training_sample, prob_cont_sample)
        if(len(cat_features_subset) and len(con_features_subset)):
            RMSE = np.concatenate((RMSE_cat, RMSE_con),axis=0)
            kl = np.concatenate((kl_cat, kl_con), axis=0)
            wasserstein = np.concatenate((wasserstein_cat, wasserstein_con), axis=0)
        elif(len(cat_features_subset)):
            RMSE = RMSE_cat
            kl = kl_cat
            wasserstein = wasserstein_cat
        elif(len(con_features_subset)):
            RMSE = RMSE_con
            kl = kl_con
            wasserstein = wasserstein_con

        return RMSE, kl, wasserstein

"""
Test
sample_1 = np.array([[1, 2], [2, 3], [3, 4], [3, 4], [4, 5]])
ml_health = MlHealth()
ml_health.fit(sample_1)
score = ml_health.score(sample_1) # score should be 1

sample_2 = np.random.randn(1000,10)
ml_health = MlHealth()
ml_health.fit(sample_2)
score = ml_health.score(sample_2) # score should be 1

sample_3 = np.random.randint(0,20,(1000,10))
score = ml_health.score(sample_3)  # score should be a low value < 0.5

sample_4 = np.concatenate((sample_2,sample_3),axis=1)
ml_health = MlHealth()
ml_health.fit(sample_4)
score = ml_health.score(sample_4)  # score should be 1

"""
