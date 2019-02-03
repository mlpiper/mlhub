"""
Calculate the data-deviation score for top N features in TELCO datasets.
Top features are determined by random forest algorithm.
"""

import sys
import numpy as np
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0,'../../..')
import ml_health.univariate.univariate_health_calculator as hc

# Read the TELCO dataset for all loads
all_path = ["periodic_load", "flashcrowd_load", "linear_increase", "constant_load", "poisson_load"]
telco_path = '/data-lake/ml-prototypes/classification/ml/realm-im2015-vod-traces/X_SAR/'

for train_path in all_path:
    for test_path in all_path:
        train_load_path = telco_path + train_path + "/"
        test_load_path = telco_path + test_path + "/"
        train_load = np.genfromtxt(train_load_path + "Train.csv", delimiter=',')
        validate_load = np.genfromtxt(train_load_path + "Validate.csv", delimiter=',')
        # Combine the train and validation data (50%)
        train_load = np.concatenate((train_load, validate_load))
        # Test data (50%)
        test_load = np.genfromtxt(test_load_path + "Test.csv", delimiter=',')

        # Fit the Primary algorithm so that feature importance can be utilized
        algorithm = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
        algorithm.fit(train_load[:,:-1],train_load[:,-1])

        # Let the number of top features used be 3
        N=3
        # Retrieve the indices of top N features
        feature_idx = np.argpartition(algorithm.feature_importances_, (-1)*N)[(-1)*N:]

        # Fit the ml-health model on training data, use only the top N features
        distribution = hc.MlHealth()
        distribution.fit(train_load[:,feature_idx])

        # Calculate the similarity score on test data for the top N features
        distribution_score = distribution.score(test_load[:,feature_idx])
        print("Similarity score when trained with load ", train_path ," and tested on load ", test_path," :", distribution_score)


