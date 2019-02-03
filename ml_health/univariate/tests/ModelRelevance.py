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

def balanceDataset(features, labels):
    index_good = np.where(labels == 1)
    index_bad = np.where(labels == 0)
    num_samples = len(index_good[0])
    bad_samples = np.squeeze(features[index_bad[0][0:num_samples], :])
    random_index_bad_samples = np.random.randint(0, bad_samples.shape[0], size=(num_samples,))
    samples_bad = bad_samples[random_index_bad_samples, :]
    sampled_features = np.concatenate((np.squeeze(features[index_good[0][0:num_samples], :]), samples_bad))
    sampled_labels = np.concatenate(
        (labels[index_good[0][0:num_samples]], labels[index_bad[0][random_index_bad_samples]]))
    return sampled_features, sampled_labels

for train_path in all_path:
    for test_path in all_path:
        train_load_path = telco_path + train_path + "/"
        test_load_path = telco_path + test_path + "/"
        train_load = np.genfromtxt(train_load_path + "Train.csv", delimiter=',')
        validate_load = np.genfromtxt(train_load_path + "Validate.csv", delimiter=',')
        # Combine the train and validation data (50%)
        # train_load = np.concatenate((train_load, validate_load))
        # Test data (50%)
        test_load = np.genfromtxt(test_load_path + "Test.csv", delimiter=',')

        # Fit the Primary algorithm so that feature importance can be utilized
        algorithm = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
        algorithm.fit(train_load[:,:-1],train_load[:,-1])

        # Let the number of top features used be 3
        N=3
        # Retrieve the indices of top N features
        feature_idx = np.argpartition(algorithm.feature_importances_, (-1)*N)[(-1)*N:] - 1

        # Fit the ml-health model on training data, use only the top N features
        distribution = hc.MlHealth()
        distribution.fit(train_load[:,feature_idx])

        # Fit the watchdog model on validation data
        predictions_validation = algorithm.predict(validate_load[:,:-1])
        labels_watchdog_validation = (predictions_validation == validate_load[:,-1])
        balanced_validation_features, balanced_validation_labels = balanceDataset(validate_load[:,:-1], labels_watchdog_validation)
        watchDogModel = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
        watchDogModel.fit(balanced_validation_features[:,feature_idx], balanced_validation_labels)

        # Calculate the similarity score on test data for the top N features
        distribution_score = distribution.score(test_load[:,feature_idx])
        print("Similarity score when trained with load ", train_path ," and tested on load ", test_path," :", distribution_score)

        # Predict test model accuracy
        accuracy = algorithm.score(test_load[:,:-1],test_load[:,-1])
        print("Prediction score when trained with load ", train_path ," and tested on load ", test_path," :", accuracy)

        # Watchdog model predicted accuracy
        labels_watchdog_test = (algorithm.predict(test_load[:,:-1]) == test_load[:,-1])
        watchdog_score = watchDogModel.score(test_load[:,feature_idx], labels_watchdog_test)
        watchdog_prediction = np.mean(watchDogModel.predict(test_load[:,feature_idx]))
        print("Watchdog prediction when trained with load ", train_path ," and tested on load ", test_path," :", watchdog_prediction)
          
 
