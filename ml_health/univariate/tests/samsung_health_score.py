"""
This code generates data for 2 plots
plot 1: For logistic regression, it generates data to plot the accuracy vs health-score graph for
        5 datasets, where noise is added to datasets.
plot 2: For random forest, it generates data to plot the accuracy vs health-score graph for
        5 datasets, where noise is added to datasets.
The datasets are divided into two sets, noise is the addition of samples that were from a cluster
different from the one trained  by the algorithm.
As a result the accuracy of algorithm decreases. We look at the ml-health score in such a case.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import sys
sys.path.insert(0,'../../..')
import ml_health.experiments.machineLearning.datasetMix as datasetMix
import ml_health.univariate.univariate_health_calculator as hc
from sklearn.cluster import KMeans
import ml_health.experiments.machineLearning.datasetMix as dm
import ml_health.experiments.machineLearning.randomNoise as rn

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


def calcualteStats(train_load, validate_load, test_load):
    # Fit the Primary algorithm so that feature importance can be utilized
    algorithm = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
    algorithm.fit(train_load[:,1:],train_load[:,0])

    # Let the number of top features used be 3
    N=10
    # Retrieve the indices of top N features
    feature_idx = np.argpartition(algorithm.feature_importances_, (-1)*N)[(-1)*N:]
    
    # Fit the ml-health model on training data, use only the top N features
    distribution = hc.MlHealth()
    distribution.fit(train_load[:,feature_idx])

    # Fit the watchdog model on validation data
    predictions_validation = algorithm.predict(validate_load[:,1:])
    labels_watchdog_validation = (predictions_validation == validate_load[:,0])
    balanced_validation_features, balanced_validation_labels = balanceDataset(validate_load[:,1:], labels_watchdog_validation)
    watchDogModel = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
    watchDogModel.fit(balanced_validation_features[:,feature_idx], balanced_validation_labels)


    # Calculate the similarity score on test data for the top N features
    distribution_score = distribution.score(test_load[:,feature_idx])
    # Predict test model accuracy
    accuracy = algorithm.score(test_load[:,1:],test_load[:,0])

    # Watchdog model predicted accuracy
    labels_watchdog_test = (algorithm.predict(test_load[:,1:]) == test_load[:,0])
    watchdog_score = watchDogModel.score(test_load[:,feature_idx], labels_watchdog_test)
    watchdog_prediction = np.mean(watchDogModel.predict(test_load[:,feature_idx]))
    return accuracy, distribution_score, watchdog_prediction

# Path to datasets on data-lake
path = "/data-lake/ml-prototypes/classification/ml/"
np.random.seed(45)
# Read training and test datasets
Train = np.genfromtxt(path + 'samsung/original/train/samsung_train.csv', dtype = float, delimiter=",")
Test = np.genfromtxt(path + 'samsung/original/test/samsung_test.csv', dtype = float, delimiter=",")

# 1. Add noise to test data
# Add noise to test data, label is the first column
Test_noisy_random = np.copy(Test)
Test_noisy_random[:,1:] = rn.randomNoise(Test_noisy_random[:,1:], 5)

# Mix up the data
half_samples = np.int(len(Train)/2)
num_points = 10
num_samples = len(Test)
for a in range(0, num_points):
    ratio = a/(num_points-1)
    Noisy_Test = dm.randomSample(Test,Test_noisy_random,ratio,num_samples)
    accuracy, distribution_score, watchdog_prediction = calcualteStats(Train[0:half_samples,:], Train[half_samples:,:], Noisy_Test)
    print("noise, accuracy, similarity, watchdog", ratio, accuracy, distribution_score, watchdog_prediction)



# 2. Cluster the data
algorithm = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
algorithm.fit(Train[:,1:],Train[:,0])

# Let the number of top features used be 3
N=50
# Retrieve the indices of top N features
feature_idx = np.argpartition(algorithm.feature_importances_, (-1)*N)[(-1)*N:]
num_samples, num_features = Train.shape
# Cluster the training data
kmeans = KMeans(n_clusters=2,n_init=100).fit(Train[:, feature_idx])
#kmeans = KMeans(n_clusters=2,n_init=100,max_iter=100).fit(Train[:,1:10])
labels_train = kmeans.predict(Train[:, feature_idx])
#labels_train = kmeans.predict(Train[:,1:10])
Train_set1 = Train[labels_train==0,:]
Mix = 400

Train_set2 = np.concatenate((Train[labels_train==1,:],Train_set1[0:Mix,:]))
Train_set1 = np.concatenate((Train_set1,Train_set2[0:Mix,:]))
#Train_set2 = Train_set2[Mix:,:]

labels_test = kmeans.predict(Test[:, feature_idx])
Test_set1 = Test[labels_test==0,:]
Test_set2 = Test[labels_test==1,:]


# Train on Train_set1 and Test_set1
np.random.shuffle(Train_set1)
num_samples, num_features = Train_set1.shape
half_samples = np.int(num_samples/2)

accuracy, distribution_score, watchdog_prediction = calcualteStats(Train_set1[0:half_samples,:], Train_set1[half_samples:,:], Test_set1)
print("accuracy, similarity, watchdog",accuracy, distribution_score, watchdog_prediction)

# Train on Train_set1 and Test_set2
accuracy, distribution_score, watchdog_prediction = calcualteStats(Train_set1[0:half_samples,:], Train_set1[half_samples:,:], Test_set2)

print("accuracy, similarity, watchdog",accuracy, distribution_score, watchdog_prediction)
# Train on Train_set2 and Test_set1
np.random.shuffle(Train_set2)
num_samples, num_features = Train_set2.shape
half_samples = np.int(num_samples/2)
accuracy, distribution_score, watchdog_prediction = calcualteStats(Train_set2[0:half_samples,:], Train_set2[half_samples:,:], Test_set1)
print("accuracy, similarity, watchdog",accuracy, distribution_score, watchdog_prediction)

# Train on Train_set2 and Test_set2
accuracy, distribution_score, watchdog_prediction = calcualteStats(Train_set2[0:half_samples,:], Train_set2[half_samples:,:], Test_set2)
print("accuracy, similarity, watchdog",accuracy, distribution_score, watchdog_prediction)


# 3. Create a dataset with high percentage of bad predictions
algorithm = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
algorithm.fit(Train[:,1:],Train[:,0])

Test_set2 = np.squeeze(Test[np.where(Test[:,0] != algorithm.predict(Test[:,1:])),:])
Test_set1 = np.squeeze(Test[np.where(Test[:,0] == algorithm.predict(Test[:,1:])),:])

half_samples = np.int(len(Train)/2)
accuracy, distribution_score, watchdog_prediction = calcualteStats(Train[0:half_samples,:], Train[half_samples:,:], Test_set1)
print("accuracy, similarity, watchdog for good set",accuracy, distribution_score, watchdog_prediction)
accuracy, distribution_score, watchdog_prediction = calcualteStats(Train[0:half_samples,:], Train[half_samples:,:], Test_set2)
print("accuracy, similarity, watchdog for bad set",accuracy, distribution_score, watchdog_prediction)

