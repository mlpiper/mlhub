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

# Path to datasets on data-lake
path = "/data-lake/ml-prototypes/classification/ml/"
# Number of points in the plots per curve
num_points = 10
num_datasets = 5
# Number of minimum points to consider
N = 1

# Initialize the variable to plot
accuracyLR_train = np.zeros((num_datasets, num_points))
accuracyRF_train = np.zeros((num_datasets, num_points))
accuracyLR_test = np.zeros((num_datasets, num_points))
accuracyRF_test = np.zeros((num_datasets, num_points))
mlHealth_cluster_train = np.zeros((num_datasets, num_points))
mlHealth_cluster_test = np.zeros((num_datasets, num_points))

# Initialize the logistic regression and random forest classification algorithms
clf_logistic = LogisticRegression(C=0.025, multi_class='multinomial', penalty='l1', solver='saga', tol=0.1)
clf_rf = RandomForestClassifier(max_depth=20, random_state=0, n_estimators=50)
kmeans = KMeans(n_clusters = 2, n_init = 50)

# First dataset is Samsung
plot_index = 0

# Read training and test datasets
Train = np.genfromtxt(path + 'samsung/original/train/samsung_train.csv', dtype = float, delimiter=",")
Test = np.genfromtxt(path + 'samsung/original/test/samsung_test.csv', dtype = float, delimiter=",")

num_samples, num_features = Train.shape
# Cluster the training data
labels_train = kmeans.fit_predict(Train[:, 1:num_features])
Train_set1 = Train[labels_train==0,:]
Train_set2 = Train[labels_train==1,:]
labels_test = kmeans.predict(Test[:,1:num_features])
Test_set1 = Test[labels_test==0,:]
Test_set2 = Test[labels_test==1,:]

clf_logistic.fit(Train_set1[:,1:num_features], Train_set1[:,0])
clf_rf.fit(Train_set1[:,1:num_features], Train_set1[:,0])

# Fit the ml-health model
clf_health_samsung = hc.MlHealth()
clf_health_samsung.fit(Train_set1[:,1:num_features])

for a in range(0, num_points):
    ratio = a/(num_points-1)
    train_data_cluster_noise = datasetMix.randomSample(Train_set1, Train_set2, ratio, num_samples)
    test_data_cluster_noise = datasetMix.randomSample(Test_set1, Test_set2, ratio, num_samples)
    accuracyLR_train[plot_index,a] = clf_logistic.score(train_data_cluster_noise[:,1:num_features], train_data_cluster_noise[:,0])
    accuracyRF_train[plot_index,a] = clf_rf.score(train_data_cluster_noise[:,1:num_features], train_data_cluster_noise[:,0])
    accuracyLR_test[plot_index,a] = clf_logistic.score(test_data_cluster_noise[:,1:num_features], test_data_cluster_noise[:,0])
    accuracyRF_test[plot_index,a] = clf_rf.score(test_data_cluster_noise[:,1:num_features], test_data_cluster_noise[:,0])
    mlHealth_cluster_train[plot_index,a] = clf_health_samsung.score(train_data_cluster_noise[:,1:num_features],N)
    mlHealth_cluster_test[plot_index,a] = clf_health_samsung.score(test_data_cluster_noise[:,1:num_features],N)
    print("Samsung iteration: ",a)

print("accuracyLR_train value for samsung dataset is: {0}".format(accuracyLR_train[plot_index,:]))
print("accuracyRF_train value for samsung dataset is: {0}".format(accuracyRF_train[plot_index,:]))
print("accuracyLR_test value for samsung dataset is: {0}".format(accuracyLR_test[plot_index,:]))
print("accuracyRF_test value for samsung dataset is: {0}".format(accuracyRF_test[plot_index,:]))
print("mlHealth_cluster_train for samsung dataset is: {0}".format(mlHealth_cluster_train[plot_index,:]))
print("mlHealth_cluster_test for samsung dataset is: {0}".format( mlHealth_cluster_train[plot_index,:]))

# Second dataset is Yelp
plot_index = 1

# Read training and test datasets
Train = np.genfromtxt(path + 'yelp/original/train/yelp_train.csv', dtype = float, delimiter=",")
Test = np.genfromtxt(path + 'yelp/original/test/yelp_test.csv', dtype = float, delimiter=",")

num_samples, num_features = Train.shape
# Cluster the training data
labels_train = kmeans.fit_predict(Train[:, 1:num_features])
Train_set1 = Train[labels_train==0,:]
Train_set2 = Train[labels_train==1,:]
labels_test = kmeans.predict(Test[:,1:num_features])
Test_set1 = Test[labels_test==0,:]
Test_set2 = Test[labels_test==1,:]

clf_logistic.fit(Train_set1[:,1:num_features], Train_set1[:,0])
clf_rf.fit(Train_set1[:,1:num_features], Train_set1[:,0])

# Fit the ml-health model
clf_health_yelp = hc.MlHealth()
clf_health_yelp.fit(Train_set1[:,1:num_features])

for a in range(0, num_points):
    ratio = a/(num_points-1)
    train_data_cluster_noise = datasetMix.randomSample(Train_set1, Train_set2, ratio, num_samples)
    test_data_cluster_noise = datasetMix.randomSample(Test_set1, Test_set2, ratio, num_samples)
    accuracyLR_train[plot_index,a] = clf_logistic.score(train_data_cluster_noise[:,1:num_features], train_data_cluster_noise[:,0])
    accuracyRF_train[plot_index,a] = clf_rf.score(train_data_cluster_noise[:,1:num_features], train_data_cluster_noise[:,0])
    accuracyLR_test[plot_index,a] = clf_logistic.score(test_data_cluster_noise[:,1:num_features], test_data_cluster_noise[:,0])
    accuracyRF_test[plot_index,a] = clf_rf.score(test_data_cluster_noise[:,1:num_features], test_data_cluster_noise[:,0])
    mlHealth_cluster_train[plot_index,a] = clf_health_yelp.score(train_data_cluster_noise[:,1:num_features],N)
    mlHealth_cluster_test[plot_index,a] = clf_health_yelp.score(test_data_cluster_noise[:,1:num_features],N)
    print("Yelp iteration: ",a)

print("accuracyLR_train value for yelp dataset is: {0}".format(accuracyLR_train[plot_index,:]))
print("accuracyRF_train value for yelp dataset is: {0}".format(accuracyRF_train[plot_index,:]))
print("accuracyLR_test value for yelp dataset is: {0}".format(accuracyLR_test[plot_index,:]))
print("accuracyRF_test value for yelp dataset is: {0}".format(accuracyRF_test[plot_index,:]))
print("mlHealth_cluster_train for yelp dataset is: {0}".format(mlHealth_cluster_train[plot_index,:]))
print("mlHealth_cluster_test for yelp dataset is: {0}".format( mlHealth_cluster_train[plot_index,:]))

# Census data
plot_index = 2

# Read the test and train datasets
Train = np.genfromtxt(path + 'census/original/train/census_train.csv', dtype = float, delimiter=",")
Test = np.genfromtxt(path + 'census/original/test/census_test.csv', dtype = float, delimiter=",")
num_samples, num_features = Train.shape

# Cluster the training data
subset_data_train = []
subset_data_test = []
for a in range(0,num_features):
    if(len(np.unique(Train[:,a])) > 30):
        subset_data_train.append(Train[:,a])
        subset_data_test.append(Test[:,a])
subset_data_train = np.array(subset_data_train)
subset_data_test = np.array(subset_data_test)

# Cluster only continuous values features
labels_train = kmeans.fit_predict(subset_data_train.T)
Train_set1 = Train[labels_train==0,:]
Train_set2 = Train[labels_train==1,:]
labels_test = kmeans.predict(subset_data_test.T)
Test_set1 = Test[labels_test==0,:]
Test_set2 = Test[labels_test==1,:]

clf_logistic.fit(Train_set1[:,0:num_features-1], Train_set1[:,num_features-1])
clf_rf.fit(Train_set1[:,0:num_features-1], Train_set1[:,num_features-1])

# Fit the ml-health model
clf_health_census = hc.MlHealth()
clf_health_census.fit(Train_set1[:,0:num_features-1])

for a in range(0, num_points):
    ratio = a/(num_points-1)
    train_data_cluster_noise = datasetMix.randomSample(Train_set1, Train_set2, ratio, num_samples)
    test_data_cluster_noise = datasetMix.randomSample(Test_set1, Test_set2, ratio, num_samples)
    accuracyLR_train[plot_index,a] = clf_logistic.score(train_data_cluster_noise[:,0:num_features-1], train_data_cluster_noise[:,num_features-1])
    accuracyRF_train[plot_index,a] = clf_rf.score(train_data_cluster_noise[:,0:num_features-1], train_data_cluster_noise[:,num_features-1])
    accuracyLR_test[plot_index,a] = clf_logistic.score(test_data_cluster_noise[:,0:num_features-1], test_data_cluster_noise[:,num_features-1])
    accuracyRF_test[plot_index,a] = clf_rf.score(test_data_cluster_noise[:,0:num_features-1], test_data_cluster_noise[:,num_features-1])
    mlHealth_cluster_train[plot_index,a] = clf_health_census.score(train_data_cluster_noise[:,0:num_features-1],N)
    mlHealth_cluster_test[plot_index,a] = clf_health_census.score(test_data_cluster_noise[:,0:num_features-1],N)
    print("Census iteration: ", a)

print("accuracyLR_train value for census dataset is: {0}".format(accuracyLR_train[plot_index,:]))
print("accuracyRF_train value for census dataset is: {0}".format(accuracyRF_train[plot_index,:]))
print("accuracyLR_test value for census dataset is: {0}".format(accuracyLR_test[plot_index,:]))
print("accuracyRF_test value for census dataset is: {0}".format(accuracyRF_test[plot_index,:]))
print("mlHealth_cluster_train for census dataset is: {0}".format(mlHealth_cluster_train[plot_index,:]))
print("mlHealth_cluster_test for census dataset is: {0}".format( mlHealth_cluster_train[plot_index,:]))

# Forest data
plot_index = 3

# Read the test and train datasets
Train = np.genfromtxt(path + 'covertype/original/train/covertype_train.csv', dtype = float, delimiter=",")
Test = np.genfromtxt(path + 'covertype/original/test/covertype_test.csv', dtype = float, delimiter=",")
num_samples, num_features = Train.shape

# Cluster the training data
labels_train = kmeans.fit_predict(Train[:,0:num_features-1])
Train_set1 = Train[labels_train==0,:]
Train_set2 = Train[labels_train==1,:]
labels_test = kmeans.predict(Test[:,0:num_features-1])
Test_set1 = Test[labels_test==0,:]
Test_set2 = Test[labels_test==1,:]

clf_logistic.fit(Train_set1[:,0:num_features-1], Train_set1[:,num_features-1])
clf_rf.fit(Train_set1[:,0:num_features-1], Train_set1[:,num_features-1])

# Fit the ml-health model
clf_health_forest = hc.MlHealth()
clf_health_forest.fit(Train_set1[:,0:num_features-1])

for a in range(0, num_points):
    ratio = a/(num_points-1)
    train_data_cluster_noise = datasetMix.randomSample(Train_set1, Train_set2, ratio, num_samples)
    test_data_cluster_noise = datasetMix.randomSample(Test_set1, Test_set2, ratio, num_samples)
    accuracyLR_train[plot_index,a] = clf_logistic.score(train_data_cluster_noise[:,0:num_features-1], train_data_cluster_noise[:,num_features-1])
    accuracyRF_train[plot_index,a] = clf_rf.score(train_data_cluster_noise[:,0:num_features-1], train_data_cluster_noise[:,num_features-1])
    accuracyLR_test[plot_index,a] = clf_logistic.score(test_data_cluster_noise[:,0:num_features-1], test_data_cluster_noise[:,num_features-1])
    accuracyRF_test[plot_index,a] = clf_rf.score(test_data_cluster_noise[:,0:num_features-1], test_data_cluster_noise[:,num_features-1])
    mlHealth_cluster_train[plot_index,a] = clf_health_forest.score(train_data_cluster_noise[:,0:num_features-1],N)
    mlHealth_cluster_test[plot_index,a] = clf_health_forest.score(test_data_cluster_noise[:,0:num_features-1],N)
    print("Census iteration: ", a)
print("accuracyLR_train value for forest dataset is: {0}".format(accuracyLR_train[plot_index,:]))
print("accuracyRF_train value for forest dataset is: {0}".format(accuracyRF_train[plot_index,:]))
print("accuracyLR_test value for forest dataset is: {0}".format(accuracyLR_test[plot_index,:]))
print("accuracyRF_test value for forest dataset is: {0}".format(accuracyRF_test[plot_index,:]))
print("mlHealth_cluster_train for forest dataset is: {0}".format(mlHealth_cluster_train[plot_index,:]))
print("mlHealth_cluster_test for forest dataset is: {0}".format( mlHealth_cluster_train[plot_index,:]))

# Fifth dataset is letter
plot_index = 4

# Read training and test datasets
Train = np.genfromtxt(path + 'letter/original/train/letter_train.csv', dtype = float, delimiter=",")
Test = np.genfromtxt(path + 'letter/original/test/letter_test.csv', dtype = float, delimiter=",")

num_samples, num_features = Train.shape
# Cluster the training data
labels_train = kmeans.fit_predict(Train[:, 1:num_features])
Train_set1 = Train[labels_train==0,:]
Train_set2 = Train[labels_train==1,:]
labels_test = kmeans.predict(Test[:,1:num_features])
Test_set1 = Test[labels_test==0,:]
Test_set2 = Test[labels_test==1,:]

clf_logistic.fit(Train_set1[:,1:num_features], Train_set1[:,0])
clf_rf.fit(Train_set1[:,1:num_features], Train_set1[:,0])

# Fit the ml-health model
clf_health_letter = hc.MlHealth()
clf_health_letter.fit(Train_set1[:,1:num_features])

for a in range(0, num_points):
    ratio = a/(num_points-1)
    train_data_cluster_noise = datasetMix.randomSample(Train_set1, Train_set2, ratio, num_samples)
    test_data_cluster_noise = datasetMix.randomSample(Test_set1, Test_set2, ratio, num_samples)
    accuracyLR_train[plot_index,a] = clf_logistic.score(train_data_cluster_noise[:,1:num_features], train_data_cluster_noise[:,0])
    accuracyRF_train[plot_index,a] = clf_rf.score(train_data_cluster_noise[:,1:num_features], train_data_cluster_noise[:,0])
    accuracyLR_test[plot_index,a] = clf_logistic.score(test_data_cluster_noise[:,1:num_features], test_data_cluster_noise[:,0])
    accuracyRF_test[plot_index,a] = clf_rf.score(test_data_cluster_noise[:,1:num_features], test_data_cluster_noise[:,0])
    mlHealth_cluster_train[plot_index,a] = clf_health_letter.score(train_data_cluster_noise[:,1:num_features],N)
    mlHealth_cluster_test[plot_index,a] = clf_health_letter.score(test_data_cluster_noise[:,1:num_features],N)
    print("letter iteration: ",a)

print("accuracyLR_train value for letter dataset is: {0}".format(accuracyLR_train[plot_index,:]))
print("accuracyRF_train value for letter dataset is: {0}".format(accuracyRF_train[plot_index,:]))
print("accuracyLR_test value for letter dataset is: {0}".format(accuracyLR_test[plot_index,:]))
print("accuracyRF_test value for letter dataset is: {0}".format(accuracyRF_test[plot_index,:]))
print("mlHealth_cluster_train for letter dataset is: {0}".format(mlHealth_cluster_train[plot_index,:]))
print("mlHealth_cluster_test for letter dataset is: {0}".format( mlHealth_cluster_train[plot_index,:]))

import pickle
# Saving the objects:
pickle.dump( [accuracyLR_train, accuracyRF_train, accuracyLR_test, accuracyRF_test, mlHealth_cluster_train, mlHealth_cluster_test], open( "classification_plot_cluster_noise_"+ str(N) +".p", "wb" ) )

