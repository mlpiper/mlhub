"""
This code generates data for 2 plots
plot 1: For logistic regression, it generates data to plot the accuracy vs health-score graph for
        5 datasets, where random noise is added to datasets.
plot 2: For random forest, it generates data to plot the accuracy vs health-score graph for
        5 datasets, where random noise is added to datasets.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import sys
sys.path.insert(0,'../../..')
import ml_health.experiments.machineLearning.datasetMix as dm
import ml_health.experiments.machineLearning.randomNoise as rn
import ml_health.univariate.univariate_health_calculator as hc

# Path to datasets on data-lake
path = "/data-lake/ml-prototypes/classification/ml/"
# Number of points in the plots per curve
num_points = 10
num_datasets = 5
# Number of minimum points to consider for calcualtion of health score
N = 1

# Initialize the variable to plot
meanAccuracyLR_random_noise = np.zeros((num_datasets, num_points))
meanAccuracyRF_random_noise = np.zeros((num_datasets, num_points))
meanHealth_random_noise = np.zeros((num_datasets, num_points))

# Initialize the logistic regression and random forest classification algorithms
clf_logistic = LogisticRegression(C=0.025, multi_class='multinomial', penalty='l1', solver='saga', tol=0.1)
clf_rf = RandomForestClassifier(max_depth=20, random_state=0, n_estimators=50)

# First dataset is Samsung
plot_index = 0

# Read training and test datasets
Train = np.genfromtxt(path + 'samsung/original/train/samsung_train.csv', dtype = float, delimiter=",")
Test = np.genfromtxt(path + 'samsung/original/test/samsung_test.csv', dtype = float, delimiter=",")
num_samples, num_features = Test.shape

# Add noise to test data, label is the first column
Test_noisy_random = np.copy(Test)
Test_noisy_random[:,1:num_features] = rn.randomNoise(Test[:,1:num_features], 10)

# Fit using logistic regression and random forest
clf_logistic.fit(Train[:,1:num_features], Train[:,0])
clf_rf.fit(Train[:,1:num_features], Train[:,0])

# Fit the ml-health model
clf_health_samsung = hc.MlHealth()
clf_health_samsung.fit(Train[:,1:num_features])

# Calculate the accuracy of both algorithms and corresponding ml-health scores
for a in range(0, num_points):
    ratio = a/(num_points-1)
    test_data_random_noise = dm.randomSample(Test,Test_noisy_random, ratio, num_samples)
    meanAccuracyLR_random_noise[plot_index,a] = clf_logistic.score(test_data_random_noise[:,1:num_features], test_data_random_noise[:,0])
    meanAccuracyRF_random_noise[plot_index,a] = clf_rf.score(test_data_random_noise[:,1:num_features], test_data_random_noise[:,0])
    meanHealth_random_noise[plot_index,a] = clf_health_samsung.score(test_data_random_noise[:,1:num_features], N)
    print("Samsung iteration: ",a)

print("meanAccuracyLR_random_noise[samsung,a]", meanAccuracyLR_random_noise[plot_index,:])
print("meanAccuracyRF_random_noise[samsung,a]", meanAccuracyRF_random_noise[plot_index,:])
print("meanHealth_ransom_noise[samsung,a]", meanHealth_random_noise[plot_index,:])

# Yelp data
plot_index = 1

# Read the test and training datasets
Train = np.genfromtxt(path + 'yelp/original/train/yelp_train.csv', dtype = float, delimiter=",")
Test = np.genfromtxt(path + 'yelp/original/test/yelp_test.csv', dtype = float, delimiter=",")
num_samples, num_features = Test.shape

# Add noise to test data, label is the first column
Test_noisy_random = np.copy(Test)
Test_noisy_random[:,1:num_features] = rn.randomNoise(Test[:,1:num_features], 10)

# Fit the algorithms with this dataset
clf_logistic.fit(Train[:,1:num_features], Train[:,0])
clf_rf.fit(Train[:,1:num_features], Train[:,0])

# Fit the ml-health model
clf_health_yelp = hc.MlHealth()
clf_health_yelp.fit(Train[:,1:num_features])

# Calculate the accuracy of both algorithms and corresponding health score
for a in range(0, num_points):
    ratio = a/(num_points-1)
    test_data_random_noise = dm.randomSample(Test,Test_noisy_random,ratio,num_samples)
    meanAccuracyLR_random_noise[plot_index,a] = clf_logistic.score(test_data_random_noise[:,1:num_features], test_data_random_noise[:,0])
    meanAccuracyRF_random_noise[plot_index,a] = clf_rf.score(test_data_random_noise[:,1:num_features], test_data_random_noise[:,0])
    meanHealth_random_noise[plot_index,a] = clf_health_yelp.score(test_data_random_noise[:,1:num_features], N)
    print("Yelp iteration: ",a)

print("meanAccuracyLR_random_noise[yelp,a]", meanAccuracyLR_random_noise[plot_index,:])
print("meanAccuracyRF_random_noise[yelp,a]", meanAccuracyRF_random_noise[plot_index,:])
print("meanHealth_ransom_noise[yelp,a]", meanHealth_random_noise[plot_index,:])
"""
"""
# Census data
plot_index = 2

# Read the test and train datasets
Train = np.genfromtxt(path + 'census/original/train/census_train.csv', dtype = float, delimiter=",")
Test = np.genfromtxt(path + 'census/original/test/census_test.csv', dtype = float, delimiter=",")
num_samples, num_features = Test.shape

# Add noise to test data, label is the first column
Test_noisy_random = np.copy(Test)
Test_noisy_random[:,0:num_features-1] = rn.randomNoise(Test[:,0:num_features-1], 10)

# Fit the algorithms with this dataset
clf_logistic.fit(Train[:,0:num_features-1], Train[:,num_features-1])
clf_rf.fit(Train[:,0:num_features-1], Train[:,num_features-1])

# Fit the ml-health model
clf_health_census = hc.MlHealth()
clf_health_census.fit(Train[:,0:num_features-1])

for a in range(0, num_points):
    ratio = a/(num_points-1)
    test_data_random_noise = dm.randomSample(Test,Test_noisy_random,ratio,num_samples)
    meanAccuracyLR_random_noise[plot_index,a] = clf_logistic.score(test_data_random_noise[:,0:num_features-1], test_data_random_noise[:,num_features-1])
    meanAccuracyRF_random_noise[plot_index,a] = clf_rf.score(test_data_random_noise[:,0:num_features-1], test_data_random_noise[:,num_features-1])
    meanHealth_random_noise[plot_index,a] = clf_health_census.score(test_data_random_noise[:,0:num_features-1], N)
    print("Census iteration: ",a)
print("meanAccuracyLR_random_noise[census,a]", meanAccuracyLR_random_noise[plot_index,:])
print("meanAccuracyRF_random_noise[census,a]", meanAccuracyRF_random_noise[plot_index,:])
print("meanHealth_ransom_noise[census,a]", meanHealth_random_noise[plot_index,:])

# Forest cover
plot_index = 3

# Read the training and test datasets
Train = np.genfromtxt(path + 'covertype/original/train/covertype_train.csv', dtype = float, delimiter=",")
Test = np.genfromtxt(path + 'covertype/original/test/covertype_test.csv', dtype = float, delimiter=",")
num_samples, num_features = Test.shape

# Add noise to test data, label is the first column
Test_noisy_random = np.copy(Test)
Test_noisy_random[:,0:num_features-1] = rn.randomNoise(Test[:,0:num_features-1],10)

# Fit the algorithms with this dataset
clf_logistic.fit(Train[:,0:num_features-1], Train[:,num_features-1])
clf_rf.fit(Train[:,0:num_features-1], Train[:,num_features-1])

# Fit the ml-health model
clf_health_forest = hc.MlHealth()
clf_health_forest.fit(Train[:,0:num_features-1])

for a in range(0, num_points):
    ratio = a/(num_points-1)
    test_data_random_noise = dm.randomSample(Test,Test_noisy_random,ratio,num_samples)
    meanAccuracyLR_random_noise[plot_index,a] = clf_logistic.score(test_data_random_noise[:,0:num_features-1], test_data_random_noise[:,num_features-1])
    meanAccuracyRF_random_noise[plot_index,a] = clf_rf.score(test_data_random_noise[:,0:num_features-1], test_data_random_noise[:,num_features-1])
    meanHealth_random_noise[plot_index,a] = clf_health_forest.score(test_data_random_noise[:,0:num_features-1], N)
    print("Forest cover iteration: ",a)

print("meanAccuracyLR_random_noise[census,a]", meanAccuracyLR_random_noise[plot_index,:])
print("meanAccuracyRF_random_noise[census,a]", meanAccuracyRF_random_noise[plot_index,:])
print("meanHealth_ransom_noise[census,a]", meanHealth_random_noise[plot_index,:])

# letter dataset
plot_index = 4

# Read the train and test datasets
Train = np.genfromtxt(path + 'letter/original/train/letter_train.csv', dtype = float, delimiter=",")
Test = np.genfromtxt(path + 'letter/original/test/letter_test.csv', dtype = float, delimiter=",")
num_samples, nr_features = Test.shape

# Add noise to test data, label is the first column
Test_noisy_random = np.copy(Test)
Test_noisy_random[:,1:num_features] = rn.randomNoise(Test[:,1:num_features],10)

# Fit the algorithms with this dataset
clf_logistic.fit(Train[:,1:num_features], Train[:,0])
clf_rf.fit(Train[:,1:num_features], Train[:,0])

# Fit the ml-health model
clf_health_letter = hc.MlHealth()
clf_health_letter.fit(Train[:,1:num_features])

for a in range(0, num_points):
    ratio = a/(num_points-1)
    test_data_random_noise = dm.randomSample(Test,Test_noisy_random,ratio,num_samples)
    meanAccuracyLR_random_noise[plot_index,a] = clf_logistic.score(test_data_random_noise[:,1:num_features], test_data_random_noise[:,0])
    meanAccuracyRF_random_noise[plot_index,a] = clf_rf.score(test_data_random_noise[:,1:num_features], test_data_random_noise[:,0])
    meanHealth_random_noise[plot_index,a] = clf_health_letter.score(test_data_random_noise[:,1:num_features], N)
    print("Letter iteration: ",a)

print("meanAccuracyLR_random_noise[census,a]", meanAccuracyLR_random_noise[plot_index,:])
print("meanAccuracyRF_random_noise[census,a]", meanAccuracyRF_random_noise[plot_index,:])
print("meanHealth_ransom_noise[census,a]", meanHealth_random_noise[plot_index,:])

import pickle

# Saving the objects:
pickle.dump( [meanAccuracyLR_random_noise, meanAccuracyRF_random_noise, meanHealth_random_noise], open( "classification_plot_random_noise_" + str(N) + ".p", "wb" ) )

