"""
This code generates data for 2 plots
plot 1: For logistic regression, it generates data to plot the accuracy vs health-score graph for
        5 datasets, where classification noise is added to datasets.
plot 2: For random forest, it generates data to plot the accuracy vs health-score graph for
        5 datasets, where classification noise is added to datasets.
Classification noise is the addition of samples that were misclassified by the algorithm.
As a result the accuracy of algorithm decreases. We look at the ml-health score in such a case.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import sys
sys.path.insert(0,'../../..')
import ml_health.experiments.machineLearning.datasetMix as datasetMix
import ml_health.univariate.univariate_health_calculator as hc

# Path to datasets on data-lake
path = "/data-lake/ml-prototypes/classification/ml/"
# Number of points in the plots per curve
num_points = 10
num_datasets = 5
# Number of minimum points to consider
N = 1

# Initialize the variable to plot
meanAccuracyLR_class_noise = np.zeros((num_datasets, num_points))
meanAccuracyRF_class_noise = np.zeros((num_datasets, num_points))
meanHealthLR_class_noise = np.zeros((num_datasets, num_points))
meanHealthRF_class_noise = np.zeros((num_datasets, num_points))

# Initialize the logistic regression and random forest classification algorithms
clf_logistic = LogisticRegression(C=0.025, multi_class='multinomial', penalty='l1', solver='saga', tol=0.1)
clf_rf = RandomForestClassifier(max_depth=20, random_state=0, n_estimators=50)

# First dataset is Samsung
plot_index = 0

# Read training and test datasets
Train = np.genfromtxt(path + 'samsung/original/train/samsung_train.csv', dtype = float, delimiter=",")
Test = np.genfromtxt(path + 'samsung/original/test/samsung_test.csv', dtype = float, delimiter=",")
num_samples, num_features = Test.shape

clf_logistic.fit(Train[:,1:num_features], Train[:,0])
clf_rf.fit(Train[:,1:num_features], Train[:,0])

Test_lr_bad = np.squeeze(Test[np.where(Test[:,0] != clf_logistic.predict(Test[:,1:num_features])),:])
Test_lr_good = np.squeeze(Test[np.where(Test[:,0] == clf_logistic.predict(Test[:,1:num_features])),:])
Test_rf_bad = np.squeeze(Test[np.where(Test[:,0] != clf_rf.predict(Test[:,1:num_features])),:])
Test_rf_good = np.squeeze(Test[np.where(Test[:,0] == clf_rf.predict(Test[:,1:num_features])),:])

# Fit the ml-health model
clf_health_samsung = hc.MlHealth()
clf_health_samsung.fit(Train[:,1:num_features])


for a in range(0, num_points):
    ratio = a/(num_points-1)
    test_data_class_noise_lr = datasetMix.randomSample(Test_lr_good, Test_lr_bad, ratio, num_samples)
    test_data_class_noise_rf = datasetMix.randomSample(Test_rf_good, Test_rf_bad, ratio, num_samples)
    meanAccuracyLR_class_noise[plot_index,a] = clf_logistic.score(test_data_class_noise_lr[:,1:num_features], test_data_class_noise_lr[:,0])
    meanAccuracyRF_class_noise[plot_index,a] = clf_rf.score(test_data_class_noise_rf[:,1:num_features], test_data_class_noise_rf[:,0])
    meanHealthLR_class_noise[plot_index,a] = clf_health_samsung.score(test_data_class_noise_lr[:,1:num_features],N)
    meanHealthRF_class_noise[plot_index,a] = clf_health_samsung.score(test_data_class_noise_rf[:,1:num_features],N)
    print("Samsung iteration: ",a)

print("meanAccuracyLR_class_noise[samsung,a]", meanAccuracyLR_class_noise[plot_index,:])
print("meanAccuracyRF_class_noise[samsung,a]", meanAccuracyRF_class_noise[plot_index,:])
print("meanHealthLR_class_noise[samsung,a]", meanHealthLR_class_noise[plot_index,:])
print("meanHealthRF_class_noise[samsung,a]", meanHealthRF_class_noise[plot_index,:])



# Second dataset is Yelp
plot_index = 1

# Read training and test datasets
Train = np.genfromtxt(path + 'yelp/original/train/yelp_train.csv', dtype = float, delimiter=",")
Test = np.genfromtxt(path + 'yelp/original/test/yelp_test.csv', dtype = float, delimiter=",")
num_samples, num_features = Test.shape

clf_logistic.fit(Train[:,1:num_features], Train[:,0])
clf_rf.fit(Train[:,1:num_features], Train[:,0])

Test_lr_bad = np.squeeze(Test[np.where(Test[:,0] != clf_logistic.predict(Test[:,1:num_features])),:])
Test_lr_good = np.squeeze(Test[np.where(Test[:,0] == clf_logistic.predict(Test[:,1:num_features])),:])
Test_rf_bad = np.squeeze(Test[np.where(Test[:,0] != clf_rf.predict(Test[:,1:num_features])),:])
Test_rf_good = np.squeeze(Test[np.where(Test[:,0] == clf_rf.predict(Test[:,1:num_features])),:])

# Fit the ml-health model
clf_health_yelp = hc.MlHealth()
clf_health_yelp.fit(Train[:,1:num_features])

for a in range(0, num_points):
    ratio = a/(num_points-1)
    test_data_class_noise_lr = datasetMix.randomSample(Test_lr_good, Test_lr_bad, ratio, num_samples)
    test_data_class_noise_rf = datasetMix.randomSample(Test_rf_good, Test_rf_bad, ratio, num_samples)
    meanAccuracyLR_class_noise[plot_index,a] = clf_logistic.score(test_data_class_noise_lr[:,1:num_features], test_data_class_noise_lr[:,0])
    meanAccuracyRF_class_noise[plot_index,a] = clf_rf.score(test_data_class_noise_rf[:,1:num_features], test_data_class_noise_rf[:,0])
    meanHealthLR_class_noise[plot_index,a] = clf_health_yelp.score(test_data_class_noise_lr[:,1:num_features],N)
    meanHealthRF_class_noise[plot_index,a] = clf_health_yelp.score(test_data_class_noise_rf[:,1:num_features],N)
    print("Yelp iteration: ",a)

print("meanAccuracyLR_class_noise[yelp,a]", meanAccuracyLR_class_noise[plot_index,:])
print("meanAccuracyRF_class_noise[yelp,a]", meanAccuracyRF_class_noise[plot_index,:])
print("meanHealthLR_class_noise[yelp,a]", meanHealthLR_class_noise[plot_index,:])
print("meanHealthRF_class_noise[yelp,a]", meanHealthRF_class_noise[plot_index,:])

# Census data
plot_index = 2

# Read the test and train datasets
Train = np.genfromtxt(path + 'census/original/train/census_train.csv', dtype = float, delimiter=",")
Test = np.genfromtxt(path + 'census/original/test/census_test.csv', dtype = float, delimiter=",")
num_samples, num_features = Test.shape

# Fit the algorithms with this dataset
clf_logistic.fit(Train[:,0:num_features-1], Train[:,num_features-1])
clf_rf.fit(Train[:,0:num_features-1], Train[:,num_features-1])

Test_lr_bad = np.squeeze(Test[np.where(Test[:,num_features-1] != clf_logistic.predict(Test[:,0:num_features-1])),:])
Test_lr_good = np.squeeze(Test[np.where(Test[:,num_features-1] == clf_logistic.predict(Test[:,0:num_features-1])),:])
Test_rf_bad = np.squeeze(Test[np.where(Test[:,num_features-1] != clf_rf.predict(Test[:,0:num_features-1])),:])
Test_rf_good = np.squeeze(Test[np.where(Test[:,num_features-1] == clf_rf.predict(Test[:,0:num_features-1])),:])


# Fit the ml-health model
clf_health_census = hc.MlHealth()
clf_health_census.fit(Train[:,0:Train.shape[1]-1])

for a in range(0, num_points):
    ratio = a/(num_points-1)
    test_data_class_noise_lr = datasetMix.randomSample(Test_lr_good, Test_lr_bad, ratio, num_samples)
    test_data_class_noise_rf = datasetMix.randomSample(Test_rf_good, Test_rf_bad, ratio, num_samples)
    meanAccuracyLR_class_noise[plot_index,a] = clf_logistic.score(test_data_class_noise_lr[:,0:num_features-1], test_data_class_noise_lr[:,num_features-1])
    meanAccuracyRF_class_noise[plot_index,a] = clf_rf.score(test_data_class_noise_rf[:,0:num_features-1], test_data_class_noise_rf[:,num_features-1])
    meanHealthLR_class_noise[plot_index,a] = clf_health_census.score(test_data_class_noise_lr[:,0:num_features-1],N)
    meanHealthRF_class_noise[plot_index,a] = clf_health_census.score(test_data_class_noise_rf[:,0:num_features-1],N)
    print("Census iteration: ", a)
print("meanAccuracyLR_class_noise[census,a]", meanAccuracyLR_class_noise[plot_index,:])
print("meanAccuracyRF_class_noise[census,a]", meanAccuracyRF_class_noise[plot_index,:])
print("meanHealthLR_class_noise[census,a]", meanHealthLR_class_noise[plot_index,:])
print("meanHealthRF_class_noise[census,a]", meanHealthRF_class_noise[plot_index,:])



# Forest data
plot_index = 3

# Read the test and train datasets
Train = np.genfromtxt(path + 'covertype/original/train/covertype_train.csv', dtype = float, delimiter=",")
Test = np.genfromtxt(path + 'covertype/original/test/covertype_test.csv', dtype = float, delimiter=",")
num_samples, num_features = Test.shape

# Fit the algorithms with this dataset
clf_logistic.fit(Train[:,0:num_features-1], Train[:,num_features-1])
clf_rf.fit(Train[:,0:num_features-1], Train[:,num_features-1])


Test_lr_bad = np.squeeze(Test[np.where(Test[:,num_features-1] != clf_logistic.predict(Test[:,0:num_features-1])),:])
Test_lr_good = np.squeeze(Test[np.where(Test[:,num_features-1] == clf_logistic.predict(Test[:,0:num_features-1])),:])
Test_rf_bad = np.squeeze(Test[np.where(Test[:,num_features-1] != clf_rf.predict(Test[:,0:num_features-1])),:])
Test_rf_good = np.squeeze(Test[np.where(Test[:,num_features-1] == clf_rf.predict(Test[:,0:num_features-1])),:])

# Fit the ml-health model
clf_health_forest = hc.MlHealth()
clf_health_forest.fit(Train[:,0:Train.shape[1]-1])

for a in range(0, num_points):
    ratio = a/(num_points-1)
    test_data_class_noise_lr = datasetMix.randomSample(Test_lr_good, Test_lr_bad, ratio, num_samples)
    test_data_class_noise_rf = datasetMix.randomSample(Test_rf_good, Test_rf_bad, ratio, num_samples)
    meanAccuracyLR_class_noise[plot_index,a] = clf_logistic.score(test_data_class_noise_lr[:,0:num_features-1], test_data_class_noise_lr[:,num_features-1])
    meanAccuracyRF_class_noise[plot_index,a] = clf_rf.score(test_data_class_noise_rf[:,0:num_features-1], test_data_class_noise_rf[:,num_features-1])
    meanHealthLR_class_noise[plot_index,a] = clf_health_forest.score(test_data_class_noise_lr[:,0:num_features-1],N)
    meanHealthRF_class_noise[plot_index,a] = clf_health_forest.score(test_data_class_noise_rf[:,0:num_features-1],N)
    print("Forest Cover iteration: ", a)
print("meanAccuracyLR_class_noise[forest,a]", meanAccuracyLR_class_noise[plot_index,:])
print("meanAccuracyRF_class_noise[forest,a]", meanAccuracyRF_class_noise[plot_index,:])
print("meanHealthLR_class_noise[forest,a]", meanHealthLR_class_noise[plot_index,:])
print("meanHealthRF_class_noise[forest,a]", meanHealthRF_class_noise[plot_index,:])


# Fifth dataset is letter
plot_index = 4

# Read training and test datasets
Train = np.genfromtxt(path + 'letter/original/train/letter_train.csv', dtype = float, delimiter=",")
Test = np.genfromtxt(path + 'letter/original/test/letter_test.csv', dtype = float, delimiter=",")
num_samples, num_features = Test.shape

clf_logistic.fit(Train[:,1:num_features], Train[:,0])
clf_rf.fit(Train[:,1:num_features], Train[:,0])

Test_lr_bad = np.squeeze(Test[np.where(Test[:,0] != clf_logistic.predict(Test[:,1:num_features])),:])
Test_lr_good = np.squeeze(Test[np.where(Test[:,0] == clf_logistic.predict(Test[:,1:num_features])),:])
Test_rf_bad = np.squeeze(Test[np.where(Test[:,0] != clf_rf.predict(Test[:,1:num_features])),:])
Test_rf_good = np.squeeze(Test[np.where(Test[:,0] == clf_rf.predict(Test[:,1:num_features])),:])

# Fit the ml-health model
clf_health_letter = hc.MlHealth()
clf_health_letter.fit(Train[:,1:num_features])


for a  in range(0, num_points):
    ratio = a/(num_points-1)
    test_data_class_noise_lr = datasetMix.randomSample(Test_lr_good,Test_lr_bad, ratio, num_samples)
    test_data_class_noise_rf = datasetMix.randomSample(Test_rf_good,Test_rf_bad, ratio, num_samples)
    meanAccuracyLR_class_noise[plot_index,a] = clf_logistic.score(test_data_class_noise_lr[:,1:num_features], test_data_class_noise_lr[:,0])
    meanAccuracyRF_class_noise[plot_index,a] = clf_rf.score(test_data_class_noise_rf[:,1:num_features], test_data_class_noise_rf[:,0])
    meanHealthLR_class_noise[plot_index,a] = clf_health_letter.score(test_data_class_noise_lr[:,1:num_features],N)
    meanHealthRF_class_noise[plot_index,a] = clf_health_letter.score(test_data_class_noise_rf[:,1:num_features],N)
    print("Letter iteration: ",a)

print("meanAccuracyLR_class_noise[letter,a]", meanAccuracyLR_class_noise[plot_index,:])
print("meanAccuracyRF_class_noise[letter,a]", meanAccuracyRF_class_noise[plot_index,:])
print("meanHealthLR_class_noise[letter,a]", meanHealthLR_class_noise[plot_index,:])
print("meanHealthRF_class_noise[letter,a]", meanHealthRF_class_noise[plot_index,:])

import pickle
# Saving the objects:
pickle.dump( [meanAccuracyLR_class_noise, meanAccuracyRF_class_noise, meanHealthLR_class_noise, meanHealthRF_class_noise], open( "classification_plot_class_noise_"+ str(N) +".p", "wb" ) )
