"""
This code generates data for 2 plots
plot 1: For logistic regression, it generates data to plot the accuracy vs health-score graph for
        5 datasets, where classification noise is added to datasets.
plot 2: For random forest, it generates data to plot the accuracy vs health-score graph for
        5 datasets, where classification noise is added to datasets.
Classification noise is the addition of samples that were misclassified by the algorithm.
As a result the accuracy of algorithm decreases. We look at the ml-health score in such a case.
We train only the well classified samples (in training set) to calculate the ml-health parameters.
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
# Number of minimum scores to consider
N = 1
# Initialize the variable to plot
meanAccuracyLR_class_noise = np.zeros((num_datasets, num_points))
meanAccuracyRF_class_noise = np.zeros((num_datasets, num_points))
meanHealthLR_class_noise = np.zeros((num_datasets, num_points))
meanHealthRF_class_noise = np.zeros((num_datasets, num_points))


meanAccuracyLR_class_noise_train = np.zeros((num_datasets, num_points))
meanAccuracyRF_class_noise_train = np.zeros((num_datasets, num_points))
meanHealthLR_class_noise_train = np.zeros((num_datasets, num_points))
meanHealthRF_class_noise_train = np.zeros((num_datasets, num_points))

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


Train_lr_bad = np.squeeze(Train[np.where(Train[:,0] != clf_logistic.predict(Train[:,1:num_features])),:])
Train_lr_good = np.squeeze(Train[np.where(Train[:,0] == clf_logistic.predict(Train[:,1:num_features])),:])
Train_rf_bad = np.squeeze(Train[np.where(Train[:,0] != clf_rf.predict(Train[:,1:num_features])),:])
Train_rf_good = np.squeeze(Train[np.where(Train[:,0] == clf_rf.predict(Train[:,1:num_features])),:])

Test_lr_bad = np.squeeze(Test[np.where(Test[:,0] != clf_logistic.predict(Test[:,1:num_features])),:])
Test_lr_good = np.squeeze(Test[np.where(Test[:,0] == clf_logistic.predict(Test[:,1:num_features])),:])
Test_rf_bad = np.squeeze(Test[np.where(Test[:,0] != clf_rf.predict(Test[:,1:num_features])),:])
Test_rf_good = np.squeeze(Test[np.where(Test[:,0] == clf_rf.predict(Test[:,1:num_features])),:])

# Fit the ml-health model
clf_health_samsung_lr = hc.MlHealth()
clf_health_samsung_lr.fit(Train_lr_good[:,1:num_features])
clf_health_samsung_rf = hc.MlHealth()
clf_health_samsung_rf.fit(Train_rf_good[:,1:num_features])

# Random forest has 100% training accuracy
for a in range(0, num_points):
    ratio = a/(num_points-1)
    test_data_class_noise_lr = datasetMix.randomSample(Test_lr_good, Test_lr_bad, ratio, num_samples)
    test_data_class_noise_rf = datasetMix.randomSample(Test_rf_good, Test_rf_bad, ratio, num_samples)
    train_data_class_noise_lr = datasetMix.randomSample(Train_lr_good, Train_lr_bad, ratio, num_samples)
    #train_data_class_noise_rf = datasetMix.randomSample(Train_rf_good, Train_rf_bad, ratio, num_samples)
    meanAccuracyLR_class_noise[plot_index,a] = clf_logistic.score(test_data_class_noise_lr[:,1:num_features], test_data_class_noise_lr[:,0])
    meanAccuracyRF_class_noise[plot_index,a] = clf_rf.score(test_data_class_noise_rf[:,1:num_features], test_data_class_noise_rf[:,0])
    meanHealthLR_class_noise[plot_index,a] = clf_health_samsung_lr.score(test_data_class_noise_lr[:,1:num_features],N)
    meanHealthRF_class_noise[plot_index,a] = clf_health_samsung_rf.score(test_data_class_noise_rf[:,1:num_features],N)
    meanAccuracyLR_class_noise_train[plot_index,a] = clf_logistic.score(train_data_class_noise_lr[:,1:num_features], train_data_class_noise_lr[:,0])
    #meanAccuracyRF_class_noise_train[plot_index,a] = clf_rf.score(train_data_class_noise_rf[:,1:num_features], train_data_class_noise_rf[:,0])
    meanHealthLR_class_noise_train[plot_index,a] = clf_health_samsung_lr.score(train_data_class_noise_lr[:,1:num_features],N)
    #meanHealthRF_class_noise_train[plot_index,a] = clf_health_samsung_rf.score(train_data_class_noise_rf[:,1:num_features])
    print("Samsung iteration: ",a)

print("meanAccuracyLR_class_noise[samsung,a]", meanAccuracyLR_class_noise[plot_index,:])
print("meanAccuracyRF_class_noise[samsung,a]", meanAccuracyRF_class_noise[plot_index,:])
print("meanHealthLR_class_noise[samsung,a]", meanHealthLR_class_noise[plot_index,:])
print("meanHealthRF_class_noise[samsung,a]", meanHealthRF_class_noise[plot_index,:])

print("meanAccuracyLR_class_noise_train[samsung,a]", meanAccuracyLR_class_noise_train[plot_index,:])
print("meanAccuracyRF_class_noise_train[samsung,a]", meanAccuracyRF_class_noise_train[plot_index,:])
print("meanHealthLR_class_noise_train[samsung,a]", meanHealthLR_class_noise_train[plot_index,:])
print("meanHealthRF_class_noise_train[samsung,a]", meanHealthRF_class_noise_train[plot_index,:])


# Second dataset is Yelp
plot_index = 1

# Read training and test datasets
Train = np.genfromtxt(path + 'yelp/original/train/yelp_train.csv', dtype = float, delimiter=",")
Test = np.genfromtxt(path + 'yelp/original/test/yelp_test.csv', dtype = float, delimiter=",")
num_samples, num_features = Test.shape

clf_logistic.fit(Train[:,1:num_features], Train[:,0])
clf_rf.fit(Train[:,1:num_features], Train[:,0])

Train_lr_bad = np.squeeze(Train[np.where(Train[:,0] != clf_logistic.predict(Train[:,1:num_features])),:])
Train_lr_good = np.squeeze(Train[np.where(Train[:,0] == clf_logistic.predict(Train[:,1:num_features])),:])
Train_rf_bad = np.squeeze(Train[np.where(Train[:,0] != clf_rf.predict(Train[:,1:num_features])),:])
Train_rf_good = np.squeeze(Train[np.where(Train[:,0] == clf_rf.predict(Train[:,1:num_features])),:])

Test_lr_bad = np.squeeze(Test[np.where(Test[:,0] != clf_logistic.predict(Test[:,1:num_features])),:])
Test_lr_good = np.squeeze(Test[np.where(Test[:,0] == clf_logistic.predict(Test[:,1:num_features])),:])
Test_rf_bad = np.squeeze(Test[np.where(Test[:,0] != clf_rf.predict(Test[:,1:num_features])),:])
Test_rf_good = np.squeeze(Test[np.where(Test[:,0] == clf_rf.predict(Test[:,1:num_features])),:])

# Fit the ml-health model
clf_health_yelp_lr = hc.MlHealth()
clf_health_yelp_lr.fit(Train_lr_good[:,1:num_features])
clf_health_yelp_rf = hc.MlHealth()
clf_health_yelp_rf.fit(Train_rf_good[:,1:num_features])


for a in range(0, num_points):
    ratio = a/(num_points-1)
    test_data_class_noise_lr = datasetMix.randomSample(Test_lr_good, Test_lr_bad, ratio, num_samples)
    test_data_class_noise_rf = datasetMix.randomSample(Test_rf_good, Test_rf_bad, ratio, num_samples)
    train_data_class_noise_lr = datasetMix.randomSample(Train_lr_good, Train_lr_bad, ratio, num_samples)
    train_data_class_noise_rf = datasetMix.randomSample(Train_rf_good, Train_rf_bad, ratio, num_samples)
    meanAccuracyLR_class_noise[plot_index,a] = clf_logistic.score(test_data_class_noise_lr[:,1:num_features], test_data_class_noise_lr[:,0])
    meanAccuracyRF_class_noise[plot_index,a] = clf_rf.score(test_data_class_noise_rf[:,1:num_features], test_data_class_noise_rf[:,0])
    meanHealthLR_class_noise[plot_index,a] = clf_health_yelp_lr.score(test_data_class_noise_lr[:,1:num_features],N)
    meanHealthRF_class_noise[plot_index,a] = clf_health_yelp_rf.score(test_data_class_noise_rf[:,1:num_features],N)
    meanAccuracyLR_class_noise_train[plot_index,a] = clf_logistic.score(train_data_class_noise_lr[:,1:num_features], train_data_class_noise_lr[:,0])
    meanAccuracyRF_class_noise_train[plot_index,a] = clf_rf.score(train_data_class_noise_rf[:,1:num_features], train_data_class_noise_rf[:,0])
    meanHealthLR_class_noise_train[plot_index,a] = clf_health_yelp_lr.score(train_data_class_noise_lr[:,1:num_features],N)
    meanHealthRF_class_noise_train[plot_index,a] = clf_health_yelp_rf.score(train_data_class_noise_rf[:,1:num_features],N)
    print("Yelp iteration: ",a)

print("meanAccuracyLR_class_noise[yelp,a]", meanAccuracyLR_class_noise[plot_index,:])
print("meanAccuracyRF_class_noise[yelp,a]", meanAccuracyRF_class_noise[plot_index,:])
print("meanHealthLR_class_noise[yelp,a]", meanHealthLR_class_noise[plot_index,:])
print("meanHealthRF_class_noise[yelp,a]", meanHealthRF_class_noise[plot_index,:])

print("meanAccuracyLR_class_noise_train[yelp,a]", meanAccuracyLR_class_noise_train[plot_index,:])
print("meanAccuracyRF_class_noise_train[yelp,a]", meanAccuracyRF_class_noise_train[plot_index,:])
print("meanHealthLR_class_noise_train[yelp,a]", meanHealthLR_class_noise_train[plot_index,:])
print("meanHealthRF_class_noise_train[yelp,a]", meanHealthRF_class_noise_train[plot_index,:])


# Census data
plot_index = 2

# Read the test and train datasets
Train = np.genfromtxt(path + 'census/original/train/census_train.csv', dtype = float, delimiter=",")
Test = np.genfromtxt(path + 'census/original/test/census_test.csv', dtype = float, delimiter=",")
num_samples, num_features = Test.shape

# Fit the algorithms with this dataset
clf_logistic.fit(Train[:,0:num_features-1], Train[:,num_features-1])
clf_rf.fit(Train[:,0:num_features-1], Train[:,num_features-1])

Train_lr_bad = np.squeeze(Train[np.where(Train[:,num_features-1] != clf_logistic.predict(Train[:,0:num_features-1])),:])
Train_lr_good = np.squeeze(Train[np.where(Train[:,num_features-1] == clf_logistic.predict(Train[:,0:num_features-1])),:])
Train_rf_bad = np.squeeze(Train[np.where(Train[:,num_features-1] != clf_rf.predict(Train[:,0:num_features-1])),:])
Train_rf_good = np.squeeze(Train[np.where(Train[:,num_features-1] == clf_rf.predict(Train[:,0:num_features-1])),:])


Test_lr_bad = np.squeeze(Test[np.where(Test[:,num_features-1] != clf_logistic.predict(Test[:,0:num_features-1])),:])
Test_lr_good = np.squeeze(Test[np.where(Test[:,num_features-1] == clf_logistic.predict(Test[:,0:num_features-1])),:])
Test_rf_bad = np.squeeze(Test[np.where(Test[:,num_features-1] != clf_rf.predict(Test[:,0:num_features-1])),:])
Test_rf_good = np.squeeze(Test[np.where(Test[:,num_features-1] == clf_rf.predict(Test[:,0:num_features-1])),:])

# Fit the ml-health model
clf_health_census_lr = hc.MlHealth()
clf_health_census_lr.fit(Train[:,0:Train.shape[1]-1])
clf_health_census_rf = hc.MlHealth()
clf_health_census_rf.fit(Train[:,0:Train.shape[1]-1])

for a in range(0, num_points):
    ratio = a/(num_points-1)
    test_data_class_noise_lr = datasetMix.randomSample(Test_lr_good, Test_lr_bad, ratio, num_samples)
    test_data_class_noise_rf = datasetMix.randomSample(Test_rf_good, Test_rf_bad, ratio, num_samples)
    train_data_class_noise_lr = datasetMix.randomSample(Train_lr_good, Train_lr_bad, ratio, num_samples)
    train_data_class_noise_rf = datasetMix.randomSample(Train_rf_good, Train_rf_bad, ratio, num_samples)
    meanAccuracyLR_class_noise[plot_index,a] = clf_logistic.score(test_data_class_noise_lr[:,0:num_features-1], test_data_class_noise_lr[:,num_features-1])
    meanAccuracyRF_class_noise[plot_index,a] = clf_rf.score(test_data_class_noise_rf[:,0:num_features-1], test_data_class_noise_rf[:,num_features-1])
    meanHealthLR_class_noise[plot_index,a] = clf_health_census_lr.score(test_data_class_noise_lr[:,0:num_features-1],N)
    meanHealthRF_class_noise[plot_index,a] = clf_health_census_rf.score(test_data_class_noise_rf[:,0:num_features-1],N)
    meanAccuracyLR_class_noise_train[plot_index,a] = clf_logistic.score(train_data_class_noise_lr[:,0:num_features-1], train_data_class_noise_lr[:,num_features-1])
    meanAccuracyRF_class_noise_train[plot_index,a] = clf_rf.score(train_data_class_noise_rf[:,0:num_features-1], train_data_class_noise_rf[:,num_features-1])
    meanHealthLR_class_noise_train[plot_index,a] = clf_health_census_lr.score(train_data_class_noise_lr[:,0:num_features-1],N)
    meanHealthRF_class_noise_train[plot_index,a] = clf_health_census_rf.score(train_data_class_noise_rf[:,0:num_features-1],N)
    print("Census iteration: ", a)
print("meanAccuracyLR_class_noise[census,a]", meanAccuracyLR_class_noise[plot_index,:])
print("meanAccuracyRF_class_noise[census,a]", meanAccuracyRF_class_noise[plot_index,:])
print("meanHealthLR_class_noise[census,a]", meanHealthLR_class_noise[plot_index,:])
print("meanHealthRF_class_noise[census,a]", meanHealthRF_class_noise[plot_index,:])

print("meanAccuracyLR_class_noise_train[census,a]", meanAccuracyLR_class_noise_train[plot_index,:])
print("meanAccuracyRF_class_noise_train[census,a]", meanAccuracyRF_class_noise_train[plot_index,:])
print("meanHealthLR_class_noise_train[census,a]", meanHealthLR_class_noise_train[plot_index,:])
print("meanHealthRF_class_noise_train[census,a]", meanHealthRF_class_noise_train[plot_index,:])


# Forest data
plot_index = 3

# Read the test and train datasets
Train = np.genfromtxt(path + 'covertype/original/train/covertype_train.csv', dtype = float, delimiter=",")
Test = np.genfromtxt(path + 'covertype/original/test/covertype_test.csv', dtype = float, delimiter=",")
num_samples, num_features = Test.shape

# Fit the algorithms with this dataset
clf_logistic.fit(Train[:,0:num_features-1], Train[:,num_features-1])
clf_rf.fit(Train[:,0:num_features-1], Train[:,num_features-1])

Train_lr_bad = np.squeeze(Train[np.where(Train[:,num_features-1] != clf_logistic.predict(Train[:,0:num_features-1])),:])
Train_lr_good = np.squeeze(Train[np.where(Train[:,num_features-1] == clf_logistic.predict(Train[:,0:num_features-1])),:])
Train_rf_bad = np.squeeze(Train[np.where(Train[:,num_features-1] != clf_rf.predict(Train[:,0:num_features-1])),:])
Train_rf_good = np.squeeze(Train[np.where(Train[:,num_features-1] == clf_rf.predict(Train[:,0:num_features-1])),:])

Test_lr_bad = np.squeeze(Test[np.where(Test[:,num_features-1] != clf_logistic.predict(Test[:,0:num_features-1])),:])
Test_lr_good = np.squeeze(Test[np.where(Test[:,num_features-1] == clf_logistic.predict(Test[:,0:num_features-1])),:])
Test_rf_bad = np.squeeze(Test[np.where(Test[:,num_features-1] != clf_rf.predict(Test[:,0:num_features-1])),:])
Test_rf_good = np.squeeze(Test[np.where(Test[:,num_features-1] == clf_rf.predict(Test[:,0:num_features-1])),:])

# Fit the ml-health model
clf_health_forest_lr = hc.MlHealth()
clf_health_forest_lr.fit(Train[:,0:Train.shape[1]-1])
clf_health_forest_rf = hc.MlHealth()
clf_health_forest_rf.fit(Train[:,0:Train.shape[1]-1])

for a in range(0, num_points):
    ratio = a/(num_points-1)
    test_data_class_noise_lr = datasetMix.randomSample(Test_lr_good, Test_lr_bad, ratio, num_samples)
    test_data_class_noise_rf = datasetMix.randomSample(Test_rf_good, Test_rf_bad, ratio, num_samples)
    train_data_class_noise_lr = datasetMix.randomSample(Train_lr_good, Train_lr_bad, ratio, num_samples)
    train_data_class_noise_rf = datasetMix.randomSample(Train_rf_good, Train_rf_bad, ratio, num_samples)
    meanAccuracyLR_class_noise[plot_index,a] = clf_logistic.score(test_data_class_noise_lr[:,0:num_features-1], test_data_class_noise_lr[:,num_features-1])
    meanAccuracyRF_class_noise[plot_index,a] = clf_rf.score(test_data_class_noise_rf[:,0:num_features-1], test_data_class_noise_rf[:,num_features-1])
    meanHealthLR_class_noise[plot_index,a] = clf_health_forest_lr.score(test_data_class_noise_lr[:,0:num_features-1],N)
    meanHealthRF_class_noise[plot_index,a] = clf_health_forest_rf.score(test_data_class_noise_rf[:,0:num_features-1],N)
    meanAccuracyLR_class_noise_train[plot_index,a] = clf_logistic.score(train_data_class_noise_lr[:,0:num_features-1], train_data_class_noise_lr[:,num_features-1])
    meanAccuracyRF_class_noise_train[plot_index,a] = clf_rf.score(train_data_class_noise_rf[:,0:num_features-1], train_data_class_noise_rf[:,num_features-1])
    meanHealthLR_class_noise_train[plot_index,a] = clf_health_forest_lr.score(train_data_class_noise_lr[:,0:num_features-1],N)
    meanHealthRF_class_noise_train[plot_index,a] = clf_health_forest_rf.score(train_data_class_noise_rf[:,0:num_features-1],N)
    print("Forest Cover iteration: ", a)
print("meanAccuracyLR_class_noise[forest,a]", meanAccuracyLR_class_noise[plot_index,:])
print("meanAccuracyRF_class_noise[forest,a]", meanAccuracyRF_class_noise[plot_index,:])
print("meanHealthLR_class_noise[forest,a]", meanHealthLR_class_noise[plot_index,:])
print("meanHealthRF_class_noise[forest,a]", meanHealthRF_class_noise[plot_index,:])

print("meanAccuracyLR_class_noise_train[forest,a]", meanAccuracyLR_class_noise_train[plot_index,:])
print("meanAccuracyRF_class_noise_train[forest,a]", meanAccuracyRF_class_noise_train[plot_index,:])
print("meanHealthLR_class_noise_train[forest,a]", meanHealthLR_class_noise_train[plot_index,:])
print("meanHealthRF_class_noise_train[forest,a]", meanHealthRF_class_noise_train[plot_index,:])

# Fifth dataset is letter
plot_index = 4

# Read training and test datasets
Train = np.genfromtxt(path + 'letter/original/train/letter_train.csv', dtype = float, delimiter=",")
Test = np.genfromtxt(path + 'letter/original/test/letter_test.csv', dtype = float, delimiter=",")
num_samples, num_features = Test.shape

clf_logistic.fit(Train[:,1:num_features], Train[:,0])
clf_rf.fit(Train[:,1:num_features], Train[:,0])

Train_lr_bad = np.squeeze(Train[np.where(Train[:,0] != clf_logistic.predict(Train[:,1:num_features])),:])
Train_lr_good = np.squeeze(Train[np.where(Train[:,0] == clf_logistic.predict(Train[:,1:num_features])),:])
Train_rf_bad = np.squeeze(Train[np.where(Train[:,0] != clf_rf.predict(Train[:,1:num_features])),:])
Train_rf_good = np.squeeze(Train[np.where(Train[:,0] == clf_rf.predict(Train[:,1:num_features])),:])

Test_lr_bad = np.squeeze(Test[np.where(Test[:,0] != clf_logistic.predict(Test[:,1:num_features])),:])
Test_lr_good = np.squeeze(Test[np.where(Test[:,0] == clf_logistic.predict(Test[:,1:num_features])),:])
Test_rf_bad = np.squeeze(Test[np.where(Test[:,0] != clf_rf.predict(Test[:,1:num_features])),:])
Test_rf_good = np.squeeze(Test[np.where(Test[:,0] == clf_rf.predict(Test[:,1:num_features])),:])

# Fit the ml-health model
clf_health_letter_lr = hc.MlHealth()
clf_health_letter_lr.fit(Train_lr_good[:,1:num_features])
clf_health_letter_rf = hc.MlHealth()
clf_health_letter_rf.fit(Train_rf_good[:,1:num_features])


for a in range(0, num_points):
    ratio = a/(num_points-1)
    test_data_class_noise_lr = datasetMix.randomSample(Test_lr_good, Test_lr_bad, ratio, num_samples)
    test_data_class_noise_rf = datasetMix.randomSample(Test_rf_good, Test_rf_bad, ratio, num_samples)
    train_data_class_noise_lr = datasetMix.randomSample(Train_lr_good, Train_lr_bad, ratio, num_samples)
    train_data_class_noise_rf = datasetMix.randomSample(Train_rf_good, Train_rf_bad, ratio, num_samples)
    meanAccuracyLR_class_noise[plot_index,a] = clf_logistic.score(test_data_class_noise_lr[:,1:num_features], test_data_class_noise_lr[:,0])
    meanAccuracyRF_class_noise[plot_index,a] = clf_rf.score(test_data_class_noise_rf[:,1:num_features], test_data_class_noise_rf[:,0])
    meanHealthLR_class_noise[plot_index,a] = clf_health_letter_lr.score(test_data_class_noise_lr[:,1:num_features],N)
    meanHealthRF_class_noise[plot_index,a] = clf_health_letter_rf.score(test_data_class_noise_rf[:,1:num_features],N)
    meanAccuracyLR_class_noise_train[plot_index,a] = clf_logistic.score(train_data_class_noise_lr[:,1:num_features], train_data_class_noise_lr[:,0])
    meanAccuracyRF_class_noise_train[plot_index,a] = clf_rf.score(train_data_class_noise_rf[:,1:num_features], train_data_class_noise_rf[:,0])
    meanHealthLR_class_noise_train[plot_index,a] = clf_health_letter_lr.score(train_data_class_noise_lr[:,1:num_features],N)
    meanHealthRF_class_noise_train[plot_index,a] = clf_health_letter_rf.score(train_data_class_noise_rf[:,1:num_features],N)
    print("Yelp iteration: ",a)

print("meanAccuracyLR_class_noise[letter,a]", meanAccuracyLR_class_noise[plot_index,:])
print("meanAccuracyRF_class_noise[letter,a]", meanAccuracyRF_class_noise[plot_index,:])
print("meanHealthLR_class_noise[letter,a]", meanHealthLR_class_noise[plot_index,:])
print("meanHealthRF_class_noise[letter,a]", meanHealthRF_class_noise[plot_index,:])

print("meanAccuracyLR_class_noise_train[letter,a]", meanAccuracyLR_class_noise_train[plot_index,:])
print("meanAccuracyRF_class_noise_train[letter,a]", meanAccuracyRF_class_noise_train[plot_index,:])
print("meanHealthLR_class_noise_train[letter,a]", meanHealthLR_class_noise_train[plot_index,:])
print("meanHealthRF_class_noise_train[letter,a]", meanHealthRF_class_noise_train[plot_index,:])

import pickle
# Saving the objects:
pickle.dump( [meanAccuracyLR_class_noise, meanAccuracyRF_class_noise, meanHealthLR_class_noise, meanHealthRF_class_noise_train, meanAccuracyLR_class_noise_train, meanAccuracyRF_class_noise_train, meanHealthLR_class_noise_train, meanHealthRF_class_noise_train], open( "classification_subset_class_noise_"+ str(N) +".p", "wb" ) )
