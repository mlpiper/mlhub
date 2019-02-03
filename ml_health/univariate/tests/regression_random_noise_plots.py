"""
This code generates 2 plots
plot 1: For support vector regression, it plots the RMSE vs health-score graph for
        5 datasets, where random noise is added to datasets.
plot 2: For random forest regression, it plots the RMSE vs health-score graph for
        5 datasets, where random noise is added to datasets.
"""
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import sys
sys.path.insert(0,'../../..')
import ml_health.experiments.machineLearning.datasetMix as dm
import ml_health.experiments.machineLearning.randomNoise as rn
import ml_health.univariate.univariate_health_calculator as hc

# Path to datasets on data-lake
path = "/data-lake/ml-prototypes/regression/ml/"
# Number of points in the plots per curve
num_points = 10
num_datasets = 5

# Initialize the variable to plot
meanMSESVR_random_noise = np.zeros((num_datasets, num_points))
meanMSERF_random_noise = np.zeros((num_datasets, num_points))
meanHealth_random_noise = np.zeros((num_datasets, num_points))

# Initialize support vector regression and random forest regression algorithms
clf_svr = SVR(C=1.0, epsilon=0.2)
clf_rf = RandomForestRegressor(max_depth=20, random_state=0, n_estimators=50)

# First dataset is Facebook
plot_index = 0

# Read training and test datasets
Train = np.genfromtxt(path + 'facebook/original/train/facebook_train.csv', dtype = float, delimiter=",")
Test = np.genfromtxt(path + 'facebook/original/test/facebook_test.csv', dtype = float, delimiter=",")
num_samples, num_features = Test.shape

# Add noise to test data, label is the first column
Test_noisy_random = np.copy(Test)
Test_noisy_random[:,0:num_features-1] = rn.randomNoise(Test[:,0:num_features-1], 10)

# Fit using logistic regression and random forest
clf_svr.fit(Train[:,0:num_features-1], Train[:,num_features-1])
clf_rf.fit(Train[:,0:num_features-1], Train[:,num_features-1])

# Fit the ml-health model
clf_health_facebook = hc.MlHealth()
clf_health_facebook.fit(Train[:,0:num_features-1])

# Calculate the accuracy of both algorithms and corresponding ml-health scores
for a in range(0, num_points):
    ratio = a/(num_points-1)
    test_data_random_noise = dm.randomSample(Test,Test_noisy_random, ratio, num_samples)
    meanMSESVR_random_noise[plot_index,a] = mean_squared_error(clf_svr.predict(test_data_random_noise[:,0:num_features-1]), test_data_random_noise[:,num_features-1])
    meanMSERF_random_noise[plot_index,a] = mean_squared_error(clf_rf.predict(test_data_random_noise[:,0:num_features-1]), test_data_random_noise[:,num_features-1])
    meanHealth_random_noise[plot_index,a] = clf_health_facebook.score(test_data_random_noise[:,0:num_features-1])
    print("Facebook iteration: ",a)

print("meanMSESVR_random_noise[facebook,a]", meanMSESVR_random_noise[plot_index,:])
print("meanMSERF_random_noise[facebook,a]", meanMSERF_random_noise[plot_index,:])
print("meanHealth_ransom_noise[facebook,a]", meanHealth_random_noise[plot_index,:])

# Songs data
plot_index = 1

# Read the test and training datasets
Train = np.genfromtxt(path + 'songs/original/train/songs_train_subset.csv', dtype = float, delimiter=",")
Test = np.genfromtxt(path + 'songs/original/test/songs_test.csv', dtype = float, delimiter=",")
num_samples, num_features = Test.shape

# Add noise to test data, label is the first column
Test_noisy_random = np.copy(Test)
Test_noisy_random[:,1:num_features] = rn.randomNoise(Test[:,1:num_features], 10)

# Fit the algorithms with this dataset
clf_svr.fit(Train[:,1:num_features], Train[:,0])
clf_rf.fit(Train[:,1:num_features], Train[:,0])

# Fit the ml-health model
clf_health_songs = hc.MlHealth()
clf_health_songs.fit(Train[:,1:num_features])

# Calculate the accuracy of both algorithms and corresponding health score
for a in range(0, num_points):
    ratio = a/(num_points-1)
    test_data_random_noise = dm.randomSample(Test,Test_noisy_random,ratio,num_samples)
    meanMSESVR_random_noise[plot_index,a] = mean_squared_error(clf_svr.predict(test_data_random_noise[:,1:num_features]), test_data_random_noise[:,0])
    meanMSERF_random_noise[plot_index,a] = mean_squared_error(clf_rf.predict(test_data_random_noise[:,1:num_features]), test_data_random_noise[:,0])
    meanHealth_random_noise[plot_index,a] = clf_health_songs.score(test_data_random_noise[:,1:num_features])
    print("Songs iteration: ",a)

print("meanAccuracyMSESVR_random_noise[songs,a]", meanMSESVR_random_noise[plot_index,:])
print("meanAccuracyMSERF_random_noise[songs,a]", meanMSERF_random_noise[plot_index,:])
print("meanHealth_ransom_noise[songs,a]", meanHealth_random_noise[plot_index,:])

# Turbine data
plot_index = 2

# Read the test and train datasets
Train = np.genfromtxt(path + 'turbine/original/train/turbine_train.csv', dtype = float, delimiter=",")
Test = np.genfromtxt(path + 'turbine/original/test/turbine_test.csv', dtype = float, delimiter=",")
num_samples, num_features = Test.shape

# Add noise to test data, label is the first column
Test_noisy_random = np.copy(Test)
Test_noisy_random[:,0:num_features-1] = rn.randomNoise(Test[:,0:num_features-1], 10)

# Fit using logistic regression and random forest
clf_svr.fit(Train[:,0:num_features-1], Train[:,num_features-1])
clf_rf.fit(Train[:,0:num_features-1], Train[:,num_features-1])


# Fit the ml-health model
clf_health_turbine = hc.MlHealth()
clf_health_turbine.fit(Train[:,0:num_features-1])

# Calculate the accuracy of both algorithms and corresponding ml-health scores
for a in range(0, num_points):
    ratio = a/(num_points-1)
    test_data_random_noise = dm.randomSample(Test,Test_noisy_random, ratio, num_samples)
    meanMSESVR_random_noise[plot_index,a] = mean_squared_error(clf_svr.predict(test_data_random_noise[:,0:num_features-1]), test_data_random_noise[:,num_features-1])
    meanMSERF_random_noise[plot_index,a] = mean_squared_error(clf_rf.predict(test_data_random_noise[:,0:num_features-1]), test_data_random_noise[:,num_features-1])
    meanHealth_random_noise[plot_index,a] = clf_health_turbine.score(test_data_random_noise[:,0:num_features-1])
    print("Turbine iteration: ",a)

print("meanMSESVR_random_noise[turbine,a]", meanMSESVR_random_noise[plot_index,:])
print("meanMSERF_random_noise[turbine,a]", meanMSERF_random_noise[plot_index,:])
print("meanHealth_ransom_noise[turbine,a]", meanHealth_random_noise[plot_index,:])

# Video cover
plot_index = 3

# Read the training and test datasets
Train = np.genfromtxt(path + 'videos/original/train/videos_train.csv', dtype = float, delimiter=",")
Test = np.genfromtxt(path + 'videos/original/test/videos_test.csv', dtype = float, delimiter=",")
num_samples, num_features = Test.shape


Test_noisy_random = np.copy(Test)
Test_noisy_random[:,0:num_features-1] = rn.randomNoise(Test[:,0:num_features-1], 10)

# Fit the algorithms with this dataset
clf_svr.fit(Train[:,0:num_features-1], Train[:,num_features-1])
clf_rf.fit(Train[:,0:num_features-1], Train[:,num_features-1])

# Fit the ml-health model
clf_health_video = hc.MlHealth()
clf_health_video.fit(Train[:,0:num_features-1])

for a in range(0, num_points):
    ratio = a/(num_points-1)
    test_data_random_noise = dm.randomSample(Test,Test_noisy_random,ratio,num_samples)
    meanMSESVR_random_noise[plot_index,a] = mean_squared_error(clf_svr.predict(test_data_random_noise[:,0:num_features-1]), test_data_random_noise[:,num_features-1])
    meanMSERF_random_noise[plot_index,a] = mean_squared_error(clf_rf.predict(test_data_random_noise[:,0:num_features-1]), test_data_random_noise[:,num_features-1])
    meanHealth_random_noise[plot_index,a] = clf_health_video.score(test_data_random_noise[:,0:num_features-1])
    print("Video iteration: ",a)
print("meanAccuracyLR_random_noise[video,a]", meanMSESVR_random_noise[plot_index,:])
print("meanAccuracyRF_random_noise[video,a]", meanMSERF_random_noise[plot_index,:])
print("meanHealth_ransom_noise[video,a]", meanHealth_random_noise[plot_index,:])

# Blog cover
plot_index = 4

# Read the training and test datasets
Train = np.genfromtxt(path + 'blog/original/train/blog_train.csv', dtype = float, delimiter=",")
Test = np.genfromtxt(path + 'blog/original/test/blog_test.csv', dtype = float, delimiter=",")
num_samples, num_features = Test.shape

Test_noisy_random = np.copy(Test)
Test_noisy_random[:,0:num_features-1] = rn.randomNoise(Test[:,0:num_features-1], 10)

# Fit the algorithms with this dataset
clf_svr.fit(Train[:,0:num_features-1], Train[:,num_features-1])
clf_rf.fit(Train[:,0:num_features-1], Train[:,num_features-1])

# Fit the ml-health model
clf_health_blog = hc.MlHealth()
clf_health_blog.fit(Train[:,0:num_features-1])

for a in range(0, num_points):
    ratio = a/(num_points-1)
    test_data_random_noise = dm.randomSample(Test,Test_noisy_random,ratio,num_samples)
    meanMSESVR_random_noise[plot_index,a] = mean_squared_error(clf_rf.predict(test_data_random_noise[:,0:num_features-1]), test_data_random_noise[:,num_features-1])
    meanMSERF_random_noise[plot_index,a] = mean_squared_error(clf_rf.predict(test_data_random_noise[:,0:num_features-1]), test_data_random_noise[:,num_features-1])
    meanHealth_random_noise[plot_index,a] = clf_health_blog.score(test_data_random_noise[:,0:num_features-1])
    print("Video iteration: ",a)
print("meanAccuracyLR_random_noise[video,a]", meanMSESVR_random_noise[plot_index,:])
print("meanAccuracyRF_random_noise[video,a]", meanMSERF_random_noise[plot_index,:])
print("meanHealth_ransom_noise[video,a]", meanHealth_random_noise[plot_index,:])

import pickle

# Saving the objects:
pickle.dump( [meanMSESVR_random_noise, meanMSERF_random_noise, meanHealth_random_noise], open( "regression_plot_random_noise.p", "wb" ) )

