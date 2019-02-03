"""
- Generate the accuracy values for Polynomial Regression and Random Forest, when
trained and tested on unmodified data (5 datasets on data-lake).
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR

path = "/data-lake/ml-prototypes/regression/ml/"
#path = "/Users/sindhu/Desktop/data-lake/ml/regression/"

#clf_svr = SVR(C=1.0, epsilon=0.2)
clf_rf = RandomForestRegressor(max_depth=20, random_state=0, n_estimators=50)
clf_gbr = GradientBoostingRegressor()

# MSE values using svr regression and random forest
# Facebook (last column is the target value)
Train = np.genfromtxt(path + 'facebook/original/train/facebook_train.csv', dtype = float, delimiter=",")
Test = np.genfromtxt(path + 'facebook/original/test/facebook_test.csv', dtype = float, delimiter=",")
num_samples, num_features = Train.shape
print("Number of training samples in facebook: ", num_samples)
num_samples, num_features = Test.shape
print("Number of test samples in facebook: ", num_samples)

#clf_svr.fit(Train[:,0:num_features-1], Train[:,num_features-1])
clf_rf.fit(Train[:,0:num_features-1], Train[:,num_features-1])
clf_gbr.fit(Train[:,0:num_features-1], Train[:,num_features-1])

#mse_svr = mean_squared_error(Test[:,num_features-1],clf_svr.predict(Test[:,0:num_features-1]))
mse_rf = mean_squared_error(Test[:,num_features-1],clf_rf.predict(Test[:,0:num_features-1]))
mse_gbr = mean_squared_error(Test[:,num_features-1],clf_gbr.predict(Test[:,0:num_features-1]))
#print("The mean square value for svr regression: facebook dataset is: ", mse_svr)
print("The sqrt mean square value for rf regression: facebook dataset is: ", np.sqrt(mse_rf))
print("The sqrt mean square value for gb regression: facebook dataset is: ", np.sqrt(mse_gbr))


# Songs dataset (first value is the target)
Train = np.genfromtxt(path + 'songs/original/train/songs_train_subset.csv', dtype = float, delimiter=",")
Test = np.genfromtxt(path + 'songs/original/test/songs_test.csv', dtype = float, delimiter=",")
num_samples, num_features = Train.shape
print("Number of training samples in songs: ", num_samples)
num_samples, num_features = Test.shape
print("Number of test samples in songs: ", num_samples)

#clf_svr.fit(Train[:,1:Train.shape[1]], Train[:,0])
clf_rf.fit(Train[:,1:Train.shape[1]], Train[:,0])
clf_gbr.fit(Train[:,1:Train.shape[1]], Train[:,0])

#mse_svr = mean_squared_error(Test[:,0],clf_svr.predict(Test[:,1:Train.shape[1]]))
mse_rf = mean_squared_error(Test[:,0],clf_rf.predict(Test[:,1:Train.shape[1]]))
mse_gbr = mean_squared_error(Test[:,0],clf_gbr.predict(Test[:,1:Train.shape[1]]))
#print("The mean square value for svr regression: songs dataset is: ", mse_svr)
print("The sqrt mean square value for rf regression: songs dataset is: ", np.sqrt(mse_rf))
print("The sqrt mean square value for gb regression: songs dataset is: ", np.sqrt(mse_gbr))



# Blog (last column is the target value)
Train = np.genfromtxt(path + 'blog/original/train/blog_train.csv', dtype = float, delimiter=",")
Test = np.genfromtxt(path + 'blog/original/test/blog_test.csv', dtype = float, delimiter=",")
num_samples, num_features = Train.shape
print("Number of training samples in blog: ", num_samples)
num_samples, num_features = Test.shape
print("Number of test samples in blog: ", num_samples)

#clf_svr.fit(Train[:,0:num_features-1], Train[:,num_features-1])
clf_rf.fit(Train[:,0:num_features-1], Train[:,num_features-1])
clf_gbr.fit(Train[:,0:num_features-1], Train[:,num_features-1])

#mse_svr = mean_squared_error(Test[:,num_features-1],clf_svr.predict(Test[:,0:num_features-1]))
mse_rf = mean_squared_error(Test[:,num_features-1],clf_rf.predict(Test[:,0:num_features-1]))
mse_gbr = mean_squared_error(Test[:,num_features-1],clf_gbr.predict(Test[:,0:num_features-1]))
#print("The mean square value for svr regression: blog dataset is: ", mse_svr)
print("The sqrt mean square value for rf regression: blog dataset is: ", np.sqrt(mse_rf))
print("The sqrt mean square value for gb regression: blog dataset is: ", np.sqrt(mse_gbr))

# Turbine (last value is the target value)
Train = np.genfromtxt(path + 'turbine/original/train/turbine_train.csv', dtype = float, delimiter=",")
Test = np.genfromtxt(path + 'turbine/original/test/turbine_test.csv', dtype = float, delimiter=",")
num_samples, num_features = Train.shape
print("Number of training samples in turbine: ", num_samples)
num_samples, num_features = Test.shape
print("Number of test samples in turbine: ", num_samples)

#clf_svr.fit(Train[:,0:num_features-1], Train[:,num_features-1])
clf_rf.fit(Train[:,0:num_features-1], Train[:,num_features-1])
clf_gbr.fit(Train[:,0:num_features-1], Train[:,num_features-1])

#mse_svr = mean_squared_error(Test[:,num_features-1],clf_svr.predict(Test[:,0:num_features-1]))
mse_rf = mean_squared_error(Test[:,num_features-1],clf_rf.predict(Test[:,0:num_features-1]))
mse_gbr = mean_squared_error(Test[:,num_features-1],clf_gbr.predict(Test[:,0:num_features-1]))
#print("The mean square value for svr regression: turbine dataset is: ", mse_svr)
print("The sqrt mean square value for rf regression: turbine dataset is: ", np.sqrt(mse_rf))
print("The sqrt mean square value for gb regression: turbine dataset is: ", np.sqrt(mse_gbr))

# Video (last value is the target value)
Train_superset = np.genfromtxt(path + 'videos/original/train/videos_train.csv', dtype = float, delimiter=",")
Train = Train_superset[0:50000,:]
np.savetxt(path + "videos/original/train/videos_train_subset_yakov", Train, fmt = '%0.4f', delimiter=",")
Test = np.genfromtxt(path + 'videos/original/test/videos_test.csv', dtype = float, delimiter=",")
print(Train.shape)
num_samples, num_features = Train.shape
print("Number of training samples in video: ", num_samples)
num_samples, num_features = Test.shape
print("Number of test samples in video: ", num_samples)

#clf_svr.fit(Train[:,0:num_features-1], Train[:,num_features-1])
clf_rf.fit(Train[:,0:num_features-1], Train[:,num_features-1])
clf_gbr.fit(Train[:,0:num_features-1], Train[:,num_features-1])

#mse_svr = mean_squared_error(Test[:,num_features-1],clf_svr.predict(Test[:,0:num_features-1]))
mse_rf = mean_squared_error(Test[:,num_features-1],clf_rf.predict(Test[:,0:num_features-1]))
mse_gbr = mean_squared_error(Test[:,num_features-1],clf_gbr.predict(Test[:,0:num_features-1]))
#print("The mean square value for svr regression: video dataset is: ", mse_svr)
print("The sqrt mean square value for rf regression: video dataset is: ", np.sqrt(mse_rf))
print("The sqrt mean square value for gb regression: video dataset is: ", np.sqrt(mse_gbr))
