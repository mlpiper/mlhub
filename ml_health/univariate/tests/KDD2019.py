"""
This code produces results reported in Table .., .. and .. of the KDD 2019 paper
"""
import sys
import numpy as np
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0,'../../..')
import ml_health.univariate.univariate_health_calculator as hc
import ml_health.experiments.machineLearning.datasetMix as dm
import ml_health.experiments.machineLearning.randomNoise as rn

np.random.seed(seed=0)

# Read the TELCO dataset for all loads
#all_path = ["periodic_load", "flashcrowd_load", "linear_increase", "constant_load", "poisson_load"]
all_path = ["periodic_load", "flashcrowd_load", "linear_increase"]
telco_path = '/data-lake/ml-prototypes/classification/ml/realm-im2015-vod-traces/X_SAR/'

values = [10, 50, 100, 200]
RMSE = np.zeros((len(all_path),len(all_path),len(values)))
kl = np.zeros((len(all_path),len(all_path),len(values)))
wasserstein = np.zeros((len(all_path),len(all_path),len(values)))
similarity = np.zeros((len(all_path),len(all_path),len(values)))
confidence = np.zeros((len(all_path),len(all_path),len(values)))
accuracy = np.zeros((len(all_path),len(all_path),len(values)))
train_num = -1
test_num = -1
samples_num = -1

for train_path in all_path:
    train_num = train_num + 1
    for test_path in all_path:
        test_num = test_num + 1
        train_load_path = telco_path + train_path + "/"
        test_load_path = telco_path + test_path + "/"
        train_load = np.genfromtxt(train_load_path + "Train.csv", delimiter=',')
        validate_load = np.genfromtxt(train_load_path + "Validate.csv", delimiter=',')
        # Combine the train and validation data (50%)
        train_load = np.concatenate((train_load, validate_load))
        # Test data (50%)
        test_load_full = np.genfromtxt(test_load_path + "Test.csv", delimiter=',')
        N = 1
        for num_samples in values:
            samples_num = samples_num + 1
            print('Number of train samples in ', train_path, train_load.shape)
            print('Number of test samples in ', test_path, test_load_full.shape[0])
            test_load = test_load_full[np.random.randint(0,len(test_load_full),num_samples),:]
            #test_load = test_load_full
            # Fit the Primary algorithm so that feature importance can be utilized
            algorithm = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
            algorithm.fit(train_load[:,:-1], train_load[:,-1])

            # Retrieve the indices of top N features
            top_features_idx = np.argpartition(algorithm.feature_importances_, (-1)*N)[(-1)*N:] - 1

            # Fit the ml-health model on training data, use only the top N features
            distribution = hc.MlHealth()
            distribution.fit(train_load[:,top_features_idx])

            RMSE_temp, kl_temp, wasserstein_temp = distribution.other_scores(test_load[:,top_features_idx])
            RMSE[train_num, test_num, samples_num] = np.round(np.mean(RMSE_temp),3)
            kl[train_num, test_num, samples_num] = np.round(np.mean(kl_temp),3)
            wasserstein[train_num, test_num, samples_num] = np.round(np.mean(wasserstein_temp),3)
            print("RMSE, KL and Wasserstein when trained with load ", train_path ," and tested on load ", test_path," :", RMSE[train_num, test_num, samples_num], kl[train_num, test_num, samples_num], wasserstein[train_num, test_num, samples_num])
            similairty_temp = distribution.full_score(test_load[:,top_features_idx])
            similarity[train_num, test_num, samples_num] = np.min([np.round(np.mean(similairty_temp),2),1])
            print("Similarity score when trained with load ", train_path ," and tested on load ", test_path," :", similarity[train_num, test_num, samples_num])
            
            # Predict test model accuracy
            accuracy[train_num, test_num, samples_num] = np.round(algorithm.score(test_load[:,:-1],test_load[:,-1]),2)
            print("Prediction score when trained with load ", train_path ," and tested on load ", test_path," :", accuracy[train_num, test_num, samples_num])
            
            probabilities = algorithm.predict_proba(test_load[:,:-1])
            confidence[train_num, test_num, samples_num] = np.round(np.mean(np.max(probabilities,axis=1)),2)
            print("Confidence when trained with load ", train_path ," and tested on load ", test_path," :", confidence[train_num, test_num, samples_num])
        
        samples_num = -1
    test_num = -1


# Calculate correlation of different metrics for 10, 100 and full loads

print('RMSE', RMSE.T)
print('wasserstein', wasserstein.T)
print('kl', kl.T)
print('similarity', similarity.T)
print('confidence', confidence.T)
print('accuracy', accuracy.T)

RMSE_full = RMSE.flatten()
kl_full = kl.flatten()
wasserstein_full = wasserstein.flatten()
similarity_full = similarity.flatten()
confidence_full = confidence.flatten()
accuracy_full = accuracy.flatten()

RMSE_full_correlation = np.corrcoef(RMSE_full,accuracy_full)
kl_full_correlation = np.corrcoef(kl_full,accuracy_full)
wasserstein_full_correlation = np.corrcoef(wasserstein_full,accuracy_full)
similarity_full_correlation = np.corrcoef(similarity_full,accuracy_full)
confidence_full_correlation = np.corrcoef(confidence_full,accuracy_full)

print('RMSE_correlation_full', RMSE_full_correlation)
print('kl_correlation_full', kl_full_correlation)
print('wasserstein_full', wasserstein_full_correlation)
print('similarity_full',similarity_full_correlation)
print('confidence_full', confidence_full_correlation)


### Samsung data test
values = [10, 20, 30, 40, 50, 100, 200, 500, 1000, 5000, 10000, 20000]
# Path to datasets on data-lake
path = "/data-lake/ml-prototypes/classification/ml/"
np.random.seed(45)
# Read training and test datasets
Train = np.genfromtxt(path + 'samsung/original/train/samsung_train.csv', dtype = float, delimiter=",")
Test = np.genfromtxt(path + 'samsung/original/test/samsung_test.csv', dtype = float, delimiter=",")

algorithm = RandomForestClassifier(n_estimators=10, max_depth=2, random_state=0)
algorithm.fit(Train[:,1:],Train[:,0])

# Let the number of top features used be 3
N=10
# Retrieve the indices of top N features
feature_idx = np.argpartition(algorithm.feature_importances_, (-1)*N)[(-1)*N:]


# Fit the ml-health model on training data, use only the top N features
distribution = hc.MlHealth()
distribution.fit(Train[:,feature_idx])
RMSE = np.zeros((len(values),1))
distribution_score = np.zeros((len(values),1))
kl = np.zeros((len(values),1))
wasserstein = np.zeros((len(values),1))
confidence = np.zeros((len(values),1))
accuracy = np.zeros((len(values),1))
for a in range(0,len(values)):
    # Calculate the similarity score on test data for the top N features
    test_load = Test[np.random.randint(0,len(Test),values[a]),:]
    distribution_score_temp = distribution.full_score(test_load[:,feature_idx])
    RMSE_temp, kl_temp, wasserstein_temp = distribution.other_scores(test_load[:,feature_idx])
    RMSE[a] = np.round(np.mean(RMSE_temp),3)
    kl[a] = np.round(np.mean(kl_temp),3)
    wasserstein[a] = np.round(np.mean(wasserstein_temp),3)
    distribution_score[a] = np.min([round(np.mean(distribution_score_temp),2),1])
    probabilities = algorithm.predict_proba(test_load[:,1:])
    confidence[a] = np.round(np.mean(np.max(probabilities,axis=1)),2)
    accuracy[a] = np.round(algorithm.score(test_load[:,1:], test_load[:,0]),2)

print('RMSE', RMSE)
print('kl', kl)
print('wasserstein', wasserstein)
print('distribution_score', distribution_score)
print('confidence', confidence)
print('accuracy', accuracy)
RMSE_corr = np.corrcoef(RMSE.flatten(), accuracy.flatten())
kl_corr = np.corrcoef(kl.flatten(), accuracy.flatten())
wasserstein_corr = np.corrcoef(wasserstein.flatten(), accuracy.flatten())
similairty_corr = np.corrcoef(distribution_score.flatten(), accuracy.flatten())
confidence_corr = np.corrcoef(confidence.flatten(), accuracy.flatten())

print('RMSE', RMSE_corr)
print('kl_corr', kl_corr)
print('wasserstein_corr',wasserstein_corr)
print('similairty_corr',similairty_corr)
print('confidence_corr',confidence_corr)


# Add noise to test data, label is the first column
Test_noisy_random = np.copy(Test)
Test_noisy_random[:,1:] = rn.randomNoise(Test_noisy_random[:,1:], 1)

# Mix up the data
half_samples = np.int(len(Train)/2)
num_points = 10
num_samples = len(Test)
RMSE = np.zeros((num_points,1))
distribution_score = np.zeros((num_points,1))
kl = np.zeros((num_points,1))
wasserstein = np.zeros((num_points,1))
confidence = np.zeros((num_points,1))
accuracy = np.zeros((num_points,1))
for a in range(0, num_points):
    ratio = a/num_points
    print('ratio: ', ratio)
    Noisy_Test = dm.randomSample(Test_noisy_random,Test,ratio,num_samples)
    distribution_score_temp = distribution.full_score(Noisy_Test[:,feature_idx])
    RMSE_temp, kl_temp, wasserstein_temp = distribution.other_scores(Noisy_Test[:,feature_idx])
    RMSE[a] = np.round(np.mean(RMSE_temp),3)
    kl[a] = np.round(np.mean(kl_temp),3)
    wasserstein[a] = np.round(np.mean(wasserstein_temp),3)
    distribution_score[a] = np.min([round(np.mean(distribution_score_temp),2),1])
    probabilities = algorithm.predict_proba(Noisy_Test[:,1:])
    confidence[a] = np.round(np.mean(np.max(probabilities,axis=1)),2)
    accuracy[a] = np.round(algorithm.score(Noisy_Test[:,1:], Noisy_Test[:,0]),2)


print('RMSE noise', RMSE)
print('kl noise', kl)
print('wasserstein noise', wasserstein)
print('distribution_score noise', distribution_score)
print('confidence noise', confidence)
print('accuracy noise', accuracy)
RMSE_corr = np.corrcoef(RMSE.flatten(), accuracy.flatten())
kl_corr = np.corrcoef(kl.flatten(), accuracy.flatten())
wasserstein_corr = np.corrcoef(wasserstein.flatten(), accuracy.flatten())
similairty_corr = np.corrcoef(distribution_score.flatten(), accuracy.flatten())
confidence_corr = np.corrcoef(confidence.flatten(), accuracy.flatten())

print('RMSE noise', RMSE_corr)
print('kl_corr noise', kl_corr)
print('wasserstein_corr noise',wasserstein_corr)
print('similairty_corr noise',similairty_corr)
print('confidence_corr noise',confidence_corr)



import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from ml_health.experiments.machineLearning.misc.epsilon_regression import EpsilonDecider

values = [10, 20, 30, 40, 50, 100, 200, 500, 1000, 5000, 10000, 20000]
path = "/data-lake/ml-prototypes/regression/ml/"
# Video (last value is the target value)
Train_superset = np.genfromtxt(path + 'videos/original/train/videos_train.csv', dtype = float, delimiter=",")
Train = Train_superset
np.savetxt(path + "videos/original/train/videos_train_subset_yakov", Train, fmt = '%0.4f', delimiter=",")
Test = np.genfromtxt(path + 'videos/original/test/videos_test.csv', dtype = float, delimiter=",")
num_samples, num_features = Train.shape
print("Number of training samples in video: ", num_samples)
num_samples, num_features = Test.shape
print("Number of test samples in video: ", num_samples)
algorithm = RandomForestRegressor(max_depth=50, random_state=0, n_estimators=10)
algorithm.fit(Train[:,:-1], Train[:,-1])

# Let the number of top features used be 3
N=5
# Retrieve the indices of top N features
feature_idx = np.argpartition(algorithm.feature_importances_, (-1)*N)[(-1)*N:] - 1

# Fit the ml-health model on training data, use only the top N features
distribution = hc.MlHealth()
distribution.fit(Train[:,feature_idx])
RMSE = np.zeros((len(values),1))
distribution_score = np.zeros((len(values),1))
kl = np.zeros((len(values),1))
wasserstein = np.zeros((len(values),1))
confidence = np.zeros((len(values),1))
accuracy = np.zeros((len(values),1))
print('shape of superset', Train_superset.shape)


for a in range(0,len(values)):
    # Calculate the similarity score on test data for the top N features
    test_load = Test[np.random.randint(0,len(Test),values[a]),:]
    distribution_score_temp = distribution.full_score(test_load[:,feature_idx])
    RMSE_temp, kl_temp, wasserstein_temp = distribution.other_scores(test_load[:,feature_idx])
    RMSE[a] = np.round(np.mean(RMSE_temp),3)
    kl[a] = np.round(np.mean(kl_temp),3)
    wasserstein[a] = np.round(np.mean(wasserstein_temp),3)
    distribution_score[a] = np.round([np.min(np.mean(distribution_score_temp,1))],2)
    accuracy[a] = np.round(algorithm.score(test_load[:,:-1], test_load[:,-1]),2)

print('RMSE', RMSE)
print('kl', kl)
print('wasserstein', wasserstein)
print('distribution_score', distribution_score)
print('accuracy', accuracy)
RMSE_corr = np.corrcoef(RMSE.flatten(), accuracy.flatten())
kl_corr = np.corrcoef(kl.flatten(), accuracy.flatten())
wasserstein_corr = np.corrcoef(wasserstein.flatten(), accuracy.flatten())
similairty_corr = np.corrcoef(distribution_score.flatten(), accuracy.flatten())
print('RMSE', RMSE_corr)
print('kl_corr', kl_corr)
print('wasserstein_corr',wasserstein_corr)
print('similairty_corr',similairty_corr)


# Add noise to test data, label is the first column
Test_noisy_random = np.copy(Test)
Test_noisy_random[:,:-1] = rn.randomNoise(Test_noisy_random[:,:-1], 1)

# Mix up the data
half_samples = np.int(len(Train)/2)
num_points = 10
num_samples = len(Test)
RMSE = np.zeros((num_points,1))
distribution_score = np.zeros((num_points,1))
kl = np.zeros((num_points,1))
wasserstein = np.zeros((num_points,1))
confidence = np.zeros((num_points,1))
accuracy = np.zeros((num_points,1))

for a in range(0, num_points):
    ratio = a/num_points
    print('ratio: ', ratio)
    Noisy_Test = dm.randomSample(Test_noisy_random,Test,ratio,num_samples)
    distribution_score_temp = distribution.full_score(Noisy_Test[:,feature_idx])
    RMSE_temp, kl_temp, wasserstein_temp = distribution.other_scores(Noisy_Test[:,feature_idx])
    RMSE[a] = np.round(np.mean(RMSE_temp),3)
    kl[a] = np.round(np.mean(kl_temp),3)
    wasserstein[a] = np.round(np.mean(wasserstein_temp),3)
    distribution_score[a] = np.min([round(np.mean(distribution_score_temp),2),1])
    accuracy[a] = np.round(algorithm.score(Noisy_Test[:,:-1], Noisy_Test[:,-1]),2)


print('RMSE noise', RMSE)
print('kl noise', kl)
print('wasserstein noise', wasserstein)
print('distribution_score noise', distribution_score)
print('accuracy noise', accuracy)
RMSE_corr = np.corrcoef(RMSE.flatten(), accuracy.flatten())
kl_corr = np.corrcoef(kl.flatten(), accuracy.flatten())
wasserstein_corr = np.corrcoef(wasserstein.flatten(), accuracy.flatten())
similairty_corr = np.corrcoef(distribution_score.flatten(), accuracy.flatten())

print('RMSE noise', RMSE_corr)
print('kl_corr noise', kl_corr)
print('wasserstein_corr noise',wasserstein_corr)
print('similairty_corr noise',similairty_corr)


