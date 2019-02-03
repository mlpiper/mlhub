"""
- Convert the Samsung dataset into two parts, training and testing.
- Create a noisy test dataset
"""

import numpy as np

path = "/data-lake/ml-prototypes/classification/ml/samsung/"


###############################################################################
########## Read the dataset and save it in a numpy array ######################
###############################################################################
dataset_train = np.genfromtxt(path + 'original/UCIHARDataset/train/X_train.txt', dtype = float)
labels_train = np.genfromtxt(path + 'original/UCIHARDataset/train/y_train.txt', dtype = float, delimiter=",")
dataset_test = np.genfromtxt(path + 'original/UCIHARDataset/test/X_test.txt', dtype = float)
labels_test = np.genfromtxt(path + 'original/UCIHARDataset/test/y_test.txt', dtype = float, delimiter=",")


###############################################################################
######### Concatenate labels of the dataset and write them ####################
###############################################################################

labels_train = np.reshape(labels_train,(len(labels_train),1))
labels_test = np.reshape(labels_test,(len(labels_test),1))
Train = np.concatenate((labels_train,dataset_train),axis=1)
Test = np.concatenate((labels_test,dataset_test),axis=1)

write_path = "/data-lake/ml-prototypes/classification/ml/samsung/"
np.savetxt(write_path + 'original/train/samsung_train.csv', Train, fmt='%.4e', delimiter=',')
np.savetxt(write_path + 'original/test/samsung_test.csv', Test, fmt='%.4e', delimiter=',')


###############################################################################
######### Add noise to the test dataset #######################################
###############################################################################

# Random noise is added to the features (not the labels)
Test_noisy = 2*np.random.randn(Test.shape[0],Test.shape[1]-1) + Test[:,1:Test.shape[1]]
# Labels are extracted and concatenated to the noisy dataset
label = Test[:,0].reshape(Test.shape[0],-1)
Test_noisy = np.concatenate((label,Test_noisy),axis=1)

np.savetxt(write_path + 'noisy/samsung_test_noisy.csv', Test_noisy, fmt='%.4e', delimiter=',')
