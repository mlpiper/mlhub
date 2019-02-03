"""
- Convert the Yelp dataset into two parts, training and testing.
- Create a noisy test dataset
"""

import numpy as np
import arff

path = "/data-lake/ml-prototypes/classification/ml/yelp/"


###############################################################################
########## Read the dataset and save it in a numpy array ######################
###############################################################################
dataset_train = arff.load(open(path + 'original/Yelp/train.arff', 'r'))
data_train = np.array(dataset_train['data'])

dataset_test = arff.load(open(path + 'original/Yelp/test.arff', 'r'))
data_test = np.array(dataset_test['data'])

data_train = data_train.astype(float)
data_test = data_test.astype(float)

binary_labels_train = data_train[:,668:672]
label_train = np.zeros((len(binary_labels_train),1))
labels_train = np.sum([8,4,2,1]*binary_labels_train,axis=1)

binary_labels_test = data_test[:,668:672]
label_test = np.zeros((len(binary_labels_test),1))
labels_test = np.sum([8,4,2,1]*binary_labels_test,axis=1)

labels_train = labels_train.reshape(-1,1)
labels_test = labels_test.reshape(-1,1)
Train = np.concatenate((labels_train,data_train),axis=1)
Test = np.concatenate((labels_test,data_test),axis=1)

### Divide the dataset into two batches.
np.random.shuffle(Train)
np.random.shuffle(Test)



write_path = "/data-lake/ml-prototypes/classification/ml/yelp/"
np.savetxt(write_path + 'original/train/yelp_train.csv', Train, fmt='%.4e', delimiter=',')
np.savetxt(write_path + 'original/test/yelp_test.csv', Test, fmt='%.4e', delimiter=',')

###############################################################################
######### Add noise to the test dataset #######################################
###############################################################################

# Add random binary noise to the features alone
Test_noisy = np.random.randint(0,2,size=[Test.shape[0],Test.shape[1]-1]) + Test[:,1:Test.shape[1]]
# Labels are extracted and concatenated to the noisy dataset
label = Test[:,0].reshape(Test.shape[0],-1)
Test_noisy = np.concatenate((label,Test_noisy),axis=1)

np.savetxt(write_path + 'noisy/yelp_test_noisy.csv', Test_noisy, fmt='%.4e', delimiter=',')
