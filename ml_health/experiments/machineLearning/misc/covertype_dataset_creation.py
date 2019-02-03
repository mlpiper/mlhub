"""
- Convert the Yelp dataset into two parts, training and testing.
- Create a noisy test dataset
"""

import pandas as pd
import numpy as np
import sklearn.preprocessing as preprocessing


path = "/data-lake/ml-prototypes/classification/ml/covertype/"


###############################################################################
########## Read the dataset and save it in a numpy array ######################
###############################################################################

data = np.genfromtxt(path + 'original/UCIHARDataset/train/y_train.txt', dtype = float, delimiter=",")

np.random.shuffle(data)

likely_cat = np.ones((data.shape))

# If a feature has less than max_categories, we assume it is continuous.
# The datasets we use in our experiemnts typically have less than 30 categories
# when a feature is categorical.
max_categories = 30
for a in range(0,data.shape[1]):
    likely_cat[a] = len(np.unique(data[:,a])) < max_categories

Train = data[0:2*len(data)/3,:]
Test = data[2*len(data)/3:len(data),:]


write_path = "/data-lake/ml-prototypes/classification/ml/covertype/"
np.savetxt(write_path + 'original/train/covertype_train.csv', Train, fmt='%.4e', delimiter=',')
np.savetxt(write_path + 'original/test/covertype_test.csv', Test, fmt='%.4e', delimiter=',')

###############################################################################
######### Add noise to the test dataset #######################################
###############################################################################

Test_noisy = Test
for a in range(0,Test.shape[1]-1):
    if(likely_cat[a] == 1):
        Test_noisy[:,a] = np.random.randint(0,2,size=[Test.shape[0],]) + Test[:,a]
    else:
        Test_noisy[:,a] = 100*np.random.rand(Test.shape[0],) + Test[:,a]

np.savetxt(write_path + 'noisy/covertype_test_noisy.csv', Test_noisy, fmt='%.4e', delimiter=',')
