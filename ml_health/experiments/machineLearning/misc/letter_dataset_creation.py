"""
- Convert the letter type dataset into two parts, training and testing.
- Create a noisy test dataset
"""

import pandas as pd
import numpy as np


path = "/data-lake/ml-prototypes/classification/ml/letter/"


###############################################################################
########## Read the dataset and save it in a numpy array ######################
###############################################################################


original_data = pd.read_csv(path + "original/letter-recognition.data")

label, index = pd.factorize(original_data[original_data.columns[0]])

data = original_data.values
data[:,0] = label

# Set 2/3 of data to train set and the rest for testing
Train = data[0:np.int(2*len(data)/3),:]
Test = data[np.int(2*len(data)/3):len(data),:]


write_path = "/data-lake/ml-prototypes/classification/ml/letter/"
np.savetxt(write_path + 'original/train/letter_train.csv', Train, fmt='%.4e', delimiter=',')
np.savetxt(write_path + 'original/test/letter_test.csv', Test, fmt='%.4e', delimiter=',')

###############################################################################
######### Add noise to the test dataset #######################################
###############################################################################

# Noise is added only to the features
Test_noisy = np.random.randint(0,5,(Test.shape[0],Test.shape[1]-1)) + Test[:,1:Test.shape[1]]
# Labels are extracted and concatenated to the noisy dataset
label = Test[:,0].reshape(Test.shape[0],-1)
Test_noisy = np.concatenate((label,Test_noisy),axis=1)


np.savetxt(write_path + 'noisy/letter_test_noisy.csv', Test_noisy, fmt='%.4e', delimiter=',')
