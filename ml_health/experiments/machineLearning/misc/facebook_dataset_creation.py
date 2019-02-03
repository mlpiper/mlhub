"""
- Convert the facebook dataset into two parts, training and testing.
"""

import numpy as np
import os
import re

path = "/data-lake/ml-prototypes/regression/ml/facebook/"

###############################################################################
########## Read the dataset and save it in a numpy array ######################
###############################################################################
Train = np.genfromtxt(path + 'original/Dataset/Training/Features_Variant_1.csv', dtype = float, delimiter=",")

# read the test files into one numpy array
files = [f for f in os.listdir(path + 'original/Dataset/Testing/TestSet/') if re.match(r'Test_Case+.*\.csv', f)]

Test = np.genfromtxt(path + 'original/Dataset/Testing/TestSet/' + files[0], dtype = float, delimiter=",")
for a in range(1,len(files)):
    temp = np.genfromtxt(path + 'original/Dataset/Testing/TestSet/' + files[a], dtype = float, delimiter=",")
    Test = np.concatenate((Test, temp), axis=0)

### Divide the dataset into two batches based on the sugession made at data source
np.random.shuffle(Train)
np.random.shuffle(Test)

write_path = "/data-lake/ml-prototypes/regression/ml/facebook/"
np.savetxt(write_path + 'original/train/facebook_train.csv', Train, fmt='%.4e', delimiter=',')
np.savetxt(write_path + 'original/test/facebook_test.csv', Test, fmt='%.4e', delimiter=',')

