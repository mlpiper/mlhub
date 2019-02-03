"""
- Convert the Songs dataset into two parts, training and testing.
"""

import numpy as np

path = "/Users/sindhu/Desktop/data-lake/ml/regression/songs/"

###############################################################################
########## Read the dataset and save it in a numpy array ######################
###############################################################################
dataset = np.genfromtxt(path + 'original/YearPredictionMSD.txt', dtype = float, delimiter=",")


# Divide the dataset into two batches
# 463715 is chosen based on the sugession made at data source from which this dataset is downloaded.
Train = dataset[0:463715,:]
Test = dataset[463715:len(dataset),:]
np.random.shuffle(Train)
np.random.shuffle(Test)



write_path = "/Users/sindhu/Desktop/data-lake/ml/regression/songs/"
np.savetxt(write_path + 'original/train/songs_train.csv', Train, fmt='%.4e', delimiter=',')
np.savetxt(write_path + 'original/test/songs_test.csv', Test, fmt='%.4e', delimiter=',')

