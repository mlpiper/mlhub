"""
- Convert the Blog dataset into two csv files, training and testing.
"""

import numpy as np
import os
import re

path = "/data-lake/ml-prototypes/regression/blog/"

###############################################################################
########## Read the dataset and save it in a numpy array ######################
###############################################################################
Train = np.genfromtxt(path + 'original/BlogFeedback/blogData_train.csv', dtype = float, delimiter=",")

# read the test files into one numpy array
files = [f for f in os.listdir(path + 'original/BlogFeedback/') if re.match(r'blogData_test+.*\.csv', f)]

Test = np.genfromtxt(path + 'original/BlogFeedback/' + files[0], dtype = float, delimiter=",")
for a in range(1,len(files)):
    temp = np.genfromtxt(path + 'original/BlogFeedback/' + files[a], dtype = float, delimiter=",")
    Test = np.concatenate((Test, temp), axis=0)


np.random.shuffle(Train)
np.random.shuffle(Test)

write_path = "/data-lake/ml-prototypes/regression/blog/"
np.savetxt(write_path + 'original/train/blog_train.csv', Train, fmt='%.4e', delimiter=',')
np.savetxt(write_path + 'original/test/blog_test.csv', Test, fmt='%.4e', delimiter=',')

