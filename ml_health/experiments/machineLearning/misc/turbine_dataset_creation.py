"""
- Convert the turbine dataset into two parts, training and testing.
"""

import numpy as np

path = "/data-lake/ml-prototypes/regression/ml/turbine/"

###############################################################################
########## Read the dataset and save it in a numpy array ######################
###############################################################################
data_temp = np.genfromtxt(path + 'original/UCI_CBM_Dataset/data.txt', dtype = float, delimiter="   ")

# Feature 17 and 18 represent GT Compressor decay state coefficient and GT Turbine decay state coefficient.
# We predict only the turbine decay coefficient and hence drop the compressor decay state coefficient.
Data = data_temp[:,0:17]
Data[:,16] = data_temp[:,17]

# 2/3rd dataset is used for training and 1/3rd is used for testing
Train = Data[0:np.int(2*len(Data)/3),:]
Test = Data[np.int(2*len(Data)/3):len(Data),:]

### Divide the dataset into two batches based on the sugession made at data source
np.random.shuffle(Train)
np.random.shuffle(Test)

write_path = "/data-lake/ml-prototypes/regression/ml/turbine/"
np.savetxt(write_path + 'original/train/turbine_train.csv', Train, fmt='%.4e', delimiter=',')
np.savetxt(write_path + 'original/test/turbine_test.csv', Test, fmt='%.4e', delimiter=',')

