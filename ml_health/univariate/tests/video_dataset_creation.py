"""
- Convert the youtube video dataset into two parts, training and testing.
"""

import numpy as np
import pandas as pd

path = "/data-lake/ml-prototypes/regression/ml/videos/"

###############################################################################
########## Read the dataset and save it in a numpy array ######################
###############################################################################
# Read as a pandas dataframe followed by encoding of cateogrical features.

Data = pd.read_csv(path + 'original/online_video_dataset/transcoding_mesurment.tsv', sep='\t', header = 0)
Data = Data.drop(["id"],axis=1)

Data["codec"] = Data["codec"].astype('category')
Data["o_codec"] = Data["o_codec"].astype('category')

Data["codec"] = Data["codec"].cat.codes
Data["o_codec"] = Data["o_codec"].cat.codes

Data_array = Data.values

# Divide the dataset with 2/3rd as training and 1/3rd as test.
Train = Data_array[0:np.int(2*len(Data_array)/3),:]
Test = Data_array[np.int(2*len(Data_array)/3):len(Data_array),:]
# shuffle the data
np.random.shuffle(Train)
np.random.shuffle(Test)


write_path = "/data-lake/ml-prototypes/regression/ml/videos/"
np.savetxt(write_path + 'original/train/videos_train.csv', Train, fmt='%.4e', delimiter=',')
np.savetxt(write_path + 'original/test/videos_test.csv', Test, fmt='%.4e', delimiter=',')
