"""
- Convert the youtube video dataset into two parts, training and testing.
"""

import numpy as np
import pandas as pd
import argparse
import os

def parse_args():
    """
    Parse Arguments from component
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="original data path", default="/data-lake/ml-prototypes/regression/ml/videos/")
    parser.add_argument("--output", help="output data path", default="/data-lake/ml-prototypes/regression/ml/videos/")
    options = parser.parse_args()
    return options

options = parse_args()
#path = "/data-lake/ml-prototypes/regression/ml/videos/"
#write_path = "/data-lake/ml-prototypes/regression/ml/videos/"
path = options.path
write_path = options.output

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


os.makedirs(write_path + '/original/train')
os.makedirs(write_path + '/original/test')

np.savetxt(write_path + 'original/train/videos_train.csv', Train, fmt='%.4e', delimiter=',')
np.savetxt(write_path + 'original/test/videos_test.csv', Test, fmt='%.4e', delimiter=',')
