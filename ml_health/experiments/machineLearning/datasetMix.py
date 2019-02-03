"""
This code mixes samples from two datasets in the ratio specified
"""
import numpy as np

def randomSample(data1, data2, ratio, num_samples):
    num_samples_1 = np.int(ratio*num_samples)
    num_samples_2 = num_samples - num_samples_1
    # pick num_samples_1 number of samples from data1
    if(num_samples_1 != 0):
        random_index_1 = np.random.randint(0,data1.shape[0],size=(num_samples_1,))
        samples_data1 = data1[random_index_1,:]
    else:
        samples_data1 = data1[0,:]
        samples_data1 = samples_data1.reshape(1,data1.shape[1])
    # pick num_samples_2 number of samples from data2
    if(num_samples_2 != 0):
        random_index_2 = np.random.randint(0,data2.shape[0],size=(num_samples_2,))
        samples_data2 = data2[random_index_2,:]
    else:
        samples_data2 = data2[0,:]
        samples_data2 = samples_data2.reshape(1,data2.shape[1])
    output_dataset = np.concatenate((samples_data1,samples_data2))
    np.random.shuffle(output_dataset)
    return output_dataset
