"""
This code adds noise to data within a scalar number of standard deviations
"""
import numpy as np

def randomContNoise(data, scale):
    mean_attributes = np.mean(data, axis = 0)
    std_attributes = scale * np.std(data, axis = 0)
    noisy_data = data + np.random.normal(0, std_attributes, (data.shape))
    return noisy_data

def randomIntNoise(data):
    # Randomize ~ 15% data
    out =  np.random.choice(np.unique(data),(data.shape))
    out[0:np.int(len(data)/7)] = data[0:np.int(len(data)/7)]
    return out

def randomNoise(data, scale):
    num_samples, num_attributes = data.shape
    noisy_data = data
    for a in range(0,num_attributes):
        if(len(np.unique(data[:,a])) < 30):
            noisy_data[:,a] = randomIntNoise(data[:,a])
        else:
            noisy_data[:,a] = randomContNoise(data[:,a], scale)
    return noisy_data
