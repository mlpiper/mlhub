"""
This code creates a class called normalizer. It provides the following functions:
    fit: Find the normalization coefficients based on given datai (only for continuous values).
    transform: Normalize a given set of data (only for continuous values)
    fit_transform: Find the normalization coefficients and then normalize the data (only for continuous values).
"""

import numpy as np

class normalizer:
    def __init__(self):
        self._norm_type = 1 # Corresponds to normalization to a value between [0,1]
        self._norm_values = []
        self._continuous_index = []
        self._num_attributes = []

    """
    This function calculated the normalization coefficients
    """
    def fit(self, training_dataset):
        # Detect continuous values
        num_samples, self._num_attributes = training_dataset.shape
        self._continuous_index = np.zeros((self._num_attributes, 1))
        for a in range(0,self._num_attributes):
            if(len(np.unique(training_dataset[:,a])) > 30):
                self._continuous_index[a] = 1
        # Detect the minimum and maximum value
        self._norm_values = np.zeros((self._num_attributes,2))
        self._norm_values[:,0] = np.min(training_dataset)
        self._norm_values[:,1] = np.max(training_dataset)
    def transform(self, dataset):
        # Normalize the dataset based on fit values
        for a in range(0,self._num_attributes):
            if(self._continuous_index[a]==1):
                dataset[:,a] = (dataset[:,a] - self._norm_values[a,0])/(self._norm_values[a,1] - self._norm_values[a,0])
        return dataset
"""
Test code
import numpy as np
import normalizeData
data = np.random.randn(100,10)
data[:,9] = 2
normalizeData = normalizeData.normalizer()
normalizeData.fit(data)
newData = normalizeData.transform(data)
"""
