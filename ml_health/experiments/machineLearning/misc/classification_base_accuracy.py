"""
- Generate the accuracy values for Logistic Regression and Random Forest, when
trained and tested on unmodified data (5 datasets on data-lake).
- Save the mis-classified samples, so that they can be used for dropping predictive performance.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

path = "/data-lake/ml-prototypes/classification/ml/"

clf_logistic = LogisticRegression(C=0.025, multi_class='multinomial', penalty='l1', solver='saga', tol=0.1)
# Accuracy values using logistic regression (save mis-classified samples)
# Samsung (save mis-classified samples)
Train = np.genfromtxt(path + 'samsung/original/train/samsung_train.csv', dtype = float, delimiter=",")
Test = np.genfromtxt(path + 'samsung/original/test/samsung_test.csv', dtype = float, delimiter=",")
clf_logistic.fit(Train[:,1:Train.shape[1]], Train[:,0])
mean_accuracy = clf_logistic.score(Test[:,1:Train.shape[1]], Test[:,0])
print("The mean accuracy value for LR: samsung dataset is: ", mean_accuracy)
# misclassified samples in the dataset
misclassified = np.where(Test[:,0] != clf_logistic.predict(Test[:,1:Train.shape[1]]))
np.savetxt(path + 'samsung/noisy/samsung_lr_misclassified.csv', np.squeeze(Test[misclassified,:]), fmt='%.4e', delimiter=',')

# Yelp (save mis-classified samples)
Train = np.genfromtxt(path + 'yelp/original/train/yelp_train.csv', dtype = float, delimiter=",")
Test = np.genfromtxt(path + 'yelp/original/test/yelp_test.csv', dtype = float, delimiter=",")
clf_logistic.fit(Train[:,1:Train.shape[1]], Train[:,0])
mean_accuracy = clf_logistic.score(Test[:,1:Train.shape[1]], Test[:,0])
print("The mean accuracy value for LR: yelp dataset is: ", mean_accuracy)
# misclassified samples in the dataset
misclassified = np.where(Test[:,0] != clf_logistic.predict(Test[:,1:Train.shape[1]]))
np.savetxt(path + 'yelp/noisy/yelp_lr_misclassified.csv', np.squeeze(Test[misclassified,:]), fmt='%.4e', delimiter=',')

# Census (save mis-classified samples)
Train = np.genfromtxt(path + 'census/original/train/census_train.csv', dtype = float, delimiter=",")
Test = np.genfromtxt(path + 'census/original/test/census_test.csv', dtype = float, delimiter=",")
clf_logistic.fit(Train[:,0:Train.shape[1]-1], Train[:,Train.shape[1]-1])
mean_accuracy = clf_logistic.score(Test[:,0:Train.shape[1]-1], Test[:,Train.shape[1]-1])
print("The mean accuracy value for LR: census dataset is: ", mean_accuracy)
# misclassified samples in the dataset
misclassified = np.where(Test[:,Train.shape[1]-1] != clf_logistic.predict(Test[:,0:Train.shape[1]-1]))
np.savetxt(path + 'census/noisy/census_lr_misclassified.csv', np.squeeze(Test[misclassified,:]), fmt='%.4e', delimiter=',')

# Forest Cover (save mis-classified samples)
Train = np.genfromtxt(path + 'covertype/original/train/covertype_train.csv', dtype = float, delimiter=",")
Test = np.genfromtxt(path + 'covertype/original/test/covertype_test.csv', dtype = float, delimiter=",")
clf_logistic.fit(Train[:,0:Train.shape[1]-1], Train[:,Train.shape[1]-1])
mean_accuracy = clf_logistic.score(Test[:,0:Train.shape[1]-1], Test[:,Train.shape[1]-1])
print("The mean accuracy value for LR: covertype dataset is: ", mean_accuracy)
# misclassified samples in the dataset
misclassified = np.where(Test[:,Train.shape[1]-1] != clf_logistic.predict(Test[:,0:Train.shape[1]-1]))
np.savetxt(path + 'covertype/noisy/covertype_lr_misclassified.csv', np.squeeze(Test[misclassified,:]), fmt='%.4e', delimiter=',')

# Letter Recognition (save mis-classified samples)
Train = np.genfromtxt(path + 'letter/original/train/letter_train.csv', dtype = float, delimiter=",")
Test = np.genfromtxt(path + 'letter/original/test/letter_test.csv', dtype = float, delimiter=",")
clf_logistic.fit(Train[:,1:Train.shape[1]], Train[:,0])
mean_accuracy = clf_logistic.score(Test[:,1:Train.shape[1]], Test[:,0])
print("The mean accuracy value for LR: letter dataset is: ", mean_accuracy)
# misclassified samples in the dataset
misclassified = np.where(Test[:,0] != clf_logistic.predict(Test[:,1:Train.shape[1]]))
np.savetxt(path + 'letter/noisy/letter_lr_misclassified.csv', np.squeeze(Test[misclassified,:]), fmt='%.4e', delimiter=',')

clf_rf = RandomForestClassifier(max_depth=20, random_state=0, n_estimators=50)
# Accuracy values using random forest (save mis-classified samples)
# Samsung (save mis-classified samples)
Train = np.genfromtxt(path + 'samsung/original/train/samsung_train.csv', dtype = float, delimiter=",")
Test = np.genfromtxt(path + 'samsung/original/test/samsung_test.csv', dtype = float, delimiter=",")
clf_rf.fit(Train[:,1:Train.shape[1]], Train[:,0])
mean_accuracy = clf_rf.score(Test[:,1:Train.shape[1]], Test[:,0])
print("The mean accuracy value for RF: samsung dataset is: ", mean_accuracy)
# misclassified samples in the dataset
misclassified = np.where(Test[:,0] != clf_rf.predict(Test[:,1:Train.shape[1]]))
np.savetxt(path + 'samsung/noisy/samsung_rf_misclassified.csv', np.squeeze(Test[misclassified,:]), fmt='%.4e', delimiter=',')

# Yelp (save mis-classified samples)
Train = np.genfromtxt(path + 'yelp/original/train/yelp_train.csv', dtype = float, delimiter=",")
Test = np.genfromtxt(path + 'yelp/original/test/yelp_test.csv', dtype = float, delimiter=",")
clf_rf.fit(Train[:,1:Train.shape[1]], Train[:,0])
mean_accuracy = clf_rf.score(Test[:,1:Train.shape[1]], Test[:,0])
print("The mean accuracy value for RF: yelp dataset is: ", mean_accuracy)
# misclassified samples in the dataset
misclassified = np.where(Test[:,0] != clf_rf.predict(Test[:,1:Train.shape[1]]))
np.savetxt(path + 'yelp/noisy/yelp_rf_misclassified.csv', np.squeeze(Test[misclassified,:]), fmt='%.4e', delimiter=',')

# Census (save mis-classified samples)
Train = np.genfromtxt(path + 'census/original/train/census_train.csv', dtype = float, delimiter=",")
Test = np.genfromtxt(path + 'census/original/test/census_test.csv', dtype = float, delimiter=",")
clf_rf.fit(Train[:,0:Train.shape[1]-1], Train[:,Train.shape[1]-1])
mean_accuracy = clf_rf.score(Test[:,0:Train.shape[1]-1], Test[:,Train.shape[1]-1])
print("The mean accuracy value for RF: census dataset is: ", mean_accuracy)
# misclassified samples in the dataset
misclassified = np.where(Test[:,Train.shape[1]-1] != clf_rf.predict(Test[:,0:Train.shape[1]-1]))
np.savetxt(path + 'census/noisy/census_rf_misclassified.csv', np.squeeze(Test[misclassified,:]), fmt='%.4e', delimiter=',')

# Forest Cover (save mis-classified samples)
Train = np.genfromtxt(path + 'covertype/original/train/covertype_train.csv', dtype = float, delimiter=",")
Test = np.genfromtxt(path + 'covertype/original/test/covertype_test.csv', dtype = float, delimiter=",")
clf_rf.fit(Train[:,0:Train.shape[1]-1], Train[:,Train.shape[1]-1])
mean_accuracy = clf_rf.score(Test[:,0:Train.shape[1]-1], Test[:,Train.shape[1]-1])
print("The mean accuracy value for RF: covertype dataset is: ", mean_accuracy)
# misclassified samples in the dataset
misclassified = np.where(Test[:,Train.shape[1]-1] != clf_rf.predict(Test[:,0:Train.shape[1]-1]))
np.savetxt(path + 'covertype/noisy/covertype_rf_misclassified.csv', np.squeeze(Test[misclassified,:]), fmt='%.4e', delimiter=',')

# Letter Recognition (save mis-classified samples)
Train = np.genfromtxt(path + 'letter/original/train/letter_train.csv', dtype = float, delimiter=",")
Test = np.genfromtxt(path + 'letter/original/test/letter_test.csv', dtype = float, delimiter=",")
clf_rf.fit(Train[:,1:Train.shape[1]], Train[:,0])
mean_accuracy = clf_rf.score(Test[:,1:Train.shape[1]], Test[:,0])
print("The mean accuracy value for RF: letter dataset is: ", mean_accuracy)
# misclassified samples in the dataset
misclassified = np.where(Test[:,0] != clf_rf.predict(Test[:,1:Train.shape[1]]))
np.savetxt(path + 'letter/noisy/letter_rf_misclassified.csv', np.squeeze(Test[misclassified,:]), fmt='%.4e', delimiter=',')

