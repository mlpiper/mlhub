"""
- Convert the Yelp dataset into two parts, training and testing.
- Create a noisy test dataset
"""

import pandas as pd
import numpy as np
import sklearn.preprocessing as preprocessing


path = "/data-lake/ml-prototypes/classification/ml/census/"


def number_encode_features(df):
    result = df.copy()
    encoders = {}
    for column in result.columns:
        if result.dtypes[column] == np.object:
            encoders[column] = preprocessing.LabelEncoder()
            result[column] = encoders[column].fit_transform(result[column].fillna('0'))
    return result, encoders

###############################################################################
########## Read the dataset and save it in a numpy array ######################
###############################################################################

original_data_train = pd.read_csv(
    path + "original/adult.data",
    names=[
        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week", "Country", "Target"],
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")




data_df, _ = number_encode_features(original_data_train)

data = data_df.values

Train = data[0:np.int(2*len(data)/3),:]
Test = data[np.int(2*len(data)/3):len(data),:]

### Divide the dataset into two batches.
np.random.shuffle(Train)
np.random.shuffle(Test)



write_path = "/data-lake/ml-prototypes/classification/ml/census/"

# Remove nan and encode the label to match the training data
Test = np.nan_to_num(Test)
Train = np.nan_to_num(Train)
#Test[:,Train.shape[1]-1] = Test[:,Train.shape[1]-1]-1

np.savetxt(write_path + 'original/train/census_train.csv', Train, fmt='%.4e', delimiter=',')
np.savetxt(write_path + 'original/test/census_test.csv', Test, fmt='%.4e', delimiter=',')


###############################################################################
######### Add noise to the test dataset #######################################
###############################################################################
"""
age: continuous.
workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
fnlwgt: continuous.
education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
education-num: continuous.
marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
sex: Female, Male.
capital-gain: continuous.
capital-loss: continuous.
hours-per-week: continuous.
native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc),
India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam,
Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary,
Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.
"""

continuous = [1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0]
Test_noisy = Test
for a in range(0,Test.shape[1]-1):
    if(continuous[a]==0):
        # Add random binary noise when the variable is categorical
        Test_noisy[:,a] = np.random.randint(0,2,size=[Test.shape[0],]) + Test[:,a]
    else:
        # Add random continuous noise when the variable is continuous
        Test_noisy[:,a] = (np.max(Test[:,a]))*100*np.random.rand(Test.shape[0],) + Test[:,a]

np.savetxt(write_path + 'noisy/census_test_noisy.csv', Test_noisy, fmt='%.4e', delimiter=',')


