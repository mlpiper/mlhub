import numpy as np


# Samsung data alert
X_samsung = np.genfromtxt('/Users/sindhu/Desktop/poc/Wix/Wix_logistic_test_batch1_featurenames.csv',dtype = float, delimiter=",",skip_header=1)

X_samsung[:,1] = X_samsung[:,1] + (15*np.random.rand(X_samsung.shape[0],))
X_header_samsung = np.genfromtxt('/Users/sindhu/Desktop/poc/Wix/Wix_logistic_test_batch1_featurenames.csv',dtype = str, delimiter=",")
header_string= ' '
for a in range(0,X_header_samsung.shape[1]):
    header_string = header_string +X_header_samsung[0][a]+ ',' 

#/Users/sindhu/Desktop/ClassificationDataset/CategoricalData
np.savetxt('/Users/sindhu/Desktop/ClassificationDataset/CategoricalData/Wix_logistic_test_batch1_featurenames_alert.csv', X_samsung, fmt='%.18e', delimiter=',', header=header_string)


# Yelp data alert
X_yelp = np.genfromtxt('/Users/sindhu/Desktop/poc/Wix/Wix_rf_test_batch1_featurenames.csv',dtype = float, delimiter=",",skip_header=1)
X_yelp[:,0] = 1
X_header_yelp = np.genfromtxt('/Users/sindhu/Desktop/poc/Wix/Wix_rf_test_batch1_featurenames.csv',dtype = str, delimiter=",")
header_string= ' '
for a in range(0,X_header_yelp.shape[1]):
    header_string = header_string +X_header_yelp[0][a]+ ',' 


#/Users/sindhu/Desktop/ClassificationDataset/CategoricalData
np.savetxt('/Users/sindhu/Desktop/ClassificationDataset/CategoricalData/Wix_rf_test_batch1_featurenames_alert.csv', X_yelp, fmt='%.18e', delimiter=',', header=header_string)


# Online news dataset
X_online_news = np.genfromtxt('/Users/sindhu/Downloads/OnlineNewsPopularity/OnlineNewsPopularity.csv',dtype = float, delimiter=",",skip_header=1)

X_header_online = np.genfromtxt('/Users/sindhu/Downloads/OnlineNewsPopularity/OnlineNewsPopularity.csv',dtype = str, delimiter=",")
header_string= ' '
for a in range(0,X_online_news.shape[1]):
    header_string = header_string +X_header_online[0][a]+ ',' 



# Only categorical alert
# data_channel_is_entertainment is modified
X_online_news_categorical = X_online_news
X_online_news_categorical[:,14] = 1
np.savetxt('/Users/sindhu/Desktop/ClassificationDataset/CategoricalData/OnlineNewsPopularity_categorical_alert.csv', X_online_news_categorical, fmt='%.18e', delimiter=',', header=header_string)


# Only continous alert
# LDA_00 is modified
X_online_news = np.genfromtxt('/Users/sindhu/Downloads/OnlineNewsPopularity/OnlineNewsPopularity.csv',dtype = float, delimiter=",",skip_header=1)
X_online_news_continuous = X_online_news
X_online_news_continuous[:,39] = X_online_news_continuous[:,39] + 50*np.random.rand(X_online_news_continuous.shape[0],)
np.savetxt('/Users/sindhu/Desktop/ClassificationDataset/CategoricalData/OnlineNewsPopularity_continuous_alert.csv', X_online_news_continuous, fmt='%.18e', delimiter=',', header=header_string)

# Both alerts
X_online_news[:,14] = 1
X_online_news[:,39] = X_online_news[:,39] + 50*np.random.rand(X_online_news.shape[0],)
np.savetxt('/Users/sindhu/Desktop/ClassificationDataset/CategoricalData/OnlineNewsPopularity_both_alert.csv', X_online_news, fmt='%.18e', delimiter=',', header=header_string)


# Pure visualization (Samsung + categorical)
X_samsung_categorical = np.zeros((X_samsung.shape[0],X_samsung.shape[1]+1))
X_samsung = np.genfromtxt('/Users/sindhu/Desktop/poc/Wix/Wix_logistic_test_batch1_featurenames.csv',dtype = float, delimiter=",",skip_header=1)
X_header_samsung = np.genfromtxt('/Users/sindhu/Desktop/poc/Wix/Wix_logistic_test_batch1_featurenames.csv',dtype = str, delimiter=",")
header_string= ' '
for a in range(0,X_header_samsung.shape[1]):
    header_string = header_string +X_header_samsung[0][a]+ ','
    
X_samsung_categorical[:,0:X_samsung.shape[1]] = X_samsung
X_samsung_categorical[1:10,X_samsung.shape[1]-1] = 1
np.savetxt('/Users/sindhu/Desktop/ClassificationDataset/CategoricalData/visualization_samsung_categorical.csv', X_samsung_categorical, fmt='%.18e', delimiter=',', header=header_string)



# Pure visualization (Yelp + continuous)
X_yelp_continous = np.zeros((X_yelp.shape[0],X_yelp.shape[1]+1))
X_yelp = np.genfromtxt('/Users/sindhu/Desktop/poc/Wix/Wix_rf_test_batch1_featurenames.csv',dtype = float, delimiter=",",skip_header=1)
X_header_yelp = np.genfromtxt('/Users/sindhu/Desktop/poc/Wix/Wix_rf_test_batch1_featurenames.csv',dtype = str, delimiter=",")
header_string= ' '
for a in range(0,X_header_yelp.shape[1]):
    header_string = header_string +X_header_yelp[0][a]+ ',' 

X_yelp_continous[:,0:X_yelp.shape[1]] = X_yelp
X_yelp_continous[:,X_yelp_continous.shape[1]-1] = np.random.rand(X_yelp.shape[0],)

np.savetxt('/Users/sindhu/Desktop/ClassificationDataset/CategoricalData/visualization_yelp_continuous.csv', X_yelp_continous, fmt='%.18e', delimiter=',', header=header_string)



