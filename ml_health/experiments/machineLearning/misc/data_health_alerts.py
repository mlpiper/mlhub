import numpy as np 
import matplotlib.pyplot as plt
#from multi_variate_statistics_create import multi_variate_statistics_create
#from multi_variate_statistics_inference import multi_variate_statistics_inference
from multi_variate_stats import multi_variate_statistics_create, multi_variate_statistics_inference 

k=11 # Number of attributes 
N=10000 # Number of samples


########################################################################################
########### Generating Dataset #########################################################
########### 2 multivariate Gaussian with mui vector and Signam matrix.##################
########################################################################################

mui1 = np.zeros((1,k))
mui1[0,:] = np.arange(1,k+1)
mui2 = np.zeros((1,k))
#mui2[0,:] = np.arange(k,0,-1)
mui2[0,:] = mui1
mui2[0,0] = 10

root_sigma1 = 0.3*np.eye(k) # root sigma. sigma = root_sigma.T @ root_sigma
root_sigma2 = 0.4*np.eye(k)

root_sigma1[0,:]=0.03 * np.arange(1,k+1)
root_sigma1[:,0]=0.03 * np.arange(1,k+1)

#root_sigma2[0,:]=0.04 * np.arange(k,0,-1)
#root_sigma2[:,0]=0.04 * np.arange(k,0,-1)
root_sigma2 = root_sigma1


eye_norm_sample = np.random.randn(N,k)

norm_sample1 = eye_norm_sample @ root_sigma1 + mui1

eye_norm_sample = np.random.randn(N,k)

norm_sample2 = eye_norm_sample @ root_sigma2 + mui2

print("Sythetic_mui1 = ", mui1)
print("Sythetic_mui2 = ", mui2)

print("Sythetic_sigma1 = ", root_sigma1.T @ root_sigma1)
print("Sythetic_sigma2 = ", root_sigma2.T @ root_sigma2)

########################################################################################


########### Calculating the multivariate statistics from  Dataset1 and 2 ###############
########################################################################################

mean_sample1, var_sample1, test_prob_sample1, max_prob_sample1, detect_test_prob_sample1  = multi_variate_statistics_create(norm_sample1)

mean_sample2, var_sample2, test_prob_sample2, max_prob_sample2, detect_test_prob_sample2  = multi_variate_statistics_create(norm_sample2)

plt.plot(test_prob_sample1, label = "dataset1")
print("max_prob_sample1 = ", max_prob_sample1)
print("detect_test_prob_sample1 = ",detect_test_prob_sample1)


plt.plot(test_prob_sample2,'r', label = "dataset2")
print("max_prob_sample2 = ",max_prob_sample2)
print("detect_test_prob_sample2 = ",detect_test_prob_sample2)
plt.title("Probability graphs, 1st dataset using dataset1 stats, 2nd dataset using dataset2 stats")
plt.legend(bbox_to_anchor=(1, 1),prop={'size':10})

threshold1 = 0.5 * detect_test_prob_sample1 
threshold2 = 0.5 * detect_test_prob_sample2
#threshold equals (2*mean training likelihood - maximum likelihood).

print("threshold1 = ", threshold1)
print("threshold2 = ", threshold2)


########### Calculating the multivariate statistics inference from  Dataset1 and 2 #####
########################################################################################

inf_test_prob_sample1, inf_detect_test_prob_sample1  = multi_variate_statistics_inference(norm_sample1, mean_sample1,var_sample1, max_prob_sample1)
inf_test_prob_sample2, inf_detect_test_prob_sample2  = multi_variate_statistics_inference(norm_sample2, mean_sample2,var_sample2, max_prob_sample2)

infCross_test_prob_sample1, infCross_detect_test_prob_sample1  = multi_variate_statistics_inference(norm_sample1, mean_sample2,var_sample2, max_prob_sample2)
infCross_test_prob_sample2, infCross_detect_test_prob_sample2  = multi_variate_statistics_inference(norm_sample2, mean_sample1,var_sample1, max_prob_sample1)



plt.figure()
plt.plot(infCross_test_prob_sample1, label = "dataset1")
plt.title("Probability graphs, 1st dataset using dataset2 stats")
plt.legend(bbox_to_anchor=(1, 1),prop={'size':10})

print("infCross_detect_test_prob_sample1 = ",infCross_detect_test_prob_sample1)

plt.figure()
plt.plot(infCross_test_prob_sample2,'r', label = "dataset2")
print("infCross_detect_test_prob_sample2 = ",infCross_detect_test_prob_sample2)
plt.title("Probability graphs, 2nd dataset using dataset1 stats")
plt.legend(bbox_to_anchor=(1, 1),prop={'size':10})

plt.show()

