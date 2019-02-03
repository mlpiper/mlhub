"""
Generate the epsilon values for trained and tested on unmodified data (5 datasets on data-lake).
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

path = "/data-lake/ml-prototypes/regression/ml/"

clf_rf = RandomForestRegressor(max_depth=20, random_state=0, n_estimators=50)
clf_gbr = GradientBoostingRegressor()

plot_graph = False


class EpsilonDecider:

    def __init__(self):
        pass

    def plot_1D_bar_graph(self,
                          y_array,
                          title):
        if plot_graph:
            print("# --- Plotting Graph --- #")
            plt.hist(y_array, bins=50, facecolor='blue')

            plt.title(title)

            plt.show()
            print("# --- Plotting Graph Is Done--- #")
        else:
            print("### NOT PLOTTING GRAPH ###")

    def plot_1D_algo_graph(self,
                           algo="",
                           data_set=None,
                           null_model_acc=None,
                           algorithm_acc=None,
                           epsilon=None,
                           first_diff=None,
                           second_diff=None):
        if plot_graph:
            print("# --- Plotting Graph --- #")
            legends = []
            if algorithm_acc is not None:
                plt.plot(epsilon, algorithm_acc)
                legends.append(algo)

            if null_model_acc is not None:
                plt.plot(epsilon, null_model_acc)
                legends.append("Null Model")

            if first_diff is not None:
                plt.plot(epsilon, first_diff)
                legends.append("First Derivative With Epsilon")

            if second_diff is not None:
                plt.plot(epsilon, second_diff)
                legends.append("Second Derivative With Epsilon")

            plt.xlabel('epsilon')
            plt.ylabel('accuracy')

            for each_eps in epsilon:
                plt.axvline(x=each_eps, linestyle='dashed')

            plt.title(str(data_set) + " - " + str(algo))
            plt.legend(legends)

            plt.show()
            print("# --- Plotting Graph Is Done --- #")
        else:
            print("### NOT PLOTTING GRAPH ###")

    def plot_1D_all_detail_graph(self,
                                 data_set="",
                                 epsilons=None,
                                 rf_accuracy=None,
                                 gbr_accuracy=None,
                                 null_accuracy=None,
                                 first_diff_rf=None,
                                 first_diff_gbr=None,
                                 second_diff_rf=None,
                                 second_diff_gbr=None):
        if plot_graph:

            print("# --- Plotting Graph --- #")
            legends = []
            if rf_accuracy is not None:
                plt.plot(epsilons, rf_accuracy)
                legends.append("RF")

            if gbr_accuracy is not None:
                plt.plot(epsilons, gbr_accuracy)
                legends.append("GBR")

            if null_accuracy is not None:
                plt.plot(epsilons, null_accuracy)
                legends.append("Null Model")

            if first_diff_rf is not None:
                plt.plot(epsilons, first_diff_rf)
                legends.append("First Derivative RF With Epsilon")

            if first_diff_gbr is not None:
                plt.plot(epsilons, first_diff_gbr)
                legends.append("First Derivative GBR With Epsilon")

            if second_diff_rf is not None:
                plt.plot(epsilons, second_diff_rf)
                legends.append("Second Derivative RF With Epsilon")

            if second_diff_gbr is not None:
                plt.plot(epsilons, second_diff_gbr)
                legends.append("Second Derivative GBR With Epsilon")

            plt.xlabel('epsilon')
            plt.ylabel('accuracy')

            for each_eps in epsilons:
                plt.axvline(x=each_eps, linestyle='dashed')

            plt.title(data_set)
            plt.legend(legends)

            plt.show()
            print("# --- Plotting Graph Is Done --- #")

            self.plot_1D_algo_graph(algo="RF",
                                    data_set=data_set,
                                    null_model_acc=null_accuracy,
                                    algorithm_acc=rf_accuracy,
                                    epsilon=epsilons,
                                    first_diff=first_diff_rf,
                                    second_diff=second_diff_rf)

            self.plot_1D_algo_graph(algo="GBR",
                                    data_set=data_set,
                                    null_model_acc=null_accuracy,
                                    algorithm_acc=gbr_accuracy,
                                    epsilon=epsilons,
                                    first_diff=first_diff_gbr,
                                    second_diff=second_diff_gbr)
        else:
            print("### NOT PLOTTING GRAPH ###")

    @staticmethod
    def __get_threshold_epsilon__(epsilons_array=None,
                                  algo_accuracy=None,
                                  second_diff=None):

        threshold = -1.0

        for each_index_algo_accuracy in range(len(algo_accuracy) - 2):
            index_to_consider = each_index_algo_accuracy + 1
            if algo_accuracy[index_to_consider] >= 50 and \
                    second_diff[index_to_consider] <= second_diff[index_to_consider - 1] and \
                    second_diff[index_to_consider] <= second_diff[index_to_consider + 1]:
                threshold = epsilons_array[index_to_consider]
                break

        if threshold == -1.0:
            for each_index_algo_accuracy in range(len(algo_accuracy)):
                if algo_accuracy[each_index_algo_accuracy] >= 50:
                    threshold = epsilons_array[each_index_algo_accuracy]
                    break
        return threshold

    @staticmethod
    def __generate_first_order_second_order_derivative__(epsilons,
                                                         rf_accuracy,
                                                         gbr_accuracy,
                                                         null_accuracy):
        size = len(epsilons)

        print("epsilons = ", epsilons)
        print("rf_accuracy = ", rf_accuracy)
        print("gbr_accuracy = ", gbr_accuracy)
        print("null_accuracy = ", null_accuracy)

        first_diff_RF = []
        first_diff_GBR = []
        first_diff_Null = []

        for each_index in range(size - 1):

            diff_rf = rf_accuracy[each_index + 1] - rf_accuracy[each_index]
            if diff_rf <= 1e-15:
                diff_rf = 1
            first_diff_RF.append(diff_rf)

            diff_gbr = gbr_accuracy[each_index + 1] - gbr_accuracy[each_index]
            if diff_gbr <= 1e-15:
                diff_gbr = 1
            first_diff_GBR.append(diff_gbr)

            diff_null = null_accuracy[each_index + 1] - null_accuracy[each_index]
            if diff_null <= 1e-15:
                diff_null = 1
            first_diff_Null.append(diff_null)

        first_diff_RF.insert(0, 0)

        first_diff_GBR.insert(0, 0)

        first_diff_Null.insert(0, 0)

        print("first_diff_RF =", first_diff_RF)
        print("first_diff_GBR =", first_diff_GBR)
        print("first_diff_Null =", first_diff_Null)

        second_diff_RF = []
        second_diff_GBR = []
        second_diff_Null = []

        for each_index in range(size - 2):
            diff_rf = first_diff_RF[each_index + 1] - first_diff_RF[each_index]

            second_diff_RF.append(diff_rf)

            diff_gbr = first_diff_GBR[each_index + 1] - first_diff_GBR[each_index]

            second_diff_GBR.append(diff_gbr)

            diff_null = first_diff_Null[each_index + 1] - first_diff_Null[each_index]

            second_diff_Null.append(diff_null)

        second_diff_RF.insert(0, 0)
        second_diff_RF.insert(0, 0)

        second_diff_GBR.insert(0, 0)
        second_diff_GBR.insert(0, 0)

        second_diff_Null.insert(0, 0)
        second_diff_Null.insert(0, 0)

        print("second_diff_RF =", second_diff_RF)
        print("second_diff_GBR =", second_diff_GBR)
        print("second_diff_Null =", second_diff_Null)

        return first_diff_RF, first_diff_GBR, second_diff_RF, second_diff_GBR

    @staticmethod
    def __regression_to_classification_label__(epsilon,
                                               actual_label,
                                               predicted_label):
        prediction_classification_label = []

        absolute_error = np.abs(actual_label - predicted_label)

        for each_absolute_error in absolute_error:
            if each_absolute_error > epsilon:
                prediction_classification_label.append(0.0)
            else:
                prediction_classification_label.append(1.0)

        return prediction_classification_label

    @staticmethod
    def __binary_label_accuracy__(binary_labels):
        sum = np.sum(binary_labels)
        total_elements = len(binary_labels)

        return sum * 100.0 / total_elements

    def __calculate_eps_rf_gbr_null_accuracy__(self,
                                               epsilon,
                                               points_to_consider_for_exp,
                                               test_actual_labels,
                                               test_predict_labels_rf,
                                               test_predict_labels_gbr,
                                               test_predict_labels_null,
                                               data_set):
        print("# --- epsilon is: ", epsilon, " # of points: ", points_to_consider_for_exp, " --- #")

        epsilon_step = epsilon / points_to_consider_for_exp

        epsilons_array = []
        binary_rf_accuracy_for_associated_epsilon_array = []
        binary_gbr_accuracy_for_associated_epsilon_array = []
        binary_null_accuracy_for_associated_epsilon_array = []

        for each_step in range(points_to_consider_for_exp + 1):
            referenced_epsilon = each_step * epsilon_step

            test_predict_rf_binary_labels = \
                self.__regression_to_classification_label__(epsilon=referenced_epsilon,
                                                            actual_label=test_actual_labels,
                                                            predicted_label=test_predict_labels_rf)

            rf_binary_labels_accuracy = self.__binary_label_accuracy__(binary_labels=test_predict_rf_binary_labels)

            test_predict_gbr_binary_labels = \
                self.__regression_to_classification_label__(epsilon=referenced_epsilon,
                                                            actual_label=test_actual_labels,

                                                            predicted_label=test_predict_labels_gbr)

            gbr_binary_labels_accuracy = self.__binary_label_accuracy__(binary_labels=test_predict_gbr_binary_labels)

            test_predict_null_binary_labels = \
                self.__regression_to_classification_label__(epsilon=referenced_epsilon,
                                                            actual_label=test_actual_labels,

                                                            predicted_label=test_predict_labels_null)

            null_binary_labels_accuracy = self.__binary_label_accuracy__(binary_labels=test_predict_null_binary_labels)

            epsilons_array.append(referenced_epsilon)
            binary_rf_accuracy_for_associated_epsilon_array.append(rf_binary_labels_accuracy)
            binary_gbr_accuracy_for_associated_epsilon_array.append(gbr_binary_labels_accuracy)
            binary_null_accuracy_for_associated_epsilon_array.append(null_binary_labels_accuracy)

        return \
            epsilons_array, \
            binary_rf_accuracy_for_associated_epsilon_array, \
            binary_gbr_accuracy_for_associated_epsilon_array, \
            binary_null_accuracy_for_associated_epsilon_array

    @staticmethod
    def __generate_rf_and_gbr_and_null_labels__(train_features_set,
                                                train_labels,
                                                test_features_set,
                                                test_labels,
                                                null_pred_to_be):
        clf_rf.fit(train_features_set, train_labels)
        clf_gbr.fit(train_features_set, train_labels)

        # print("\"\"\"\n --- RF Estimators ", clf_rf.estimators_, " --- \"\"\"\n")
        #
        # print("\"\"\"\n  --- GBR Estimators ", clf_gbr.estimators_, " --- \"\"\"\n")

        test_predict_labels_rf = clf_rf.predict(test_features_set)
        test_predict_labels_gbr = clf_gbr.predict(test_features_set)
        test_predict_labels_null = np.full(test_labels.shape, null_pred_to_be)

        mse_rf = mean_squared_error(test_labels, test_predict_labels_rf)
        mse_gbr = mean_squared_error(test_labels, test_predict_labels_gbr)
        mse_null = mean_squared_error(test_labels, test_predict_labels_null)

        print("# --- the sqrt mean square value for rf regression: ", np.sqrt(mse_rf), " --- #")
        print("# --- the sqrt mean square value for gb regression: ", np.sqrt(mse_gbr), " --- #")
        print("# --- the sqrt mean square value for null regression: ", np.sqrt(mse_null), " --- #")

        return test_predict_labels_rf, test_predict_labels_gbr, test_predict_labels_null

    @staticmethod
    def __get_epsilon_of_test_and_avg_of_train__(train_labels,
                                                 test_labels):
        avg_of_train_labels = np.mean(train_labels)

        avg_of_test_labels = np.mean(test_labels)

        mean_labels = np.full(test_labels.shape, avg_of_test_labels)

        epsilon_square = mean_squared_error(test_labels, mean_labels)

        epsilon = np.sqrt(epsilon_square)

        return epsilon, avg_of_train_labels

    def __generate_rf_gbr_thresholds__(self,
                                       points_to_consider_for_exp,
                                       train_features_set,
                                       train_labels,
                                       test_features_set,
                                       test_labels,
                                       data_set):
        print("# Prediction Range Actual= [", np.min(test_labels), np.max(test_labels), "] #")

        # generating epsilon_of_test_and_avg_of_train
        epsilon_for_test, avg_of_train_labels = \
            self.__get_epsilon_of_test_and_avg_of_train__(train_labels=train_labels,
                                                          test_labels=test_labels)

        # generating rf gbr and null predictions (regression)
        test_predict_labels_rf, test_predict_labels_gbr, test_predict_labels_null = \
            self.__generate_rf_and_gbr_and_null_labels__(
                train_features_set=train_features_set,
                train_labels=train_labels,
                test_features_set=test_features_set,
                test_labels=test_labels,
                null_pred_to_be=avg_of_train_labels
            )

        mad_rf_labels = np.abs(test_labels - test_predict_labels_rf)
        mad_gbr_labels = np.abs(test_labels - test_predict_labels_gbr)
        mad_null_labels = np.abs(test_labels - test_predict_labels_null)
        mad_rf_null_labels = test_predict_labels_rf - test_predict_labels_null
        mad_gbr_null_labels = test_predict_labels_gbr - test_predict_labels_null

        print("# Prediction Range RF= [", np.min(mad_rf_labels), np.max(mad_rf_labels), "] #")
        print("# Prediction Range GBR= [", np.min(mad_gbr_labels), np.max(mad_gbr_labels), "] #")
        print("# Prediction Range Null= [", np.min(mad_null_labels), np.max(mad_null_labels), "] #")
        print("# Prediction Range RF-Null= [", np.min(mad_rf_null_labels), np.max(mad_rf_null_labels), "] #")
        print("# Prediction Range GBR-Null= [", np.min(mad_gbr_null_labels), np.max(mad_gbr_null_labels), "] #")

        # generating rf gbr and null accuracy array for various epsilon steps
        epsilons_array, \
        binary_rf_accuracy_for_associated_epsilon_array, \
        binary_gbr_accuracy_for_associated_epsilon_array, \
        binary_null_accuracy_for_associated_epsilon_array = \
            self.__calculate_eps_rf_gbr_null_accuracy__(epsilon=epsilon_for_test,
                                                        points_to_consider_for_exp=points_to_consider_for_exp,
                                                        test_actual_labels=test_labels,
                                                        test_predict_labels_rf=test_predict_labels_rf,
                                                        test_predict_labels_gbr=test_predict_labels_gbr,
                                                        test_predict_labels_null=test_predict_labels_null,
                                                        data_set=data_set)

        # generating first ordered and second ordered rf gbr and null accuracy array for various epsilon steps
        first_diff_RF, first_diff_GBR, second_diff_RF, second_diff_GBR = \
            self.__generate_first_order_second_order_derivative__(
                epsilons=epsilons_array,
                rf_accuracy=binary_rf_accuracy_for_associated_epsilon_array,
                gbr_accuracy=binary_gbr_accuracy_for_associated_epsilon_array,
                null_accuracy=binary_null_accuracy_for_associated_epsilon_array
            )

        rf_threshold = self.__get_threshold_epsilon__(epsilons_array=epsilons_array,
                                                      algo_accuracy=binary_rf_accuracy_for_associated_epsilon_array,
                                                      second_diff=second_diff_RF)

        gbr_threshold = self.__get_threshold_epsilon__(epsilons_array=epsilons_array,
                                                       algo_accuracy=binary_gbr_accuracy_for_associated_epsilon_array,
                                                       second_diff=second_diff_GBR)

        # plotting all details!! yepppy
        self.plot_1D_all_detail_graph(data_set=data_set,
                                      epsilons=epsilons_array,
                                      rf_accuracy=binary_rf_accuracy_for_associated_epsilon_array,
                                      gbr_accuracy=binary_gbr_accuracy_for_associated_epsilon_array,
                                      null_accuracy=binary_null_accuracy_for_associated_epsilon_array,
                                      first_diff_rf=first_diff_RF,
                                      first_diff_gbr=first_diff_GBR,
                                      second_diff_rf=second_diff_RF,
                                      second_diff_gbr=second_diff_GBR)

        print("RF_Threshold =  ", rf_threshold)

        print("GBR_Threshold = ", gbr_threshold)

        return rf_threshold, gbr_threshold

    def fit_transform_and_get_threshold(self,
                                        points_to_consider_for_exp=20,
                                        train_features_set=None,
                                        train_labels=None,
                                        test_features_set=None,
                                        test_labels=None,
                                        data_set_name=None):

        return self.__generate_rf_gbr_thresholds__(points_to_consider_for_exp=points_to_consider_for_exp,
                                                   train_features_set=train_features_set,
                                                   train_labels=train_labels,
                                                   test_features_set=test_features_set,
                                                   test_labels=test_labels,
                                                   data_set=str(data_set_name))


def main(run_songs_flag=False,
         run_blog_flag=False,
         run_facebook_flag=False,
         run_turbine_flag=False,
         run_videos_flag=False,
         points_to_consider_for_exp=20):
    # MSE values using gbr regression and random forest

    if run_songs_flag:
        # Songs (first value is the target value)
        print("# -------- Starting Songs's Experiment -------- #")

        train_songs = np.genfromtxt(path + 'songs/original/train/songs_train_subset.csv', dtype=float, delimiter=",")
        test_songs = np.genfromtxt(path + 'songs/original/test/songs_test.csv', dtype=float, delimiter=",")

        num_samples_train_songs, num_features_songs = train_songs.shape
        print("# --- shape of training samples: ", train_songs.shape, " --- #")
        num_samples_test_songs, num_features_songs = test_songs.shape
        print("# --- shape of test samples: ", test_songs.shape, " --- #")

        train_features_set_songs = train_songs[:, 1:num_features_songs]
        train_labels_songs = train_songs[:, 0]

        test_features_set_songs = test_songs[:, 1:num_features_songs]
        test_labels_songs = test_songs[:, 0]

        rf_threshold, gbr_threshold = \
            EpsilonDecider().fit_transform_and_get_threshold(
                points_to_consider_for_exp=points_to_consider_for_exp,
                train_features_set=train_features_set_songs,
                train_labels=train_labels_songs,
                test_features_set=test_features_set_songs,
                test_labels=test_labels_songs,
                data_set_name="Songs")

        print("# -------- Ending Songs's Experiment -------- #")

    if run_blog_flag:
        # Blog (last value is the target value)
        print("# -------- Starting Blog's Experiment -------- #")

        train_blog = np.genfromtxt(path + 'blog/original/train/blog_train.csv', dtype=float, delimiter=",")
        test_blog = np.genfromtxt(path + 'blog/original/test/blog_test.csv', dtype=float, delimiter=",")

        num_samples_train_blog, num_features_blog = train_blog.shape
        print("# --- shape of training samples: ", train_blog.shape, " --- #")
        num_samples_test_blog, num_features_blog = test_blog.shape
        print("# --- shape of test samples: ", test_blog.shape, " --- #")

        train_features_set_blog = train_blog[:, 0:num_features_blog - 1]
        train_labels_blog = train_blog[:, num_features_blog - 1]

        test_features_set_blog = test_blog[:, 0:num_features_blog - 1]
        test_labels_blog = test_blog[:, num_features_blog - 1]

        rf_threshold, gbr_threshold = \
            EpsilonDecider().fit_transform_and_get_threshold(
                points_to_consider_for_exp=points_to_consider_for_exp,
                train_features_set=train_features_set_blog,
                train_labels=train_labels_blog,
                test_features_set=test_features_set_blog,
                test_labels=test_labels_blog,
                data_set_name="Blog")

        print("# -------- Ending Blog's Experiment -------- #")

    if run_facebook_flag:
        # Facebook (last value is the target value)
        print("# -------- Starting Facebook's Experiment -------- #")

        train_facebook = np.genfromtxt(path + 'facebook/original/train/facebook_train.csv', dtype=float, delimiter=",")
        test_facebook = np.genfromtxt(path + 'facebook/original/test/facebook_test.csv', dtype=float, delimiter=",")

        num_samples_train_facebook, num_features_facebook = train_facebook.shape
        print("# --- shape of training samples: ", train_facebook.shape, " --- #")
        num_samples_test_facebook, num_features_facebook = test_facebook.shape
        print("# --- shape of test samples: ", test_facebook.shape, " --- #")

        train_features_set_facebook = train_facebook[:, 0:num_features_facebook - 1]
        train_labels_facebook = train_facebook[:, num_features_facebook - 1]

        test_features_set_facebook = test_facebook[:, 0:num_features_facebook - 1]
        test_labels_facebook = test_facebook[:, num_features_facebook - 1]

        rf_threshold, gbr_threshold = \
            EpsilonDecider().fit_transform_and_get_threshold(
                points_to_consider_for_exp=points_to_consider_for_exp,
                train_features_set=train_features_set_facebook,
                train_labels=train_labels_facebook,
                test_features_set=test_features_set_facebook,
                test_labels=test_labels_facebook,
                data_set_name="Facebook")

        print("# -------- Ending Facebook's Experiment -------- #")

    if run_turbine_flag:
        # Turbine (last value is the target value)
        print("# -------- Starting Turbine's Experiment -------- #")

        train_turbine = np.genfromtxt(path + 'turbine/original/train/turbine_train.csv', dtype=float, delimiter=",")
        test_turbine = np.genfromtxt(path + 'turbine/original/test/turbine_test.csv', dtype=float, delimiter=",")

        num_samples_train_turbine, num_features_turbine = train_turbine.shape
        print("# --- shape of training samples: ", train_turbine.shape, " --- #")
        num_samples_test_turbine, num_features_turbine = test_turbine.shape
        print("# --- shape of test samples: ", test_turbine.shape, " --- #")

        train_features_set_turbine = train_turbine[:, 0:num_features_turbine - 1]
        train_labels_turbine = train_turbine[:, num_features_turbine - 1]

        test_features_set_turbine = test_turbine[:, 0:num_features_turbine - 1]
        test_labels_turbine = test_turbine[:, num_features_turbine - 1]

        rf_threshold, gbr_threshold = \
            EpsilonDecider().fit_transform_and_get_threshold(
                points_to_consider_for_exp=points_to_consider_for_exp,
                train_features_set=train_features_set_turbine,
                train_labels=train_labels_turbine,
                test_features_set=test_features_set_turbine,
                test_labels=test_labels_turbine,
                data_set_name="Turbine")

        print("# -------- Ending Turbine's Experiment -------- #")

    if run_videos_flag:
        # Video (last value is the target value)
        print("# -------- Starting Video's Experiment -------- #")

        train_videos = np.genfromtxt(path + 'videos/original/train/videos_train.csv', dtype=float, delimiter=",")
        test_videos = np.genfromtxt(path + 'videos/original/test/videos_test.csv', dtype=float, delimiter=",")

        num_samples_train_videos, num_features_videos = train_videos.shape
        print("# --- shape of training samples: ", train_videos.shape, " --- #")
        num_samples_test_videos, num_features_videos = test_videos.shape
        print("# --- shape of test samples: ", test_videos.shape, " --- #")

        train_features_set_videos = train_videos[:, 0:num_features_videos - 1]
        train_labels_videos = train_videos[:, num_features_videos - 1]

        test_features_set_videos = test_videos[:, 0:num_features_videos - 1]
        test_labels_videos = test_videos[:, num_features_videos - 1]

        rf_threshold, gbr_threshold = \
            EpsilonDecider().fit_transform_and_get_threshold(
                points_to_consider_for_exp=points_to_consider_for_exp,
                train_features_set=train_features_set_videos,
                train_labels=train_labels_videos,
                test_features_set=test_features_set_videos,
                test_labels=test_labels_videos,
                data_set_name="Video's")

        print("# -------- Ending Video's Experiment -------- #")


if __name__ == "__main__":
    print("# -------- Starting Experiment -------- #")

    main(
        run_songs_flag=False,
        run_blog_flag=False,
        run_facebook_flag=False,
        run_turbine_flag=False,
        run_videos_flag=False,
        points_to_consider_for_exp=20
    )

    print("# -------- Ending Experiment -------- #")

"""
example of how to use

# Turbine (last value is the target value)
print("# -------- Starting Turbine's Experiment -------- #")

train_turbine = np.genfromtxt(path + 'turbine/original/train/turbine_train.csv', dtype=float, delimiter=",")
test_turbine = np.genfromtxt(path + 'turbine/original/test/turbine_test.csv', dtype=float, delimiter=",")

num_samples_train_turbine, num_features_turbine = train_turbine.shape
print("# --- shape of training samples: ", train_turbine.shape, " --- #")
num_samples_test_turbine, num_features_turbine = test_turbine.shape
print("# --- shape of test samples: ", test_turbine.shape, " --- #")

train_features_set_turbine = train_turbine[:, 0:num_features_turbine - 1]
train_labels_turbine = train_turbine[:, num_features_turbine - 1]

test_features_set_turbine = test_turbine[:, 0:num_features_turbine - 1]
test_labels_turbine = test_turbine[:, num_features_turbine - 1]

rf_threshold, gbr_threshold = \
    EpsilonDecider().fit_transform_and_get_threshold(
        points_to_consider_for_exp=20,
        train_features_set=train_features_set_turbine,
        train_labels=train_labels_turbine,
        test_features_set=test_features_set_turbine,
        test_labels=test_labels_turbine,
        data_set_name="Turbine")

print("# -------- Ending Turbine's Experiment -------- #")

"""
