"""
- Code was originally provided by LiorK.
- Code is responsible for reading datafiles and creating Training (30%), Validation(20%) and Testing(50%) datasets.
- Last column of dataset is labels.
"""
import numpy as np
import argparse
import os

def parse_args():
    """
    Parse Arguments from component
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--xsar-path", help="path containing the xsar folder", default="/data-lake/ml-prototypes/classification/ml/realm-im2015-vod-traces/X_SAR/")
    parser.add_argument("--output", help="output path", default="/data-lake/ml-prototypes/classification/ml/realm-im2015-vod-traces/X_SAR/")
    options = parser.parse_args()
    return options

options = parse_args()
data_path = options.xsar_path
op_path = options.output

# periodic_load is the path to the first Dataset in time
periodic_load_path = data_path + '/periodic_load/'
periodic_op_path = op_path + '/periodic_load/'

# flashcrowd_load is the path to the second Dataset in time
flashcrowd_load_path = data_path + '/flashcrowd_load/'
flashcrowd_op_path = op_path + '/flashcrowd_load/'

# linear_increase is the path to the second Dataset in time
linear_increase_path = data_path + '/linear_increase/'
linear_increase_op_path = op_path + '/linear_increase/'

# constant_load is the path to the second Dataset in time
constant_load_path = data_path + '/constant_load/'
constant_op_path = op_path + '/constant_load/'

# poisson_load is the path to the second Dataset in time
poisson_load_path = data_path + '/poisson_load/'
poisson_op_path = op_path + '/poisson_load/'

all_path = [periodic_load_path, flashcrowd_load_path, linear_increase_path, constant_load_path, poisson_load_path]
op_paths = [periodic_op_path, flashcrowd_op_path, linear_increase_op_path, constant_op_path, poisson_op_path]

def load_dataset(PATH):
    # X_names is the name of the features to parse from the file
    X_names = [b'all_%%usr', b'all_%%sys', b'all_%%iowait', b'all_%%idle', b'fault/s', b'%%memused',
               b'%%commit', b'%%swpused', b'%%swpcad', b'eth3_rxpck/s', b'eth3_txpck/s', b'eth3_rxkB/s',
               b'eth3_txkB/s', b'eth2_rxpck/s', b'eth2_txpck/s', b'eth2_rxkB/s', b'eth2_txkB/s',
               b'eth1_rxpck/s', b'eth1_txpck/s', b'eth1_rxkB/s', b'eth1_txkB/s', b'eth0_rxpck/s',
               b'eth0_txpck/s', b'eth0_rxkB/s', b'eth0_txkB/s', b'lo_rxpck/s', b'lo_txpck/s',
               b'lo_rxkB/s', b'lo_txkB/s', b'read/s', b'write/s', b'packet/s', b'sread/s', b'swrite/s',
               b'proc/s', b'cswch/s']

    # Load dataset
    print("--- Data Is Being Loaded ---")
    print("-- File: ", PATH)
    X1_header = np.genfromtxt(PATH + 'X.csv', delimiter=',', dtype=None)

    X1 = np.genfromtxt(PATH + 'X.csv', delimiter=',')

    print("--- Data Is Loaded ---")

    # Filter X to include the relevant columns only
    X_cols = np.zeros(len(X_names))  # Column vectors
    X_dev = np.zeros((X1.shape[0] - 1, len(X_names)))

    print("--- Selecting Specific Cols ---")

    for i in range(0, len(X_names)):
        X_cols[i] = np.int(np.where(X1_header[0, :] == X_names[i])[0][0])
        X_dev[:, i] = X1[1:np.int(X1.shape[0]), np.int(X_cols[i])]

    X_rec = np.zeros((X_dev.shape[0], X_dev.shape[1] - 12))
    X_rec[:, 0:8] = X_dev[:, 0:8]
    X_rec[:, 13:24] = X_dev[:, 25:36]
    X_rec[:, 9] = X_dev[:, 9] + X_dev[:, 13] + X_dev[:, 17] + X_dev[:, 21]
    X_rec[:, 10] = X_dev[:, 10] + X_dev[:, 14] + X_dev[:, 18] + X_dev[:, 22]
    X_rec[:, 11] = X_dev[:, 11] + X_dev[:, 15] + X_dev[:, 19] + X_dev[:, 23]
    X_rec[:, 12] = X_dev[:, 12] + X_dev[:, 16] + X_dev[:, 20] + X_dev[:, 24]

    # Handling the label. Load the label File
    # Label_class = 1 if FPS < 10 else 0
    Y1 = np.genfromtxt(PATH + 'Y.csv', delimiter=',')
    Y1 = Y1[1:Y1.shape[0], 12]  # FPS attribute from Y

    X_label = (Y1 < 20) * 2 - 1

    X_label = np.reshape(X_label, (-1, X_label.shape[0]))

    print("--- Normalizing Dataset ---")

    # Normalizing the dataset by dividing with Max(Abs(value))
    X_data = X_rec / (np.max(np.abs(X_rec), axis=0) + 1e-6)

    X_data_label = np.concatenate((X_data, X_label.T), axis=1)

    total_rows = np.arange(len(X_data_label))
    np.random.shuffle(total_rows)

    x_rows_shuffle = total_rows

    train_size = int(len(x_rows_shuffle) * 0.3)
    val_size = int(len(x_rows_shuffle) * 0.2)

    train_index = x_rows_shuffle[0:train_size]
    val_index = x_rows_shuffle[train_size:train_size + val_size]
    test_index = x_rows_shuffle[train_size + val_size:len(X_data_label)]

    X_train = X_data_label[train_index, :]
    X_val = X_data_label[val_index, :]
    X_test = X_data_label[test_index, :]

    print("Train Dataset Size: ", X_train.shape)
    print("Val Dataset Size:", X_val.shape)
    print("Test Dataset Size:", X_test.shape)

    return X_train, X_val, X_test


def save_new_dataset(data_train, data_val, data_test, path, output):
    print("-- Writing To File --")
    try:
        os.makedirs(output)
    except:
        pass

    train_path = output + "/Train.csv"
    np.savetxt(fname=train_path, X=data_train, delimiter=',')
    print("-- Writing To Train File Is Done --")

    validate_path = output + "/Validate.csv"
    np.savetxt(fname=validate_path, X=data_val, delimiter=',')
    print("-- Writing To Validate File Is Done --")

    test_path = output + "/Test.csv"
    np.savetxt(fname=test_path, X=data_test, delimiter=',')
    print("-- Writing To Test File Is Done --")

    print("-- Writing To File Is Done --")


def generate_dataset(all_paths, op_paths):
    index = 0
    for each_path in all_paths:
        xm_data_train, xm_data_val, xm_data_test = load_dataset(each_path)

        save_new_dataset(data_train=xm_data_train, data_val=xm_data_val, data_test=xm_data_test, path=each_path,
                output=op_paths[index])
        index = index + 1


print("All Path: ", all_path)
print("Output Path: ", op_paths)
generate_dataset(all_paths=all_path, op_paths=op_paths)
