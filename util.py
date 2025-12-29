import numpy as np
import pandas as pd
import scipy.io
import pickle
from sklearn.model_selection import train_test_split
from collections import Counter
from tqdm import tqdm


def preprocess_physionet(mode="sample"):
    """
    download the raw data from https://physionet.org/content/challenge-2017/1.0.0/, 
    and put it in ../data/challenge2017/

    The preprocessed dataset challenge2017.pkl can also be found at https://drive.google.com/drive/folders/1AuPxvGoyUbKcVaFmeyt3xsqj6ucWZezf
    """

    # read label
    if mode == "sample":
        label_df = pd.read_csv('./data/sample2017/answers.txt', sep=',', header=None)
        label = label_df.iloc[:, 1].values
        print(Counter(label))

        # read data
        all_data = []
        filenames = pd.read_csv('./data/sample2017/validation/RECORDS', header=None)
        filenames = filenames.iloc[:, 0].values
        print(filenames)
        for filename in tqdm(filenames):
            mat = scipy.io.loadmat('./data/sample2017/validation/{0}.mat'.format(filename))
            mat = np.array(mat['val'])[0]
            all_data.append(mat)
        all_data = np.array(all_data, dtype=object)

        res = {'data': all_data, 'label': label}
        with open('./data/sample_challenge2017.pkl', 'wb') as fout:
            pickle.dump(res, fout)
    elif mode == "full":
        label_df = pd.read_csv('./data/REFERENCE-v3.csv', header=None)
        label = label_df.iloc[:, 1].values
        print(Counter(label))

        # read data
        all_data = []
        filenames = pd.read_csv('./data/training2017/RECORDS', header=None)
        filenames = filenames.iloc[:, 0].values
        print(filenames)
        for filename in tqdm(filenames):
            mat = scipy.io.loadmat('./data/training2017/{0}.mat'.format(filename))
            mat = np.array(mat['val'])[0]
            all_data.append(mat)
        all_data = np.array(all_data, dtype=object)

        res = {'data': all_data, 'label': label}
        with open('./data/challenge2017.pkl', 'wb') as fout:
            pickle.dump(res, fout)
    else:
        print("PLEASE SELECT A MODE (sample or full)")


def slide_and_cut(X, Y, window_size, stride, output_pid=False, datatype=4):
    out_X = []
    out_Y = []
    out_pid = []
    n_sample = X.shape[0]
    # mode = 0
    for i in range(n_sample):
        tmp_ts = X[i]
        tmp_Y = Y[i]
        if tmp_Y == 0:
            i_stride = stride
        elif tmp_Y == 1:
            if datatype == 4:
                i_stride = stride//6
            elif datatype == 2:
                i_stride = stride//10
            elif datatype == 2.1:
                i_stride = stride//7
        elif tmp_Y == 2:
            i_stride = stride//2
        elif tmp_Y == 3:
            i_stride = stride//20
        for j in range(0, len(tmp_ts)-window_size, i_stride):
            out_X.append(tmp_ts[j:j+window_size])
            out_Y.append(tmp_Y)
            out_pid.append(i)
    if output_pid:
        return np.array(out_X), np.array(out_Y), np.array(out_pid)
    else:
        return np.array(out_X), np.array(out_Y)


def read_data_physionet_4(mode="sample", window_size=3000, stride=500):

    # read pkl
    if mode == "sample":
        with open('./data/sample_challenge2017.pkl', 'rb') as fin:
            res = pickle.load(fin)
    elif mode == "full":
        with open('./data/challenge2017.pkl', 'rb') as fin:
            res = pickle.load(fin)
    else:
        print("PLEASE SELECT A MODE (sample or full)")

    # scale data
    all_data = res['data']
    for i in range(len(all_data)):
        tmp_data = all_data[i]
        tmp_std = np.std(tmp_data)
        tmp_mean = np.mean(tmp_data)
        all_data[i] = (tmp_data - tmp_mean) / tmp_std
    # encode label
    all_label = []
    for i in res['label']:
        if i == 'N':
            all_label.append(0)
        elif i == 'A':
            all_label.append(1)
        elif i == 'O':
            all_label.append(2)
        elif i == '~':
            all_label.append(3)
    all_label = np.array(all_label)

    # split train test
    X_train, X_test, Y_train, Y_test = train_test_split(all_data, all_label, test_size=0.1, random_state=0)
    
    # slide and cut
    print('before: ')
    print(Counter(Y_train), Counter(Y_test))
    X_train, Y_train = slide_and_cut(X_train, Y_train, window_size=window_size, stride=stride)
    X_test, Y_test, pid_test = slide_and_cut(X_test, Y_test, window_size=window_size, stride=stride, output_pid=True)
    print('after: ')
    print(Counter(Y_train), Counter(Y_test))
    
    # shuffle train
    shuffle_pid = np.random.permutation(Y_train.shape[0])
    X_train = X_train[shuffle_pid]
    Y_train = Y_train[shuffle_pid]

    X_train = np.expand_dims(X_train, 1)
    X_test = np.expand_dims(X_test, 1)

    return X_train, X_test, Y_train, Y_test, pid_test
