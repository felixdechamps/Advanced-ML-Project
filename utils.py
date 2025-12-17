import numpy as np
import pandas as pd
import scipy.io
from matplotlib import pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from collections import Counter
from tqdm import tqdm


def preprocess_physionet():
    """
    download the raw data from https://physionet.org/content/challenge-2017/1.0.0/,
    and put it in ../data/challenge2017/
    """

    # read label
    # label_df = pd.read_csv('../data/challenge2017/REFERENCE-v3.csv', header=None)
    label_df = pd.read_csv("https://physionet.org/files/challenge-2017/1.0.0/training/REFERENCE.csv", header=None, names=['filename', 'label'])
    label = label_df.iloc[:, 1].values
    print(Counter(label))

    # read data
    all_data = []
    filenames = pd.read_csv('../training2017', header=None)
    filenames = filenames.iloc[:, 0].values
    print(filenames)
    for filename in tqdm(filenames):
        mat = scipy.io.loadmat('../training2017/{0}.mat'.format(filename))
        mat = np.array(mat['val'])[0]
        all_data.append(mat)
    all_data = np.array(all_data)

    res = {'data': all_data, 'label': label}
    with open('../challenge2017.pkl', 'wb') as fout:
        pickle.dump(res, fout)
