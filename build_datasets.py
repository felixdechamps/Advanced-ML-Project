# This code is the setup code use by in the repo https://github.com/awni/ecg.git
# where Hannun et al. tested there 34 layers AF Classifier on Physionet 2017 Challenge data 

import json
import numpy as np
import os
import random
import scipy.io as sio
import tqdm

STEP = 256

def load_ecg_mat(ecg_file):
    """
    Load an ECG signal from a .mat file.

    Args:
        ecg_file (str): Path to the MATLAB (.mat) file.

    Returns:
        np.ndarray: ECG signal array.
    """
    return sio.loadmat(ecg_file)['val'].squeeze()

def load_all(data_path):
    """
    Load all ECG records and their labels from a dataset directory.

    Reads the reference CSV file, loads each ECG signal, and expands labels
    to match the number of segments per record.

    Args:
        data_path (str): Path to the directory containing ECG .mat files.

    Returns:
        list: List of tuples (ecg_file_path, label_sequence).
    """
    label_file = os.path.join(data_path, "../REFERENCE-v3.csv")
    with open(label_file, 'r') as fid:
        records = [l.strip().split(",") for l in fid]

    dataset = []
    for record, label in tqdm.tqdm(records):
        ecg_file = os.path.join(data_path, record + ".mat")
        ecg_file = os.path.abspath(ecg_file)
        ecg = load_ecg_mat(ecg_file)

        num_labels = int(ecg.shape[0] / STEP)
        dataset.append((ecg_file, [label] * num_labels))

    return dataset


def split(dataset, dev_frac):
    """
    Split a dataset into training and development subsets.

    Args:
        dataset (list): Full dataset to split.
        dev_frac (float): Fraction of data to use for the development set.

    Returns:
        train (list): Training subset.
        dev (list): Development subset.
    """
    dev_cut = int(dev_frac * len(dataset))
    random.shuffle(dataset)
    dev = dataset[:dev_cut]
    train = dataset[dev_cut:]
    return train, dev



def make_json(save_path, dataset):
    """
    Save a dataset to a JSON Lines file.

    Each line contains a JSON object with the ECG file path and its labels.

    Args:
        save_path (str): Path to the output JSON file.
        dataset (list): Dataset as a list of (ecg_path, labels) tuples.
    """
    with open(save_path, 'w') as fid:
        for d in dataset:
            datum = {
                'ecg': d[0],
                'labels': d[1]
            }
            json.dump(datum, fid)
            fid.write('\n')



if __name__ == "__main__":
    random.seed(2018)

    dev_frac = 0.1
    data_path = "data/training2017/"
    dataset = load_all(data_path)
    train, dev = split(dataset, dev_frac)
    make_json("train.json", train)
    make_json("dev.json", dev)
