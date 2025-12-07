import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import urllib.request
import zipfile
import wfdb
import os
import glob
import tensorflow

data_path = 'training/training2017/'

# Load the labels
labels_df2=pd.read_csv(os.path.join(data_path, "REFERENCE.csv"), header=None, names=["record","label"])
print(f"\n All the possible labels are {pd.unique(labels_df2.label)}.")

# Turning the training record files into an array
# (IL FAUDRAIT RENDRE LE CODE UN PEU PLUS EFFICACE)

records = glob.glob(os.path.join(data_path, "*.hea"))

X = []
Y = []

for rec in records:
    record_name = os.path.basename(rec).split('.')[0]
    record = wfdb.rdrecord(os.path.join(data_path, record_name))
    signal = record.p_signal[:,0]
    freq=record.fs
    y=labels_df2[labels_df2.record==record_name]["label"].values[0]
    Y.append(y)
    X.append(signal)
    
Y = np.array(Y)

min_window_size=min([len(s) for s in X])
print(f"\n The shortest window is of size {min_window_size} eg corresponding to {round(min_window_size/freq,1)} s.")
print(f"\n The median window is of size {np.median([len(s) for s in X])}")

def pad_signal(signal, target_len, val=0):
    '''
    Returns a truncated signal of size equal to target_len, filled with zeros in when original signal is shorter than the targeted one.
    '''
    if len(signal) >= target_len:
        return signal[:target_len]        
    else:
        padded = np.full(target_len, val, dtype=signal.dtype)
        padded[:len(signal)] = signal     
        return padded

def compute_mean_std(x):
    x = np.hstack(x)
    return (np.mean(x).astype(np.float32),
           np.std(x).astype(np.float32))

def process_x(x):
    mean, std = compute_mean_std(x)
    x = (x - mean) /std
    x = x[:, :, None]
    return x

# Getting X and Y
X_pad=np.array([pad_signal(s,min_window_size,0) for s in X])
X_proc=process_x(X_pad)
classes=np.unique(Y)
class_to_int = {c:i for i,c in enumerate(classes)}
print(f"\n The conversion of the labels gives {class_to_int}.")
Y_int = np.array([class_to_int[c] for c in Y])

