import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
import numpy as np
import random
import os
import json
import scipy.io as sio
import tqdm



def load_ecg(record, step):
    """
    taken from https://github.com/awni/ecg/blob/master/ecg/load.py

    Outputs the ecg sequence truncated to the closer multiple of step (eg 256).
    """
    if os.path.splitext(record)[1] == ".npy":
        ecg = np.load(record)
    elif os.path.splitext(record)[1] == ".mat":
        ecg = sio.loadmat(record)['val'].squeeze()
    else: # Assumes binary 16 bit integers
        with open(record, 'r') as fid:
            ecg = np.fromfile(fid, dtype=np.int16)

    trunc_samp = step * int(len(ecg) / step)
    return ecg[:trunc_samp]



def load_dataset(data_json, step=256):
    """
    taken from https://github.com/awni/ecg/blob/master/ecg/load.py

    Outputs :
            - ecgs : List of truncated ecg sequences
                eg of sequence : [  72   83   93 ... -136 -133 -131] (length 8960=35*256)
            - labels : List of List of labels for each intra ecg sequence of length 256
            eg of label : ['N', 'N', 'N', 'N', ..., 'N', 'N', 'N', 'N'] (length 35)
    """
    with open(data_json, 'r') as fid:
        data = [json.loads(l) for l in fid]
    labels = []; ecgs = []
    for d in tqdm.tqdm(data):
        labels.append(d['labels'])
        ecgs.append(load_ecg(d['ecg'], step))
    return ecgs, labels


class ECGDataset(Dataset):
    """
    PyTorch Dataset for ECG sequences with normalization and label encoding.
    """

    def __init__(self, x, y):
        """
        Args:
            x: List of input sequences (arrays).
            y: List of label sequences corresponding to x.
        """
        self.data_x = x
        self.data_y = y
        # Compute dataset mean and standard deviation
        self.mean, self.std = self._compute_mean_std(x)
        print("MEAN : ", self.mean, " STD : ", self.std)
        # Optimized padding value in normalized space
        self.pad_value_x_normalized = (0.0 - self.mean) / self.std
        
        # Determine classes and create label->index mapping
        all_labels = [label for seq in y for label in seq]
        self.classes = sorted(set(all_labels))
        print("self.classes : ", self.classes)
        self.class_to_int = {c: i for i, c in enumerate(self.classes)}
        print("self.class_to_int : ", self.class_to_int)
        self.num_classes = len(self.classes)

    def _compute_mean_std(self, x):
        """Compute mean and std over all sequences concatenated."""
        flat_x = np.hstack(x)
        return (np.mean(flat_x).astype(np.float32), 
                np.std(flat_x).astype(np.float32))

    def __len__(self):
        """Return number of sequences in the dataset."""
        return len(self.data_x)

    def __getitem__(self, idx):
        """
        Return normalized input tensor and integer-encoded label tensor for a given index.
        """
        x_val = self.data_x[idx]
        x_tensor = torch.tensor(x_val, dtype=torch.float32)
        x_tensor = (x_tensor - self.mean) / self.std
        
        y_indices = [self.class_to_int[label] for label in self.data_y[idx]]
        y_tensor = torch.tensor(y_indices, dtype=torch.long)
        return x_tensor, y_tensor


class SmartBatchSampler(Sampler):
    """
    Batch sampler that sorts sequences by length, groups them into batches, 
    and shuffles the batches to reduce padding overhead.
    """

    def __init__(self, data_source, batch_size):
        """
        Args:
            data_source: Dataset with attribute `data_x` containing sequences.
            batch_size: Number of samples per batch.
        """
        self.data_source = data_source
        self.batch_size = batch_size
        
        # Precompute sequence lengths and sort indices to minimize padding
        print("Sorting dataset by length to minimize padding...")
        self.sorted_indices = sorted(
            range(len(data_source)), 
            key=lambda i: len(data_source.data_x[i])
        )

    def __iter__(self):
        """Yield batches of indices, with batches shuffled."""
        # Split sorted indices into batches
        batches = [
            self.sorted_indices[i:i + self.batch_size]
            for i in range(0, len(self.sorted_indices), self.batch_size)
        ]
        
        # Shuffle the order of batches
        random.shuffle(batches)
        
        # Yield each batch
        for batch in batches:
            yield batch

    def __len__(self):
        """Return the total number of batches."""
        return (len(self.data_source) + self.batch_size - 1) // self.batch_size


class ECGCollate:
    """
    Collate function to pad variable-length ECG sequences and labels for batching.
    """

    def __init__(self, pad_val_x, num_classes, pad_val_y=-100):
        """
        Args:
            pad_val_x: Value used to pad input sequences.
            num_classes: Number of classes for labels.
            pad_val_y: Value used to pad label sequences (default: -100 for ignore_index).
        """
        self.pad_val_x = pad_val_x
        self.pad_val_y = pad_val_y
        self.num_classes = num_classes

    def __call__(self, batch):
        """
        Pad sequences and labels in the batch to the same length.

        Args:
            batch: List of tuples (x_tensor, y_tensor) from the dataset.

        Returns:
            x_final: Padded input tensor with shape (batch, 1, seq_len).
            y_padded: Padded label tensor with shape (batch, seq_len).
        """
        xs, ys = zip(*batch)

        # Pad sequences; sequences are already roughly similar in length due to SmartBatchSampler
        x_padded = rnn_utils.pad_sequence(xs, batch_first=True, padding_value=self.pad_val_x)
        y_padded = rnn_utils.pad_sequence(ys, batch_first=True, padding_value=self.pad_val_y)
        
        # Add channel dimension for input
        x_final = x_padded.unsqueeze(-2)
        return x_final, y_padded

