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
from pathlib import Path



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
    project_root = Path(__file__).resolve().parent
    data_root = project_root.parent
    data_path = data_root / data_json
    with open(data_path, 'r') as fid:
        data = [json.loads(l) for l in fid]
    labels = []
    ecgs = []
    for d in tqdm.tqdm(data):
        labels.append(d['labels'])
        ecgs.append(load_ecg(d['ecg'], step))
    return ecgs, labels


# --- 1. Le Dataset (Identique à la version optimisée) ---
class ECGDataset(Dataset):
    def __init__(self, x, y):
        self.data_x = x
        self.data_y = y
        self.mean, self.std = self._compute_mean_std(x)
        print("MEAN : ", self.mean, " STD : ", self.std)
        # Valeur de padding optimisée : (0 - mean) / std
        self.pad_value_x_normalized = (0.0 - self.mean) / self.std
        
        all_labels = [label for seq in y for label in seq]
        self.classes = sorted(set(all_labels))
        print("self.classes : ", self.classes)
        self.class_to_int = {c: i for i, c in enumerate(self.classes)}
        print("self.class_to_int : ", self.class_to_int)
        self.num_classes = len(self.classes)

    def _compute_mean_std(self, x):
        flat_x = np.hstack(x)
        return (np.mean(flat_x).astype(np.float32), 
                np.std(flat_x).astype(np.float32))

    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, idx):
        # Normalizing data
        x_val = self.data_x[idx]
        x_tensor = torch.tensor(x_val, dtype=torch.float32)
        x_tensor = (x_tensor - self.mean) / self.std
        
        y_indices = [self.class_to_int[label] for label in self.data_y[idx]]
        y_tensor = torch.tensor(y_indices, dtype=torch.long)
        return x_tensor, y_tensor


class SmartBatchSampler(Sampler):
    """
    Reproduit la logique : 
    1. Trier les exemples par taille.
    2. Faire des paquets (batches).
    3. Mélanger les paquets.
    """
    def __init__(self, data_source, batch_size):
        self.data_source = data_source
        self.batch_size = batch_size
        
        # On pré-calcule les longueurs et on trie les indices
        # data_source.data_x suppose que l'accès est direct.
        print("Tri du dataset par longueur pour minimiser le padding...")
        self.sorted_indices = sorted(
            range(len(data_source)), 
            key=lambda i: len(data_source.data_x[i])
        )

    def __iter__(self):
        # 1. On découpe les indices triés en paquets de taille batch_size
        batches = [
            self.sorted_indices[i:i + self.batch_size]
            for i in range(0, len(self.sorted_indices), self.batch_size)
        ]
        
        # 2. On mélange l'ordre des batches (comme 'random.shuffle(batches)' dans ton code)
        random.shuffle(batches)
        
        # 3. On yield chaque batch (qui est une liste d'indices)
        for batch in batches:
            yield batch

    def __len__(self):
        return (len(self.data_source) + self.batch_size - 1) // self.batch_size

# --- 3. Le Collate (Identique à la version optimisée) ---
class ECGCollate:
    def __init__(self, pad_val_x, num_classes, pad_val_y=-100):
        self.pad_val_x = pad_val_x
        self.pad_val_y = pad_val_y
        self.num_classes = num_classes

    def __call__(self, batch):
        xs, ys = zip(*batch)
        # Ici, xs contient des séquences de longueurs très proches grâce au Sampler !
        x_padded = rnn_utils.pad_sequence(xs, batch_first=True, padding_value=self.pad_val_x)
        y_padded = rnn_utils.pad_sequence(ys, batch_first=True, padding_value=self.pad_val_y)
        
        x_final = x_padded.unsqueeze(-2)
        #y_onehot = F.one_hot(y_padded, num_classes=self.num_classes).float()
        return x_final, y_padded
