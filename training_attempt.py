import os
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from main_Aurane_py import X_proc, Y_int

# Parameters

RANDOM_SEED = 42
BATCH_SIZE = 32
NUM_EPOCHS = 25 # à changer
LR = 1e-3 # à changer
VAL_SPLIT = 0.15

model_path="models.py"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


class ECGDataset(Dataset):
    def __init__(self, X, y):
        ''' 
        X: np.array (N, L, 1)
        y: np.array (N,)
        '''
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, ind):
        x = self.X[ind]                # shape (L, 1)
        x = np.transpose(x, (1, 0))    
        return torch.from_numpy(x), torch.tensor(self.y[ind], dtype=torch.long)
    
X_train, X_val, y_train, y_val = train_test_split(
    X_proc, Y_int, test_size=VAL_SPLIT, stratify=Y_int, random_state=RANDOM_SEED
)

train_dataset = ECGDataset(X_train, y_train)
val_dataset   = ECGDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

num_classes = len(np.unique(Y_int))
from models import model

