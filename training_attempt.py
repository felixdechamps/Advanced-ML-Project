import os
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary

from main_Aurane_py import X_pad, Y_int, process_x, compute_mean_std
from resnet1d import MyDataset, ResNet1D



# Parameters

RANDOM_SEED = 42
BATCH_SIZE = 32
NUM_EPOCHS = 25 # à changer
LR = 1e-3 
VAL_SPLIT = 0.15

MODEL_PATH="resnet1d.py"


np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# Transpose the data (N,L,1) to (N, 1, L)
print(X_pad.shape)

X_torch = np.transpose(X_pad, (0, 2, 1)) 

# Train and process
X_train, X_val, y_train, y_val = train_test_split(
    X_torch, Y_int, test_size=VAL_SPLIT, stratify=Y_int, random_state=RANDOM_SEED
)
X_train_norm=process_x(X_train)
X_val_norm=process_x(X_val)

train_dataset = MyDataset(X_train_norm, y_train)
val_dataset   = MyDataset(X_val_norm, y_val)

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
num_classes = len(np.unique(Y_int))

# Model
device_str = "cuda"
device = torch.device(device_str if torch.cuda.is_available() else "cpu")
kernel_size= 16 # durée de l'ECG*fréquence ?
stride=2
n_block = 48 # Proposition de ChatGPT : 16 (pour 32 convolutions si deux convolutions par bloc)
downsample_gap = 6
increasefilter_gap = 12
downsample_gap=6
increasefilter_gap=12

model=ResNet1D(in_channels=1,
               base_filters=64, # Proposition de ChatGPT : 32
               kernel_size=kernel_size,
               stride=stride,
               groups= 32, # Proposition de ChatGPT : 1
               n_block=n_block,
               n_classes=4,
               downsample_gap=downsample_gap ,
               increasefilter_gap=increasefilter_gap,
               use_bn=True ,
               use_do=True ,
               verbose=True)

model.to(device)

summary(model, (X_train.shape[1], X_train.shape[2]), device=device_str)

# A FAIRE : revérifier qu'on a bien transposé les données
# dans la bonne direction, puis reprendre bien le code d'entraînement dans l'ordre.