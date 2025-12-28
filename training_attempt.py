import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import torch
from torch import optim
from torch.utils.data import DataLoader
from torchsummary import summary

from main_Aurane_py import X_pad, Y_int, process_x
from resnet1d_Aurane import MyDataset, ResNet1D

# Parameters

RANDOM_SEED = 42
BATCH_SIZE = 32
LR = 0.001
VAL_SPLIT = 0.15

MODEL_PATH = "resnet1d.py"

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# Transpose the data (N,L,1) to (N, 1, L)

X_torch = X_torch = X_pad[:, None, :]

# Train and process
X_train, X_test, y_train, y_test = train_test_split(
    X_torch, Y_int, test_size=VAL_SPLIT, stratify=Y_int, random_state=RANDOM_SEED
)
X_train_norm = process_x(X_train)
X_test_norm = process_x(X_test)

train_dataset = MyDataset(X_train_norm, y_train)
test_dataset = MyDataset(X_test_norm, y_test)

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
num_classes = len(np.unique(Y_int))

# Model
device_str = "cuda"
device = torch.device(device_str if torch.cuda.is_available() else "cpu")
kernel_size = 16  # durée de l'ECG*fréquence ?
stride = 2
n_block = 48  # Proposition de ChatGPT : 16 (pour 32 convolutions si deux convolutions par bloc); Proposition de l'article : 48
downsample_gap = 6
increasefilter_gap = 12
downsample_gap = 6  # 2 dans un autre repo
increasefilter_gap = 12  # 4 dans un autre repo

model = ResNet1D(in_channels=1,
               base_filters=64,  # Proposition de ChatGPT : 32/ Article 128 ou 64
               kernel_size=kernel_size,
               stride=stride,
               groups=8,  # Proposition de ChatGPT : 1/ 8 dans un autre repo/ 32 sinon
               n_block=n_block,
               n_classes=4,
               downsample_gap=downsample_gap,
               increasefilter_gap=increasefilter_gap,
               use_bn=True,
               use_do=True,
               verbose=True)

model.to(device)

summary(model, (X_train.shape[1], X_train.shape[2]), device=device_str)

model.verbose = False
loss_func = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)  # Reduces the learning rate once learning stagnate

n_epoch = 50
step = 0
f1_score_list = []

for epoch in range(n_epoch):

    # Train
    model.train()
    all_pred_train = []
    for batch_idx, batch in enumerate(train_loader):
        input_x, input_y = tuple(t.to(device) for t in batch)
        pred = model(input_x)
        all_pred_train.append(pred.cpu().data.numpy())
        loss = loss_func(pred, input_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        step += 1
        
    all_pred_train = np.concatenate(all_pred_train)
    all_pred_train_fin = np.argmax(all_pred_train, axis=1)
    print(classification_report(y_train, all_pred_train_fin))
    print(confusion_matrix(y_train, all_pred_train_fin))
    
    # Test
    model.eval()
    tot_loss = 0.0
    all_predictions = []  # outputs
    with torch.no_grad():
        for batch in test_loader:
            input_x, input_y = tuple(t.to(device) for t in batch)
            pred = model(input_x)
            loss_test = loss_func(pred, input_y)
            tot_loss += loss_test.item()
            all_predictions.append(pred.cpu().data.numpy())
    
    tot_loss /= len(test_loader)
    scheduler.step(tot_loss)

    all_predictions = np.concatenate(all_predictions)
    all_pred = np.argmax(all_predictions, axis=1)
    report = classification_report(y_test, all_pred, output_dict=True)
    print(confusion_matrix(y_test, all_pred))
    f1_score = (report['0']['f1-score'] + report['1']['f1-score'] + report['2']['f1-score'] + report['3']['f1-score'])/4
    print(f"\n Epoch {epoch} F1/f1_score {f1_score}")
    f1_score_list.append(f1_score)

plt.plot(f1_score_list)
plt.xlabel("Epoch")
plt.ylabel("F1-score")
plt.show()