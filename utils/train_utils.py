import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from tqdm import tqdm
from torchmetrics.classification import MulticlassF1Score


def create_tqdm_bar(iterable, desc):
    return tqdm(enumerate(iterable), total=len(iterable), ncols=150, desc=desc)


def calculate_priors(train_loader, device, num_classes=4, smooth=500):
    """
    Compute class prior probabilities from the training set.

    Handles sequential labels by using the first label of each sequence.
    """
    counts = torch.zeros(num_classes)

    # Automatically detect device from the dataloader
    device = next(iter(train_loader))[1].device
    
    for _, labels in train_loader:
        # If labels are (batch, sequence), keep the first label per sample
        if labels.dim() > 1:
            labels = labels[:, 0]
            
        for l in labels:
            counts[l.long().item()] += 1
    
    # Apply additive smoothing
    total = counts.sum() + num_classes
    prior = (counts + smooth) / total

    return prior.to(device)


def weights_init_kaiming(m):
    """
    Apply Kaiming He (He Normal) initialization for layers with ReLU activation.

    To be used with `model.apply(weights_init_kaiming)`.
    """
    classname = m.__class__.__name__
    
    # Initialize Conv1d and Linear layers
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        # He Normal initialization with fan_out mode, suitable for ResNets
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        
        # Initialize biases to zero if present
        if m.bias is not None:
            init.constant_(m.bias, 0)


def train_model(model, train_loader, val_loader, loss_func, device, prior, epochs=10):
    """
    Train the classifier for a number of epochs.
    """
    loss_cutoff = len(train_loader) // 10

    # Configuration of F1-Score for 4 classes (N, A, O, ~)
    f1_metric = MulticlassF1Score(num_classes=4, average=None).to(device)
    best_f1_val = 0.0
    f1_through_training = []
    optimizer = torch.optim.Adam(model.parameters(), 0.001)  # like in Hannun et al.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode='min',
                                                           factor=0.1,  # like in Hannun et al.
                                                           patience=2  # "two consecutive epochs"
                                                           )
    # Training
    for epoch in range(epochs):

        # Training stage, where we want to update the parameters.
        model.train()

        # initialize epoch metrics
        training_loss = []
        validation_loss = []
        f1_metric.reset()

        # Create a progress bar for the training loop.
        training_loop = create_tqdm_bar(train_loader, desc=f'Training Epoch [{epoch + 1}/{epochs}]')

        for train_iteration, batch in training_loop:

            optimizer.zero_grad()  # reset gradient values
            ecgs, labels = batch
            ecgs, labels = ecgs.to(device), labels.to(device)

            pred = model(ecgs)  # shape = (batch size, nb of classes, nb of time steps)
            loss = loss_func(pred, labels)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                probs = F.softmax(pred, dim=1).transpose(1, 2)
                adjusted_probs = probs / prior
                indices = torch.argmax(adjusted_probs, dim=2)
                final_preds, _ = torch.mode(indices, dim=1)  # vote for the most predicted class
                final_labels = labels[:, 0] if labels.dim() > 1 else labels
                f1_metric.update(final_preds, final_labels)

            training_loss.append(loss.item())
            training_loss = training_loss[-loss_cutoff:]
            current_f1s = f1_metric.compute()
            current_cinc = torch.mean(current_f1s[:3]).item()
            f1_through_training.append(current_cinc)
            # Update the progress bar.
            training_loop.set_postfix(curr_train_loss="{:.8f}".format(np.mean(training_loss)),
                                      curr_train_f1=f"{current_cinc:.4f}",
                                      lr="{:.8f}".format(optimizer.param_groups[0]['lr'])
                                      )
        # Validation
        model.eval()
        # reset metric for val set
        f1_metric.reset()
        # Create a progress bar for the training loop.
        val_loop = create_tqdm_bar(val_loader, desc=f'Validation Epoch [{epoch + 1}/{epochs}]')
        with torch.no_grad():
            for val_iteration, batch in val_loop:
                ecgs, labels = batch
                ecgs, labels = ecgs.to(device), labels.to(device)

                pred = model(ecgs)
                loss = loss_func(pred, labels)
                validation_loss.append(loss.item())

                # Compute f1 score on the batch
                probs = F.softmax(pred, dim=1).transpose(1, 2)
                adjusted_probs = probs / prior  # adjust with the priors
                indices = torch.argmax(adjusted_probs, dim=2)
                final_preds, _ = torch.mode(indices, dim=1)
                final_labels = labels[:, 0] if labels.dim() > 1 else labels
                f1_metric.update(final_preds, final_labels)
                current_val_f1s = f1_metric.compute()
                current_val_cinc = torch.mean(current_val_f1s[:3]).item()

                # Update the progress bar.
                val_loop.set_postfix(val_loss="{:.8f}".format(np.mean(validation_loss)),
                                     f1_val=f"{current_val_cinc:.4f}")

        # Compute end of epoch f1 score
        per_class_f1 = f1_metric.compute()
        cinc_f1 = torch.mean(per_class_f1[:3]).item()

        if cinc_f1 > best_f1_val:
            best_f1_val = cinc_f1

        scheduler.step(np.mean(validation_loss))

        # Early Stopping :
        # If the F1 exceeds 82.6% (0.826), we consider that the ticket has converged 
        # with a 1% tolerance relative to the 83.6% reported in the paper
        if cinc_f1 >= 0.826:
            print(f"\n[Early Stopping] F1 Val ({cinc_f1:.4f}) >= 0.826. End of the training for this round")
            break

    return model, best_f1_val, f1_through_training
