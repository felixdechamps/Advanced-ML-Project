import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import torch
from torch import optim
from torch.utils.data import DataLoader
from torchsummary import summary


from resnet1d_Aurane import ResNet1D
from mask import Mask
from sparse_global import prune
from pruning import PrunedModel



def reset_to_initial_weights(model: ResNet1D, initial_state_dict: dict, mask: Mask):
    '''
    Reset the surviving weights to their initial values, and keep pruned weights at zero.
    '''
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in mask:
                param.copy_(initial_state_dict[name] * mask[name].to(param.device))
            else:
                param.copy_(initial_state_dict[name])



def train_model(model, train_loader, val_loader, optimizer, criterion, epochs, device, verbose, compute_metrics):
    # Fonction ChatGPT : A REGARDER ET CORRIGER
    
    '''
    Train the model and returns 

    compute_metrics : boolean, if True returns accuracy/f1-score per epoch.
    '''
    model.to(device)
    metrics = {"train_loss": [], "val_loss": [], "val_accuracy": [], "val_f1": []}

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        all_train_preds, all_train_labels = [], []

        for batch in train_loader:
            x, y = tuple(t.to(device) for t in batch)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            if compute_metrics:
                all_train_preds.append(logits.detach().cpu())
                all_train_labels.append(y.detach().cpu())

        epoch_loss /= len(train_loader)
        metrics["train_loss"].append(epoch_loss)

        # Validation metrics
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            all_val_preds, all_val_labels = [], []
            with torch.no_grad():
                for batch in val_loader:
                    x, y = tuple(t.to(device) for t in batch)
                    logits = model(x)
                    loss = criterion(logits, y)
                    val_loss += loss.item()
                    if compute_metrics:
                        all_val_preds.append(logits.cpu())
                        all_val_labels.append(y.cpu())

            val_loss /= len(val_loader)
            metrics["val_loss"].append(val_loss)

            if compute_metrics:
                all_val_preds = torch.cat(all_val_preds)
                all_val_labels = torch.cat(all_val_labels)
                preds = torch.argmax(all_val_preds, dim=1)
                metrics["val_accuracy"].append((preds == all_val_labels).float().mean().item())

                report = classification_report(all_val_labels, preds, output_dict=True)
                f1 = np.mean([report[str(c)]["f1-score"] for c in range(preds.max().item()+1)])
                metrics["val_f1"].append(f1)

        if verbose:
            msg = f"Epoch {epoch+1}/{epochs} | Train Loss: {epoch_loss:.4f}"
            if val_loader is not None:
                msg += f" | Val Loss: {val_loss:.4f}"
            if compute_metrics and val_loader is not None:
                msg += f" | Val Acc: {metrics['val_accuracy'][-1]:.4f} | Val F1: {metrics['val_f1'][-1]:.4f}"
            print(msg)

    return metrics



def lottery_ticket(device, model: ResNet1D, train_loader, pruning_fraction: float, rounds: int, epochs_per_round: int, layers_to_ignore: str = ""):
    '''
    Implement the lottery ticket hypothesis.
    '''
    initial_state = {
        k: v.clone().detach().cpu()
        for k, v in model.state_dict().items()
    }

    # Initialize the mask
    mask = Mask.ones_like(model)

    for r in range(rounds):
        print(f"\n Round {r+1}/{rounds}")
        print(f"Sparsity before training: {mask.sparsity():.2%}")

        # Train the masked model
        pruned_model = PrunedModel(model, mask).to(device)
        optimizer = torch.optim.SGD(pruned_model.parameters(), lr=0.01, momentum=0.9) # A modifier ?

        train_model(pruned_model, train_loader, optimizer, model.loss_criterion, epochs_per_round, device) # A modifier : écrire une fonction d'entraînement

        # Prune
        mask = prune(pruning_fraction=pruning_fraction, layers_to_ignore=layers_to_ignore, trained_model=model, current_mask=mask)

        print(f"Sparsity after pruning: {mask.sparsity():.2%}")

        # Reset surviving weights to their inital value
        reset_to_initial_weights(model, initial_state, mask)

    return model, mask

