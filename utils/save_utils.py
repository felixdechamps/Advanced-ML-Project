import torch
from utils.pruning_utils import Mask


def save_checkpoint(state, filename: str = "lth_checkpoint.pth"):
    """Save complete experience variable in a checkpoint .pth file """
    print(f"--> Checkpoint backup : {filename}")
    torch.save(state, filename)


def load_checkpoint(filename="lth_checkpoint.pth"):
    """Load experience state from checkpoint"""
    print(f"--> Checkpoint loading : {filename}")
    # Detect if CUDA is available, otherwise use CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load the checkpoint to the detected device
    return torch.load(filename, map_location=device, weights_only=False)
