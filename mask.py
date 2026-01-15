# this codelines are adapted from Frankle's open_lth GitHub Repository:
# https://github.com/facebookresearch/open_lth/blob/main/pruning/mask.py

import numpy as np
import torch
from resnet1d import ResNet1D


class Mask(dict):
    def __init__(self, other_dict=None):
        super(Mask, self).__init__()
        if other_dict is not None:
            for k, v in other_dict.items(): self[k] = v

    def __setitem__(self, key, value):
        if not isinstance(key, str) or len(key) == 0:
            raise ValueError('Invalid tensor name: {}'.format(key))
        if isinstance(value, np.ndarray):
            value = torch.as_tensor(value)
        if not isinstance(value, torch.Tensor):
            raise ValueError('value for key {} must be torch Tensor or numpy ndarray.'.format(key))
        if ((value != 0) & (value != 1)).any(): raise ValueError('All entries must be 0 or 1.')

        super(Mask, self).__setitem__(key, value)

    @staticmethod
    def ones_like(model: ResNet1D) -> 'Mask':
        mask = Mask()
        for name in model.prunable_layer_names:
            mask[name] = torch.ones(list(model.state_dict()[name].shape))
        return mask

    def numpy(self):
        return {k: v.cpu().numpy() for k, v in self.items()}

    @property
    def sparsity(self):
        """Return the percent of weights that have been pruned as a decimal."""

        unpruned = torch.sum(torch.tensor([torch.sum(v) for v in self.values()]))
        total = torch.sum(torch.tensor([torch.sum(torch.ones_like(v)) for v in self.values()]))
        return 1 - unpruned.float() / total.float()

    @property
    def density(self):
        return 1 - self.sparsity

    def layerwise_sparsity(self):
        """
        Return a dictionary {layer_name: sparsity} for each layer.
        Sparsity is a float between 0.0 (dense) and 1.0 (empty).
        """
        layer_sparsities = {}
        for k, v in self.items():
            # v is a PyTorch Tensor containing 0s and 1s
            total_params = v.numel()  # Total number of elements in the tensor
            unpruned_params = v.sum().item()  # Sum of 1s (kept elements)
            
            # Computation: 1 - (kept / total)
            sparsity = 1.0 - (unpruned_params / total_params)
            layer_sparsities[k] = sparsity
            
        return layer_sparsities

    def layerwise_remaining_params(self):
        """
        Returns a dictionary {layer_name: number_of_remaining_parameters} for each layer.
        Useful for reproducing Figure 2 of the LTH-ECG paper.
        """
        layer_counts = {}
        for k, v in self.items():
            # v is the mask (0 or 1). The sum gives the number of active weights.
            # Cast to int to get a clean integer value.
            count = int(v.sum().item())
            layer_counts[k] = count
            
        return layer_counts
