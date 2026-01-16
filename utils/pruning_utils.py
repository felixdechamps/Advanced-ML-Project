
import numpy as np
import torch
import torch.nn as nn
from resnet1d import ResNet1D


class Mask(dict):
    """
    Custom mask class for pruning.

    this codelines are adapted from Frankle's open_lth GitHub Repository:
    https://github.com/facebookresearch/open_lth/blob/main/pruning/mask.py
    
    """
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

def prune(pruning_fraction: float = 0.2, pruning_layers_to_ignore: str = None, trained_model=None, current_mask: Mask = None) : 
    """
    A one iteration of pruning : returns the new updated mask after pruning.

    trained_model : the original fully trained model.
    pruning_fraction = The fraction of additional weights to prune from the network.
    layers_to_ignore = A comma-separated list of addititonal tensors that should not be pruned.

    This function is a reprodution from Frankle open_lth GitHub repo :
    https://github.com/facebookresearch/open_lth/blob/main/pruning/sparse_global.py
    """
    current_mask = Mask.ones_like(trained_model).numpy() if current_mask is None else current_mask.numpy()

    # Determine the number of weights that need to be pruned.
    number_of_remaining_weights = np.sum([np.sum(v) for v in current_mask.values()])
    number_of_weights_to_prune = np.ceil(pruning_fraction * number_of_remaining_weights).astype(int)

    # Determine which layers can be pruned.
    prunable_tensors = set(trained_model.prunable_layer_names)
    if pruning_layers_to_ignore:
        prunable_tensors -= set(pruning_layers_to_ignore.split(','))
    
    # Get the model weights.
    weights = {k: v.clone().cpu().detach().numpy()
                for k, v in trained_model.state_dict().items()
                if k in prunable_tensors}

    # Create a vector of all the unpruned weights in the model.
    weight_vector = np.concatenate([v[current_mask[k] == 1] for k, v in weights.items()])
    threshold = np.sort(np.abs(weight_vector))[number_of_weights_to_prune]

    new_mask = Mask({k: np.where(np.abs(v) > threshold, current_mask[k], np.zeros_like(v))
                        for k, v in weights.items()})
    for k in current_mask:
        if k not in new_mask:  # if this weight was already pruned add it to the new mask
            new_mask[k] = current_mask[k]

    return 


class PrunedModel(nn.Module):
    """
    Wraps a model with binary masks applied to prunable parameters.

    Adapted from the open_lth repository:
    https://github.com/facebookresearch/open_lth/blob/main/pruning/pruned_model.py
    """

    @staticmethod
    def to_mask_name(name):
        """Convert a parameter name to its corresponding mask buffer name."""
        return 'mask_' + name.replace('.', '___')

    def __init__(self, model: ResNet1D, mask: Mask):
        """
        Initialize a pruned model by attaching masks to a base model.

        Args:
            model: Base neural network to prune.
            mask: Dictionary mapping parameter names to binary masks.
        """
        if isinstance(model, PrunedModel):
            raise ValueError('Cannot nest pruned models.')
        super(PrunedModel, self).__init__()
        self.model = model

        # Validate that every prunable parameter has a matching mask
        for k in self.model.prunable_layer_names:
            if k not in mask:
                raise ValueError(f'Missing mask value {k}.')
            if not np.array_equal(mask[k].shape,
                                np.array(self.model.state_dict()[k].shape)):
                raise ValueError(
                    f'Incorrect mask shape {mask[k].shape} for tensor {k}.'
                )

        # Ensure mask does not contain extra keys
        for k in mask:
            if k not in self.model.prunable_layer_names:
                raise ValueError(
                    f'Key {k} found in mask but is not a valid model tensor.'
                )

        device = next(model.parameters()).device

        # Register masks as non-trainable buffers
        for k, v in mask.items():
            self.register_buffer(
                PrunedModel.to_mask_name(k),
                v.float().to(device)
            )

        self._apply_mask()

    def _apply_mask(self):
        """Apply masks in-place to the model parameters."""
        for name, param in self.model.named_parameters():
            if hasattr(self, PrunedModel.to_mask_name(name)):
                param.data *= getattr(self, PrunedModel.to_mask_name(name))

    def forward(self, x):
        """Apply masks and forward the input through the model."""
        self._apply_mask()
        return self.model.forward(x)

    @property
    def prunable_layer_names(self):
        """Expose prunable layer names from the wrapped model."""
        return self.model.prunable_layer_names

