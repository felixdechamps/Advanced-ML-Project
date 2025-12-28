import numpy as np
import torch
import json
import os
'cf jfrankle/open_lth'

class Mask(dict):
    def __init__(self, other_dict=None):
        super(Mask, self).__init__()
        if other_dict is not None:
            for k, v in other_dict.items(): 
                self[k] = v

    def __setitem__(self, key, value):
        if not isinstance(key, str) or len(key) == 0:
            raise ValueError('Invalid tensor name: {}'.format(key))
        if isinstance(value, np.ndarray):
            value = torch.as_tensor(value)
        if not isinstance(value, torch.Tensor):
            raise ValueError('value for key {} must be torch Tensor or numpy ndarray.'.format(key))
        if ((value != 0) & (value != 1)).any(): raise ValueError('All entries must be 0 or 1.')

        super(Mask, self).__setitem__(key, value)

    def ones_like(model):
        '''
       Initialize a mask filled with ones
        '''
        mask = Mask()
        for name in model.prunable_layers_names():
            mask[name] = torch.ones(list(model.state_dict()[name].shape)) 
        return mask

    def save(self, output_location):
        
        os.makedirs(output_location, exist_ok=True)

        # Save the mask
        mask_path = os.path.join(output_location, "mask.pt")
        torch.save({k: v.cpu().int() for k, v in self.items()},mask_path)

        # Save the sparsity report
        total_weights = np.sum([v.size for v in self.numpy().values()]).item()
        total_unpruned = np.sum([np.sum(v) for v in self.numpy().values()]).item()
        report = {"total": float(total_weights),"unpruned": float(total_unpruned)}
        report_path = os.path.join(output_location, "sparsity.json")
        with open(report_path, "w") as f:
            f.write(json.dumps(report, indent=4))
    def load(output_location):
        mask_path = os.path.join(output_location, "mask.pt")
        if not os.path.exists(mask_path):
            raise ValueError(f"Mask not found at {mask_path}")
        data = torch.load(mask_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        return Mask(data)
    
    def exists(output_location):
        return os.path.exists(os.path.join(output_location, "mask.pt"))

    def numpy(self):
        return {k: v.cpu().numpy() for k, v in self.items()}

    def sparsity(self):
        """Return the percent of weights that have been pruned as a decimal."""
        unpruned = torch.sum(torch.tensor([torch.sum(v) for v in self.values()]))
        total = torch.sum(torch.tensor([torch.sum(torch.ones_like(v)) for v in self.values()]))
        return 1 - unpruned.float() / total.float()

    def density(self):
        return 1 - self.sparsity