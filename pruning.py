import numpy as np
from mask import Mask
from resnet1d_Aurane import ResNet1D

class PrunedModel(ResNet1D):

    def to_mask_name(name):
        return 'mask_' + name.replace('.', '___')

    def __init__(self, model: Model, mask: Mask):
        if isinstance(model, PrunedModel): raise ValueError('Cannot nest pruned models.')
        super(PrunedModel, self).__init__()
        self.model = model

        for k in self.model.prunable_layer_names:
            if k not in mask: raise ValueError('Missing mask value {}.'.format(k))
            if not np.array_equal(mask[k].shape, np.array(self.model.state_dict()[k].shape)): # A modifier
                raise ValueError('Incorrect mask shape {} for tensor {}.'.format(mask[k].shape, k))

        for k in mask:
            if k not in self.model.prunable_layer_names:
                raise ValueError('Key {} found in mask but is not a valid model tensor.'.format(k))

        for k, v in mask.items(): self.register_buffer(PrunedModel.to_mask_name(k), v.float())
        self._apply_mask()

    def _apply_mask(self):
        for name, param in self.model.named_parameters(): # A modifier
            if hasattr(self, PrunedModel.to_mask_name(name)):
                param.data *= getattr(self, PrunedModel.to_mask_name(name))

    def forward(self, x):
        self._apply_mask()
        return self.model.forward(x) # A modifier 

    def prunable_layer_names(self):
        return self.model.prunable_layer_names()

    def output_layer_names(self):
        return self.model.output_layer_names # A modifier

    def loss_criterion(self):
        return self.model.loss_criterion # A modifier

    def save(self, save_location, save_step):
        self.model.save(save_location, save_step) # Idem ?


