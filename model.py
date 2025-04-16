from typing import Optional

import torch
import torchvision.models as models
import torch.nn.functional as F
from torch import nn
from huggingface_hub import hf_hub_download


class MaskedConv2d(nn.Conv2d):
    """Convolutional layer that supports masking for individual filters"""

    def __init__(self, *args, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)
        self.register_buffer('_mask', torch.ones_like(self.weight))
        self._mask = self._mask.to(self.weight.device)

    def forward(self, input):
        weight = self.weight * self._mask
        return F.conv2d(
            input,
            weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

    @property
    def mask(self):
        return self._mask[:, 0, 0, 0]

    @mask.setter
    def mask(self, m):
        """Mask m is a one-dimensional vector with 1/0 for every convolutional filter in the layer"""
        assert (
            m.shape[0] == self.out_channels
        ), "Mask length must match the number of filters"
        self._mask = m.view(-1, 1, 1, 1).float().expand_as(self.weight).to(self.weight.device)


# TODO cleanup
def replace_conv_with_custom(model, device):
    chromosome_len = 0
    for name, module in model.named_children():
        # Recursively replace in submodules
        chromosome_len += replace_conv_with_custom(module, device)

        if isinstance(module, nn.Conv2d):
            # Get arguments from existing module
            new_module = MaskedConv2d(
                module.in_channels,
                module.out_channels,
                module.kernel_size,
                module.stride,
                module.padding,
                module.dilation,
                module.groups,
                module.bias is not None,
                module.padding_mode,
            ).to(device)
            # Copy weights and bias if necessary
            # Useless, model was not loaded, but hey, maybe keep it jus tin case
            new_module.weight.data.copy_(module.weight.data)
            if module.bias is not None:
                new_module.bias.data.copy_(module.bias.data)

            # Replace the module in parent
            setattr(model, name, new_module)
            chromosome_len += module.out_channels
    return chromosome_len


def set_mask_on(model, chromosome, i=0):
    for name, module in model.named_children():
        # Recursively replace in submodules
        i = set_mask_on(module, chromosome, i)
        if isinstance(module, nn.Conv2d):
            module.mask = torch.Tensor(chromosome[i : (module.out_channels + i)])
            # Replace the module in parent
            i += module.out_channels
    return i


def load_model(device, model_path: Optional[str], logger):
    model = models.resnet18()

    model.conv1 = MaskedConv2d(
        3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
    )
    model.layer1[0].conv1 = MaskedConv2d(
        64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
    )
    model.fc = torch.nn.Linear(in_features=512, out_features=10, bias=True)

    chromosome_len = replace_conv_with_custom(model, device)

    if model_path:
        logger.info(f"Loading model from path {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    else:
        logger.info(f"Downloading model from hugging face")
        checkpoint_path = hf_hub_download(
            repo_id="edadaltocg/resnet18_cifar10", filename="pytorch_model.bin"
        )
        model.load_state_dict(torch.load(checkpoint_path), strict=False)
    model = model.to(device)
    return model, chromosome_len
