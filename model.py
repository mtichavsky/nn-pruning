import torchvision
import torch
import torchvision.models as models
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import SGD
from huggingface_hub import hf_hub_download


class MaskedConv2d(nn.Conv2d):
    """Convolutional layer that supports masking for individual filters"""

    def __init__(self, *args, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)
        self._mask = torch.ones_like(self.weight)

    def forward(self, input):
        weight = self.weight * self.mask
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
        self._mask = m.view(-1, 1, 1, 1).float().expand_as(self.weight)


# TODO cleanup
def replace_conv_with_custom(model):
    for name, module in model.named_children():
        # Recursively replace in submodules
        replace_conv_with_custom(module)

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
            )
            # Copy weights and bias if necessary
            # Useless, model was not loaded, but hey, maybe keep it jus tin case
            new_module.weight = module.weight
            if module.bias is not None:
                new_module.bias = module.bias

            # Replace the module in parent
            setattr(model, name, new_module)


def load_model(device):
    checkpoint_path = hf_hub_download(
        repo_id="edadaltocg/resnet18_cifar10", filename="pytorch_model.bin"
    )
    model = models.resnet18()

    model.conv1 = MaskedConv2d(
        3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
    )
    model.layer1[0].conv1 = MaskedConv2d(
        64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
    )
    model.fc = torch.nn.Linear(in_features=512, out_features=10, bias=True)

    replace_conv_with_custom(model)
    model.load_state_dict(torch.load(checkpoint_path))
    model = model.to(device)
    return model
