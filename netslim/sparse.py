import torch
import torch.nn as nn


def update_bn(model, s=1e-4):
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm) or \
            isinstance(m, nn.modules.instancenorm._InstanceNorm) or \
            isinstance(m, nn.GroupNorm):
            m.weight.grad.data.add_(s * torch.sign(m.weight.data))


def update_bn_by_names(model, norm_layer_names, s=1e-4):
    for norm_layer_name in norm_layer_names:
        *container_names, module_name = norm_layer_name.split('.')
        container = model
        for container_name in container_names:
            container = container._modules[container_name]
        m = container._modules[module_name]
        m.weight.grad.data.add_(s * torch.sign(m.weight.data))
