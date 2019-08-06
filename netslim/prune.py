import copy
from functools import partial
import torch
import torch.nn as nn
from .graph_parser import get_pruning_layers

OUT_CHANNEL_DIM = 0
IN_CHANNEL_DIM = 1
WEIGHT_POSTFIX = ".weight"
BIAS_POSTFIX = ".bias"
MIN_CHANNELS = 3


def group_weight_names(weight_names):
    grouped_names = {}
    for weight_name in weight_names:
        group_name = '.'.join(weight_name.split('.')[:-1])
        if group_name not in grouped_names:
            grouped_names[group_name] = [weight_name, ]
        else:
            grouped_names[group_name].append(weight_name)
    return grouped_names


def liu2017(weights, prune_ratio, prec_layers, succ_layers, per_layer_normalization=False):
    """default pruning method as described in:
            Zhuang Liu et.al., "Learning Efficient Convolutional Networks through Network Slimming", in ICCV 2017"

    Arguments:
        weights (OrderedDict): unpruned model weights
        prune_ratio (float): ratio of be pruned channels to total channels
        prec_layers (dict): mapping from BN names to preceding convs/linears
        succ_layers (dict): mapping from BN names to succeeding convs/linears
        per_layer_normalization (bool): if do normalization by layer

    Returns:
        pruned_weights (OrderedDict): pruned model weights
    """

    # find all scale weights in BN layers
    scale_weights = []
    norm_layer_names = list(set(succ_layers) & set(prec_layers))
    for norm_layer_name in norm_layer_names:
        norm_weight_name = norm_layer_name + WEIGHT_POSTFIX
        weight = weights[norm_weight_name]
        if per_layer_normalization:
            scale_weights.extend([(_.abs()/weight.sum()).item() for _ in list(weight)])
        else:
            scale_weights.extend([_.abs().item() for _ in list(weight)])

    # find threshold for pruning
    scale_weights.sort()
    prune_th_index = int(float(len(scale_weights)) * prune_ratio + 0.5)
    prune_th = scale_weights[prune_th_index]

    # unpruned_norm_layer_names = list(set(succ_layers) ^ set(prec_layers))
    grouped_weight_names = group_weight_names(weights.keys())
    for norm_layer_name in norm_layer_names:
        norm_weight_name = norm_layer_name + WEIGHT_POSTFIX
        scale_weight = weights[norm_weight_name].abs()
        if per_layer_normalization:
            scale_weight = scale_weight / scale_weight.sum()
        prune_mask = scale_weight > prune_th
        if prune_mask.sum().item() == scale_weight.size(0):
            continue

        # in case not to prune the whole layer
        if prune_mask.sum() < MIN_CHANNELS:
            scale_weight_list = [_.abs().item() for _ in list(scale_weight)]
            scale_weight_list.sort(reverse=True)
            prune_mask = scale_weight >= scale_weight_list[MIN_CHANNELS-1]

        prune_indices = torch.nonzero(prune_mask).flatten()
        
        # 1. prune source normalization layer
        for weight_name in grouped_weight_names[norm_layer_name]:
            weights[weight_name] = weights[weight_name].masked_select(prune_mask)

        # 2. prune target succeeding conv/linear/... layers
        for prune_layer_name in succ_layers[norm_layer_name]:
            for weight_name in grouped_weight_names[prune_layer_name]:
                if weight_name.endswith(WEIGHT_POSTFIX):
                    weights[weight_name] = weights[weight_name].index_select(IN_CHANNEL_DIM, prune_indices)

        # 3. prune target preceding conv/linear/... layers
        for prune_layer_name in prec_layers[norm_layer_name]:
            for weight_name in grouped_weight_names[prune_layer_name]:
                if weight_name.endswith(WEIGHT_POSTFIX):
                    weights[weight_name] = weights[weight_name].index_select(OUT_CHANNEL_DIM, prune_indices)
                elif weight_name.endswith(BIAS_POSTFIX):
                    weights[weight_name] = weights[weight_name].index_select(0, prune_indices)

    return weights


liu2017_normalized_by_layer = partial(liu2017, per_layer_normalization=True)


def _dirty_fix(module, param_name, pruned_shape):
    module_param = getattr(module, param_name)

    # identify the dimension to prune
    pruned_dim = 0
    for original_size, pruned_size in zip(module_param.shape, pruned_shape):
        if original_size != pruned_size:
            keep_indices = torch.LongTensor(range(pruned_size)).to(module_param.data.device)
            module_param.data = module_param.data.index_select(pruned_dim, keep_indices)

            # modify number of features/channels
            if param_name == "weight":
                if isinstance(module, nn.modules.batchnorm._BatchNorm) or \
                        isinstance(module, nn.modules.instancenorm._InstanceNorm) or \
                        isinstance(module, nn.GroupNorm):
                    module.num_features = pruned_size
                elif isinstance(module, nn.modules.conv._ConvNd):
                    if pruned_dim == OUT_CHANNEL_DIM:
                        module.out_channels = pruned_size
                    elif pruned_dim == IN_CHANNEL_DIM:
                        module.in_channels = pruned_size
                elif isinstance(module, nn.Linear):
                    if pruned_dim == OUT_CHANNEL_DIM:
                        module.out_features = pruned_size
                    elif pruned_dim == IN_CHANNEL_DIM:
                        module.in_features = pruned_size
                else:
                    pass
        pruned_dim += 1


def load_pruned_model(model, pruned_weights, prefix='', load_pruned_weights=True, inplace=True):
    """load pruned weights to a unpruned model instance

    Arguments:
        model (pytorch model): the model instance
        pruned_weights (OrderedDict): pruned weights
        prefix (string optional): prefix (if has) of pruned weights
        load_pruned_weights (bool optional): load pruned weights to model according to the ICLR 2019 paper:
            "Rethinking the Value of Network Pruning", without finetuning, the model may achieve comparable or even
            better results
        inplace (bool, optional): if return a copy of the model

    Returns:
        a model instance with pruned structure (and weights if load_pruned_weights==True)
    """
    model_weight_names = model.state_dict().keys()
    pruned_weight_names = pruned_weights.keys()

    # check if module names match
    assert set([prefix + _ for _ in model_weight_names]) == set(pruned_weight_names)

    # inplace or return a new copy
    if not inplace:
        pruned_model = copy.deepcopy(model)
    else:
        pruned_model = model

    # update modules with mis-matched weight
    model_weights = model.state_dict()
    for model_weight_name in model_weight_names:
        if model_weights[model_weight_name].shape != pruned_weights[prefix + model_weight_name].shape:
            *container_names, module_name, param_name = model_weight_name.split('.')
            container = model
            for container_name in container_names:
                container = container._modules[container_name]
            module = container._modules[module_name]
            _dirty_fix(module, param_name, pruned_weights[prefix + model_weight_name].shape)
    if load_pruned_weights:
        pruned_model.load_state_dict({k: v for k, v in pruned_weights.items()})
    return pruned_model


def prune(model, input_shape, prune_ratio, prune_method=liu2017):
    """prune a model

    Arguments:
        model (pytorch model): the model instance
        input_shape (tuple): shape of the input tensor
        prune_ratio (float): ratio of be pruned channels to total channels
        prune_method (method): algorithm to prune weights

    Returns:
        a model instance with pruned structure (and weights if load_pruned_weights==True)

    Pipeline:
        1. generate mapping from tensors connected to BNs by parsing torch script traced graph
        2. identify corresponding BN and conv/linear like:
            conv/linear --> ... --> BN --> ... --> conv/linear
                                     |
                                    ...
                                     | --> relu --> ... --> conv/linear
                                    ...
                                     | --> ... --> maxpool --> ... --> conv/linear
            , where ... represents per channel operations. all the floating nodes must be conv/linear
        3. prune the weights of BN and connected conv/linear
        4. load weights to a unpruned model with pruned weights
    """
    # convert to CPU for simplicity
    src_device = next(model.parameters()).device
    model = model.cpu()

    # parse & generate mappings to BN layers
    prec_layers, succ_layers = get_pruning_layers(model, input_shape)

    # prune weights
    pruned_weights = prune_method(model.state_dict(), prune_ratio, prec_layers, succ_layers)

    # prune model according to pruned weights
    pruned_model = load_pruned_model(model, pruned_weights)

    return pruned_model.to(src_device)
