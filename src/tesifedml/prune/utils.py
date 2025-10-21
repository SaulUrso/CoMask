import torch
import torch.nn as nn

from .keywords import KEY_CHANNEL, KEY_FILTER, KEY_FILTER_AND_CHANNEL


def group_lasso_by_filter_or_channel(param_group, dimension):
    return torch.sqrt(torch.sum(param_group**2, dim=dimension))


def _get_prunable_layers(model):
    """
    Get lists of prunable layers, excluding the final classification layer.
    Returns (conv_layers, linear_layers) where linear_layers excludes the final layer.
    """
    conv_layers = []
    linear_layers = []

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            conv_layers.append((name, module))
        elif isinstance(module, nn.Linear):
            linear_layers.append((name, module))

    # Exclude the final linear layer (assumed to be the classification layer)
    if linear_layers:
        linear_layers = linear_layers[:-1]

    return conv_layers, linear_layers


def computer_total(model, dim):
    assert isinstance(model, nn.Module)
    total = 0

    # Get prunable layers (excluding final classification layer)
    conv_layers, linear_layers = _get_prunable_layers(model)

    # Count the specified dimension lengths of all Conv layers
    for name, m in conv_layers:
        total += m.weight.data.shape[dim]

    # Count linear layers (excluding final layer)
    first_linear = True
    for name, m in linear_layers:
        # if doing channel wise, the first linear layer has to be considered
        # as convolutional in order to be used
        if dim == 1 and first_linear:
            first_linear = False
            n_cols = m.weight.data.shape[dim]
            n_ch = n_cols / (5 * 5)  # TODO: change if filter dimension changes
            assert n_ch == int(n_ch)
            total += int(n_ch)
        else:
            total += m.weight.data.shape[dim]

    return total


def round_to_multiple_of(val, divisor):
    """Asymmetric rounding to make `val` divisible by `divisor`. With default
    bias, will round up, i.e. (83, 8) -> 88, but (84, 8) -> 88."""
    new_val = max(divisor, int(val + divisor / 2) // divisor * divisor)
    return new_val if new_val >= val else new_val + divisor


def computer_weight(weight, prune_way, dimension):
    """Computes group importance alongside the specified dimension for neural network layer weights.

    This function calculates the importance score for groups of weights (e.g., filters or channels)
    in a neural network layer based on different pruning strategies. The importance is computed
    by aggregating weights across specified dimensions.

    Args:
        weight (torch.Tensor): The weight tensor of a neural network layer (e.g., Conv2d weights
            with shape [out_channels, in_channels, kernel_height, kernel_width]).
        prune_way (str): The method used to compute importance. Options:
            - 'group_lasso': L2 norm across specified dimensions
            - 'mean_abs': Mean of absolute values across specified dimensions
            - 'mean': Mean across specified dimensions
            - 'sum_abs': Sum of absolute values across specified dimensions
            - 'sum': Sum across specified dimensions
        dimension (tuple): Tuple of dimensions to aggregate over. For filter pruning use (1,2,3),
            for channel pruning use (0,2,3).

    Raises:
        ValueError: If prune_way is not one of the supported methods.

    Returns:
        torch.Tensor: A 1D tensor containing importance scores for each group along the non-aggregated
            dimension. The length equals the size of the dimension not included in the aggregation.

    Example:
        >>> import torch
        >>> # Conv2d layer with 6 output filters, 3 input channels, 3x3 kernel
        >>> conv_weight = torch.randn(6, 3, 3, 3)
        >>>
        >>> # Compute filter importance (aggregate over input channels and spatial dims)
        >>> filter_importance = computer_weight(conv_weight, 'mean_abs', (1, 2, 3))
        >>> print(filter_importance.shape)  # torch.Size([6])
        >>> print(filter_importance)  # tensor([0.8234, 0.9123, 0.7456, 0.8901, 0.7234, 0.8567])
        >>>
        >>> # Compute channel importance (aggregate over output filters and spatial dims)
        >>> channel_importance = computer_weight(conv_weight, 'mean_abs', (0, 2, 3))
        >>> print(channel_importance.shape)  # torch.Size([3])
        >>> print(channel_importance)  # tensor([0.8123, 0.7890, 0.8345])
    """
    if prune_way == "group_lasso":
        return group_lasso_by_filter_or_channel(weight.data, dimension)
    elif prune_way == "mean_abs":
        return torch.mean(weight.data.abs(), dim=dimension)
    elif prune_way == "mean":
        return torch.mean(weight.data, dim=dimension)
    elif prune_way == "sum_abs":
        return torch.sum(weight.data.abs(), dim=dimension)
    elif prune_way == "sum":
        return torch.sum(weight.data, dim=dimension)
    else:
        raise ValueError(f"{prune_way} does not exists")


def computer_conv(model, conv, index, dim, dimension, dimension_fc, prune_way):
    # Get prunable layers (excluding final classification layer)
    conv_layers, linear_layers = _get_prunable_layers(model)

    # Process Conv2d layers
    for name, m in conv_layers:
        size = m.weight.data.shape[dim]
        conv[index : (index + size)] = computer_weight(m.weight, prune_way, dimension)
        index += size

    # Process Linear layers (excluding final layer)
    first_linear = True
    for name, m in linear_layers:
        if dim == 1 and first_linear:
            first_linear = False
            # reshape as conv
            n_ch = m.weight.data.shape[dim] // (5 * 5)  # TODO: change filter size
            original_weights = m.weight.data.clone()
            m.weight.data = m.weight.data.view(m.out_features, n_ch, 5, 5)

            conv[index : (index + n_ch)] = computer_weight(m.weight, prune_way, dimension)
            index += n_ch

            m.weight.data = original_weights

        else:
            size = m.weight.data.shape[dim]
            conv[index : (index + size)] = computer_weight(m.weight, prune_way, dimension_fc)
            index += size

    return conv, index


def computer_conv_with_params(model, conv, param_counts, index, dim, dimension, dimension_fc, prune_way):
    """
    Computer importance scores and parameter counts for each group (filter/channel)
    """
    # Get prunable layers (excluding final classification layer)
    conv_layers, linear_layers = _get_prunable_layers(model)

    # Process Conv2d layers
    for name, m in conv_layers:
        size = m.weight.data.shape[dim]
        conv[index : (index + size)] = computer_weight(m.weight, prune_way, dimension)

        # Calculate parameters per group for Conv2d
        weight_shape = m.weight.data.shape
        if dim == 0:  # filter pruning
            # Each filter: in_channels * kernel_h * kernel_w
            params_per_group = weight_shape[1] * weight_shape[2] * weight_shape[3]
        else:  # channel pruning (dim == 1)
            # Each channel: out_channels * kernel_h * kernel_w
            params_per_group = weight_shape[0] * weight_shape[2] * weight_shape[3]

        param_counts[index : (index + size)] = params_per_group
        index += size

    # Process Linear layers (excluding final layer)
    first_linear = True
    for name, m in linear_layers:
        if dim == 1 and first_linear:
            first_linear = False
            # reshape as conv
            n_ch = m.weight.data.shape[dim] // (5 * 5)  # TODO: change filter size
            original_weights = m.weight.data.clone()
            m.weight.data = m.weight.data.view(m.out_features, n_ch, 5, 5)

            conv[index : (index + n_ch)] = computer_weight(m.weight, prune_way, dimension)

            # Parameters per channel in reshaped linear layer
            params_per_group = m.out_features * 5 * 5
            param_counts[index : (index + n_ch)] = params_per_group
            index += n_ch

            m.weight.data = original_weights

        else:
            size = m.weight.data.shape[dim]
            conv[index : (index + size)] = computer_weight(m.weight, prune_way, dimension_fc)

            # Calculate parameters per group for Linear layer
            if dim == 0:  # output features (filter-like pruning)
                params_per_group = m.weight.data.shape[1]  # input features
            else:  # input features (channel-like pruning)
                params_per_group = m.weight.data.shape[0]  # output features

            param_counts[index : (index + size)] = params_per_group
            index += size

    return conv, param_counts, index


def computer_conv_threshold(model, percent, prune_type=KEY_FILTER, prune_way="mean_abs"):
    """
    Calculate pruning threshold of Conv layer based on parameter percentage removal.
    Excludes the final classification layer from pruning calculations.
    """
    total_groups = 0

    if prune_type in [KEY_FILTER, KEY_CHANNEL]:
        dim = 0 if prune_type == KEY_FILTER else 1
        dimension = (1, 2, 3) if prune_type == KEY_FILTER else (0, 2, 3)
        dimension_fc = 1 if prune_type == KEY_FILTER else 0

        # NOTE: Excludes the final classification layer from consideration
        total_groups = computer_total(model, dim)

        conv = torch.zeros(total_groups)
        param_counts = torch.zeros(total_groups)  # Store parameter count per group
        index = 0
        conv, param_counts, index = computer_conv_with_params(
            model, conv, param_counts, index, dim, dimension, dimension_fc, prune_way
        )

    elif prune_type == KEY_FILTER_AND_CHANNEL:
        # filter_wise (excluding final layer)
        total_groups += computer_total(model, 0)
        # channel_wise (excluding final layer)
        total_groups += computer_total(model, 1)

        conv = torch.zeros(total_groups)
        param_counts = torch.zeros(total_groups)
        index = 0
        # filter_wise
        conv, param_counts, index = computer_conv_with_params(
            model, conv, param_counts, index, 0, (1, 2, 3), 1, prune_way
        )
        # channel_wise
        conv, param_counts, index = computer_conv_with_params(
            model, conv, param_counts, index, 1, (0, 2, 3), 0, prune_way
        )
    else:
        raise ValueError(f"{prune_type} does not supports")

    # Sort by importance scores (ascending order - least important first)
    y, i = torch.sort(conv)

    # Get corresponding parameter counts in the same order
    sorted_param_counts = param_counts[i]

    # Calculate cumulative parameter removal
    cumulative_params = torch.cumsum(sorted_param_counts, dim=0)
    total_params = torch.sum(param_counts)
    cumulative_percent = cumulative_params / total_params

    # Find threshold where cumulative percentage reaches target
    target_param_removal = percent
    threshold_indices = torch.where(cumulative_percent >= target_param_removal)[0]

    if len(threshold_indices) == 0:
        # If target percentage cannot be reached, use all groups
        thre_index = total_groups - 1
        thre = y[-1]
    else:
        thre_index = threshold_indices[0].item()
        thre = y[thre_index]

    # Add debugging information
    actual_param_removal = cumulative_percent[thre_index].item() if thre_index < len(cumulative_percent) else 1.0
    # print(f"Total groups: {total_groups}")
    # print(f"Total parameters: {total_params}")
    # print(f"Target parameter removal: {percent:.2%}")
    # print(f"Actual parameter removal: {actual_param_removal:.2%}")
    # print(f"Groups to remove: {thre_index + 1}")
    # print(f"Group removal percentage: {(thre_index + 1) / total_groups:.2%}")
    # print(f"Min importance: {y[0]:.6f}")
    # print(f"Max importance: {y[-1]:.6f}")
    # print(f"Threshold value: {thre:.6f}")

    return total_groups, param_counts.sum(), thre


def create_conv2d(old_conv2d, in_channels, out_filters, old_groups=None):
    assert isinstance(old_conv2d, nn.Conv2d)
    kernel_size = old_conv2d.kernel_size
    stride = old_conv2d.stride
    padding = old_conv2d.padding
    padding_mode = old_conv2d.padding_mode
    groups = old_groups if old_groups is not None else old_conv2d.groups
    dilation = old_conv2d.dilation
    bias = old_conv2d.bias is not None

    new_conv2d = nn.Conv2d(
        in_channels,
        out_filters,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        padding_mode=padding_mode,
        groups=groups,
        dilation=dilation,
        bias=bias,
    )
    return new_conv2d


def create_batchnorm2d(old_batchnorm2d, in_channels):
    assert isinstance(old_batchnorm2d, nn.BatchNorm2d), f"Got {type(old_batchnorm2d)}"

    eps = old_batchnorm2d.eps
    momentum = old_batchnorm2d.momentum

    return nn.BatchNorm2d(in_channels, eps=eps, momentum=momentum)


def create_linear(old_linear, in_channels, out_channels=None):
    assert isinstance(old_linear, nn.Linear)

    out_channels = old_linear.out_features if out_channels is None else out_channels
    bias = old_linear.bias is not None

    return nn.Linear(in_channels, out_channels, bias=bias), out_channels


# refert to: [Pytorch替换model对象任意层的方法](https://zhuanlan.zhihu.com/p/356273702)
# The core function refers to the implementation of torch.quantification.fuse_modules()
def _set_module(model, submodule_key, module):
    tokens = submodule_key.split(".")
    sub_tokens = tokens[:-1]
    cur_mod = model
    for s in sub_tokens:
        cur_mod = getattr(cur_mod, s)
    setattr(cur_mod, tokens[-1], module)


def set_module_list(model, name_list, module_list, new_module_list):
    for name, module, new_module in zip(name_list, module_list, new_module_list):
        # print(name, module, new_module)
        _set_module(model, name, new_module)


def count_parameters(model):
    """Count total number of parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
