import numpy as np
import torch
import torch.nn as nn

from .keywords import KEY_CHANNEL, KEY_FILTER, KEY_FILTER_AND_CHANNEL


def group_lasso_by_filter_or_channel(param_group, dimension):
    return torch.sqrt(torch.sum(param_group**2, dim=dimension))


def ssl_loss(model: nn.Module, model_type="resnet", loss_type=KEY_FILTER, lambda_n=1e-5, lambda_c=1e-5):
    ssl_loss = 0

    if loss_type in [KEY_FILTER, KEY_CHANNEL, KEY_FILTER_AND_CHANNEL]:
        # Get prunable layers (excluding final classification layer)
        conv_layers, linear_layers = _get_prunable_layers(model)

        # Process Conv2d layers
        for name, module in conv_layers:
            param = module.weight

            if loss_type == KEY_FILTER:
                # Group LASSO over filters of current layer
                ssl_loss += lambda_n * torch.sum(group_lasso_by_filter_or_channel(param, (1, 2, 3)))
            elif loss_type == KEY_CHANNEL:
                # Group LASSO over channel of current layer
                ssl_loss += lambda_c * torch.sum(group_lasso_by_filter_or_channel(param, (0, 2, 3)))
            elif loss_type == KEY_FILTER_AND_CHANNEL:
                # Group LASSO over filters of current layer
                ssl_loss += lambda_n * torch.sum(group_lasso_by_filter_or_channel(param, (1, 2, 3)))
                # Group LASSO over channel of current layer
                ssl_loss += lambda_c * torch.sum(group_lasso_by_filter_or_channel(param, (0, 2, 3)))

        # Process Linear layers (excluding final layer)
        for name, module in linear_layers:
            param = module.weight

            if loss_type == KEY_FILTER:
                # Treat as filter-wise for linear layers
                ssl_loss += lambda_n * torch.sum(group_lasso_by_filter_or_channel(param, (1,)))
            elif loss_type == KEY_CHANNEL:
                # Treat as channel-wise for linear layers
                ssl_loss += lambda_c * torch.sum(group_lasso_by_filter_or_channel(param, (0,)))
            elif loss_type == KEY_FILTER_AND_CHANNEL:
                # Both filter and channel for linear layers
                ssl_loss += lambda_n * torch.sum(group_lasso_by_filter_or_channel(param, (1,)))
                ssl_loss += lambda_c * torch.sum(group_lasso_by_filter_or_channel(param, (0,)))

    else:
        raise ValueError(f"{model_type} does not supports")

    return ssl_loss


def _get_prunable_layers(model):
    """
    Get lists of prunable layers, excluding the final classification layer.
    Returns (conv_layers, linear_layers) where linear_layers excludes the final layer.
    """
    conv_layers = []
    linear_layers = []

    for name, module in model.named_modules():
        # NOTE: downsample is excluded from resnet18
        if isinstance(module, nn.Conv2d) and "downsample" not in name and "depthwise" not in name:
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
    Computer importance scores and parameter counts for each group (filter/channel).

    When pruning filters/channels, we count not just the parameters in the current layer,
    but also the cascading parameter removal in subsequent layers:
    - Filter pruning: removes filter params + corresponding input channels in next layer
    - Channel pruning: removes channel params + corresponding output channels in previous layer
    """

    # TODO: when getting the next of a linear,
    # Get prunable layers (excluding final classification layer)
    conv_layers, linear_layers = _get_prunable_layers(model)
    all_layers = conv_layers + linear_layers

    def _get_next_layer_params(layer_idx, current_layer, pruned_dim):
        """Calculate parameters removed in the next layer due to current layer pruning"""
        if layer_idx >= len(all_layers) - 1:
            return 0  # No next layer

        next_name, next_layer = all_layers[layer_idx + 1]

        if dim == 0:  # filter pruning - affects input channels of next layer
            if isinstance(next_layer, nn.Conv2d):
                # Each filter removed eliminates: 1 * kernel_h * kernel_w params per output filter
                return (
                    next_layer.weight.data.shape[0] * next_layer.weight.data.shape[2] * next_layer.weight.data.shape[3]
                )
            elif isinstance(next_layer, nn.Linear):
                # Each filter removed eliminates params proportional to the spatial dimensions
                # For the first linear layer after conv, this depends on the spatial size
                return next_layer.weight.data.shape[0]  # * 25 - 5*5 spatial size assumption
        else:  # channel pruning - affects output channels of previous layer
            # This is more complex and typically handled differently in practice
            # For now, we'll use the same logic as filter pruning
            if isinstance(next_layer, nn.Conv2d):
                return (
                    next_layer.weight.data.shape[0] * next_layer.weight.data.shape[2] * next_layer.weight.data.shape[3]
                )
            elif isinstance(next_layer, nn.Linear):
                return next_layer.weight.data.shape[0] * 25

        return 0

    # Process Conv2d layers
    for layer_idx, (name, m) in enumerate(conv_layers):
        print(name)
        size = m.weight.data.shape[dim]
        conv[index : (index + size)] = computer_weight(m.weight, prune_way, dimension)

        # Calculate parameters per group for Conv2d
        weight_shape = m.weight.data.shape
        if dim == 0:  # filter pruning
            # Parameters in current layer: in_channels * kernel_h * kernel_w
            current_layer_params = weight_shape[1] * weight_shape[2] * weight_shape[3]
            # Parameters in next layer that will be removed
            next_layer_params = _get_next_layer_params(layer_idx, m, dim)
            params_per_group = current_layer_params + next_layer_params
        else:  # channel pruning (dim == 1)
            # Parameters in current layer: out_channels * kernel_h * kernel_w
            current_layer_params = weight_shape[0] * weight_shape[2] * weight_shape[3]
            # For channel pruning, we typically don't count next layer params as it's more complex
            # The current layer channel affects all output filters
            params_per_group = current_layer_params

        param_counts[index : (index + size)] = params_per_group
        index += size

    # Process Linear layers (excluding final layer)
    first_linear = True
    for layer_idx, (name, m) in enumerate(linear_layers):
        conv_layer_count = len(conv_layers)
        actual_layer_idx = conv_layer_count + layer_idx

        # NOTE: never used in experiment the channel pruning, so just refer to else branch
        # it only works for fedml CNN_WEB, for other models you need to change it
        if dim == 1 and first_linear:
            first_linear = False
            # reshape as conv
            n_ch = m.weight.data.shape[dim] // (5 * 5)  # TODO: change filter size
            original_weights = m.weight.data.clone()
            m.weight.data = m.weight.data.view(m.out_features, n_ch, 5, 5)

            conv[index : (index + n_ch)] = computer_weight(m.weight, prune_way, dimension)

            # Parameters per channel in reshaped linear layer
            current_layer_params = m.out_features * 5 * 5
            next_layer_params = _get_next_layer_params(actual_layer_idx, m, dim)
            params_per_group = current_layer_params + next_layer_params
            param_counts[index : (index + n_ch)] = params_per_group
            index += n_ch

            m.weight.data = original_weights

        else:
            size = m.weight.data.shape[dim]
            conv[index : (index + size)] = computer_weight(m.weight, prune_way, dimension_fc)

            # Calculate parameters per group for Linear layer
            if dim == 0:  # output features (filter-like pruning)
                current_layer_params = m.weight.data.shape[1]  # input features
                next_layer_params = _get_next_layer_params(actual_layer_idx, m, dim)
                params_per_group = current_layer_params + next_layer_params
            else:  # input features (channel-like pruning)
                # For channel pruning in linear layers, we affect all output features
                params_per_group = m.weight.data.shape[0]  # output features

            param_counts[index : (index + size)] = params_per_group
            index += size

    return conv, param_counts, index


def computer_conv_threshold(model, percent_or_groups, prune_type=KEY_FILTER, prune_way="mean_abs", ceil=False):
    """
    Calculate pruning threshold of Conv layer based on parameter percentage removal or absolute number of groups.
    Excludes the final classification layer from pruning calculations.
    
    Args:
        model: The model to analyze
        percent_or_groups: Either percentage (float 0-1) or absolute number of groups to prune (int)
        prune_type: Type of pruning (KEY_FILTER, KEY_CHANNEL, or KEY_FILTER_AND_CHANNEL)
        prune_way: Method for computing importance scores
        ceil: Whether to round up when computing threshold
        
    Returns:
        tuple: (total_groups, threshold, groups_to_prune)
    """
    total_groups = 0

    if prune_type in [KEY_FILTER, KEY_CHANNEL]:
        dim = 0 if prune_type == KEY_FILTER else 1
        dimension = (1, 2, 3) if prune_type == KEY_FILTER else (0, 2, 3)
        dimension_fc = 1 if prune_type == KEY_FILTER else 0

        # NOTE: Excludes the final classification layer from consideration
        total_groups = computer_total(model, dim)

        conv = torch.zeros(total_groups)
        index = 0
        conv, index = computer_conv(model, conv, index, dim, dimension, dimension_fc, prune_way)

    elif prune_type == KEY_FILTER_AND_CHANNEL:
        # filter_wise (excluding final layer)
        total_groups += computer_total(model, 0)
        # channel_wise (excluding final layer)
        total_groups += computer_total(model, 1)

        conv = torch.zeros(total_groups)
        index = 0
        # filter_wise
        conv, index = computer_conv(model, conv, index, 0, (1, 2, 3), 1, prune_way)
        # channel_wise
        conv, index = computer_conv(model, conv, index, 1, (0, 2, 3), 0, prune_way)
    else:
        raise ValueError(f"{prune_type} does not supports")

    y, i = torch.sort(conv)
    
    # Determine if input is percentage or absolute number
    if isinstance(percent_or_groups, int):
        # Absolute number of groups to prune
        groups_to_prune = min(percent_or_groups, total_groups - 1)  # Ensure at least 1 group remains
        thre_index = groups_to_prune - 1
        percent = groups_to_prune / total_groups
    else:
        # Percentage (float between 0 and 1)
        percent = percent_or_groups
        groups_to_prune = int(total_groups * percent) + 1
        thre_index = groups_to_prune - 1
    
    thre = y[thre_index +  (1 if ceil else 0)]

    print(f"Total groups: {total_groups}")
    if isinstance(percent_or_groups, int):
        print(f"Target groups to prune: {percent_or_groups}")
        print(f"Actual groups to prune: {groups_to_prune}")
    else:
        print(f"Target parameter removal: {percent:.2%}")
        print(f"Groups to remove: {groups_to_prune}")
    print(f"Group removal percentage: {groups_to_prune / total_groups:.2%}")
    print(f"Min importance: {y[0]:.6f}")
    print(f"Max importance: {y[-1]:.6f}")
    print(f"Threshold value: {thre:.6f}")

    return total_groups, thre, groups_to_prune


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
        kernel_size=kernel_size,  # type: ignore
        stride=stride,  # type: ignore
        padding=padding,  # type: ignore
        padding_mode=padding_mode,
        groups=groups,
        dilation=dilation,  # type: ignore
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


def combine_mask(mask_1, mask_2):
    """
    Combine two pruning masks to obtain a mask relative to the original model.

    This function takes the mask from the (n-1)th pruning operation and the mask from
    the nth pruning operation, and combines them to produce a mask that represents
    the cumulative effect of both pruning operations relative to the original model.

    The logic is:
    - mask_1: Represents which filters were kept in the first pruning (relative to original)
    - mask_2: Represents which filters were kept in the second pruning (relative to already pruned model)
    - combined_mask: Represents which original filters are kept after both prunings

    Args:
        mask_1 (dict): Mask dictionary from the first pruning operation (n-1)
        mask_2 (dict): Mask dictionary from the second pruning operation (n)

    Returns:
        dict: Combined mask dictionary with the same structure as input masks,
              but representing the cumulative effect relative to the original model

    Example:
        Original model has 64 filters in a layer.
        After 1st pruning: keeps filters [0,2,4,6,8] (mask_1)
        After 2nd pruning: keeps filters [0,1,3] from the already pruned model (mask_2)
        Combined mask: keeps original filters [0,4,8] from the original 64 filters
    """
    combined_masks = {}

    # Iterate through all layers in mask_1
    for layer_name, mask_1_info in mask_1.items():
        if layer_name not in mask_2:
            # If this layer wasn't pruned in the second operation, just keep mask_1
            combined_masks[layer_name] = mask_1_info.copy()
            continue

        mask_2_info = mask_2[layer_name]

        # Get the original mask from first pruning (relative to original model)
        original_mask_1 = mask_1_info["mask"]  # Boolean tensor indicating kept filters

        # Get the mask from second pruning (relative to already pruned model)
        mask_2_tensor = mask_2_info["mask"]

        # Get indices that were kept in first pruning
        kept_indices_1 = torch.where(original_mask_1 > 0)[0]

        # Get indices that were kept in second pruning (relative to pruned model)
        kept_indices_2 = torch.where(mask_2_tensor > 0)[0]

        # Map the second pruning indices back to original model indices
        # The filters kept in second pruning correspond to positions in the already pruned model
        # So we need to map them back to original indices using kept_indices_1
        final_kept_indices = kept_indices_1[kept_indices_2]

        # Create combined mask relative to original model
        combined_mask_tensor = torch.zeros_like(original_mask_1)
        combined_mask_tensor[final_kept_indices] = 1.0

        # Create combined mask info
        combined_masks[layer_name] = {
            "mask": combined_mask_tensor,
            "layer_type": mask_1_info["layer_type"],
            "original_filters": mask_1_info["original_filters"],  # This should be the same as original
            "pruned_filters": len(final_kept_indices),
            "indices_kept": final_kept_indices.numpy()
            if isinstance(final_kept_indices, torch.Tensor)
            else final_kept_indices,
        }

    # Also include any layers that were only in mask_2 (shouldn't happen in normal flow)
    for layer_name, mask_2_info in mask_2.items():
        if layer_name not in mask_1:
            combined_masks[layer_name] = mask_2_info.copy()

    return combined_masks


def shuffle_mask(names_mask, seed):
    """
    Shuffle the positions of ones in each layer's mask while keeping the same number of filters.

    This function takes a mask dictionary and randomly redistributes the positions of kept filters
    (ones) in each layer's mask. The total number of kept filters per layer remains the same,
    but their positions are randomly shuffled.

    Args:
        names_mask (dict): Dictionary containing mask information for each layer.
                          Each entry should have structure:
                          {
                              "mask": torch.Tensor,  # Boolean mask tensor
                              "layer_type": str,     # Type of layer
                              "original_filters": int,  # Original number of filters
                              "pruned_filters": int,    # Number of kept filters
                              "indices_kept": np.ndarray  # Indices of kept filters
                          }
        seed (int, optional): Random seed for reproducible shuffling. If None,
                             uses numpy's global random state.

    Returns:
        dict: New mask dictionary with shuffled positions but same structure

    Example:
        >>> mask_dict = {
        ...     "conv1": {
        ...         "mask": torch.tensor([1., 0., 1., 0., 1.]),
        ...         "layer_type": "conv",
        ...         "original_filters": 5,
        ...         "pruned_filters": 3,
        ...         "indices_kept": np.array([0, 2, 4])
        ...     }
        ... }
        >>> shuffled = shuffle_mask(mask_dict, seed=42)
        >>> # shuffled["conv1"]["mask"] might be: torch.tensor([0., 1., 0., 1., 1.])
        >>> # shuffled["conv1"]["indices_kept"] might be: np.array([1, 3, 4])
    """
    # Create random number generator with seed
    rng = np.random.default_rng(seed)

    shuffled_masks = {}

    for layer_name, mask_info in names_mask.items():
        # Get the original mask
        original_mask = mask_info["mask"].cpu()

        # Get total number of positions and number of ones
        total_positions = len(original_mask)
        num_ones = int(torch.sum(original_mask).item())

        # Create new shuffled mask with same number of ones
        new_mask = torch.zeros_like(original_mask)

        # Randomly select positions for the ones using the seeded generator
        all_positions = np.arange(total_positions)
        shuffled_positions = rng.choice(all_positions, size=num_ones, replace=False)
        shuffled_positions = np.sort(shuffled_positions)  # Sort for consistency

        # Set the selected positions to 1
        new_mask[torch.from_numpy(shuffled_positions)] = 1.0

        # Create new mask info with updated fields
        shuffled_masks[layer_name] = {
            "mask": new_mask,
            "layer_type": mask_info["layer_type"],
            "original_filters": mask_info["original_filters"],
            "pruned_filters": mask_info["pruned_filters"],  # This should remain the same
            "indices_kept": shuffled_positions,
        }

        # Verify that the number of kept filters is the same
        assert num_ones == mask_info["pruned_filters"] == len(shuffled_positions) == int(torch.sum(new_mask).item()), (
            f"Mismatch in pruned_filters for layer {layer_name}: "
            f"expected {mask_info['pruned_filters']}, got {len(shuffled_positions)}"
        )

    return shuffled_masks
