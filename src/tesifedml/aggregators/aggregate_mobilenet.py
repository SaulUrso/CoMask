from typing import Dict, List

import torch
import torch.nn as nn

from ..models.mobilenet import BasicConv2d, DepthSeperabelConv2d, MobileNet


def aggregate_mobilenet_with_masks(
    models: List[nn.Module],
    masks: List[Dict],
    counters: List[int],
    class_num: int = 62,
) -> nn.Module:
    """
    Aggregate multiple pruned MobileNet models using their pruning masks.

    This function performs weighted aggregation of pruned models, where each parameter
    is the weighted sum (weighted by counters) of parameters whose mask is not zero.

    Args:
        models: List of pruned MobileNet models
        masks: List of mask dictionaries for each model. Each mask dict contains layer names as keys,
               mapping to dict with:
               - "mask": torch.Tensor - Boolean mask indicating kept filters
               - "indices_kept": np.ndarray - Indices of kept filters in original model
               - "layer_type": str - Type of layer (e.g., "conv")
               - "original_filters": int - Number of filters in original unpruned model
               - "pruned_filters": int - Number of filters kept after pruning
        counters: List of weights for weighted averaging (e.g., number of data samples)
        width_multiplier: Width multiplier for the model architecture
        class_num: Number of output classes

    Returns:
        Aggregated MobileNet model with structure matching the union of all kept filters

    Example:
        >>> model1 = prune_model(MobileNet(), rate=0.3)
        >>> model2 = prune_model(MobileNet(), rate=0.5)
        >>> aggregated = aggregate_mobilenet_with_masks(
        ...     [model1, model2],
        ...     [mask1, mask2],
        ...     [100, 50]
        ... )
    """
    assert len(models) == len(masks) == len(counters), "Models, masks, and counters must have same length"
    assert len(models) > 0, "Must provide at least one model"

    # Normalize counters to sum to 1 (for weighted averaging)
    total_count = sum(counters)
    weights = [c / total_count for c in counters]

    # Create a new unpruned model as template for aggregation
    aggregated_model = MobileNet(class_num=class_num)

    # Iterate over all named modules in the aggregated model
    for name, module in aggregated_model.named_modules():
        if name == "":  # Skip root module
            continue

        # Handle BasicConv2d blocks
        if isinstance(module, BasicConv2d):
            mask_key = f"{name}.conv"
            _aggregate_basic_conv2d(module, models, masks, weights, mask_key, name)

        # Handle DepthSeparableConv2d blocks
        elif isinstance(module, DepthSeperabelConv2d):
            mask_key = f"{name}.pointwise.0"
            _aggregate_depth_separable_conv2d(module, models, masks, weights, mask_key, name)

        # Handle linear layer
        elif isinstance(module, nn.Linear) and name == "fc":
            _aggregate_linear_layer(module, models, masks, weights)

    return aggregated_model


def _aggregate_basic_conv2d(aggregated_block, models, masks, weights, mask_key, block_name):
    """Aggregate a BasicConv2d block from multiple models."""
    aggregated_conv = aggregated_block.conv
    aggregated_bn = aggregated_block.bn

    original_out_channels = aggregated_conv.out_channels
    original_in_channels = aggregated_conv.in_channels

    # Initialize accumulators
    conv_weight_sum = torch.zeros_like(aggregated_conv.weight.data)
    conv_weight_count = torch.zeros(original_out_channels, original_in_channels)
    conv_bias_sum = torch.zeros_like(aggregated_conv.bias.data) if aggregated_conv.bias is not None else None
    conv_bias_count = torch.zeros(original_out_channels)

    bn_weight_sum = torch.zeros_like(aggregated_bn.weight.data)
    bn_bias_sum = torch.zeros_like(aggregated_bn.bias.data)
    bn_running_mean_sum = torch.zeros_like(aggregated_bn.running_mean)
    bn_running_var_sum = torch.zeros_like(aggregated_bn.running_var)
    bn_count = torch.zeros(original_out_channels)

    # Aggregate from each model
    for model, mask_dict, weight in zip(models, masks, weights):
        if mask_key not in mask_dict:
            raise ValueError(f"Mask key '{mask_key}' not found in mask_dict for client with weight {weight}.")

        kept_indices = mask_dict[mask_key]["indices_kept"]

        # Get corresponding block from pruned model
        model_block = _get_nested_attr(model, block_name)
        model_conv = model_block.conv
        model_bn = model_block.bn

        # Get previous layer's kept indices for input channel mapping
        prev_kept_indices = _get_previous_layer_indices(block_name, mask_dict)

        # Aggregate each kept filter
        for pruned_idx, original_idx in enumerate(kept_indices):
            # Aggregate Conv2d weights
            if prev_kept_indices is None:
                # First layer - all input channels present
                conv_weight_sum[original_idx, :, :, :] += weight * model_conv.weight.data[pruned_idx, :, :, :]
                conv_weight_count[original_idx, :] += weight
            else:
                # Map input channels using previous layer's mask
                for pruned_in_idx, original_in_idx in enumerate(prev_kept_indices):
                    conv_weight_sum[original_idx, original_in_idx, :, :] += (
                        weight * model_conv.weight.data[pruned_idx, pruned_in_idx, :, :]
                    )
                    conv_weight_count[original_idx, original_in_idx] += weight

            # Aggregate Conv2d bias
            if conv_bias_sum is not None and model_conv.bias is not None:
                conv_bias_sum[original_idx] += weight * model_conv.bias.data[pruned_idx]
                conv_bias_count[original_idx] += weight

            # Aggregate BatchNorm parameters
            bn_weight_sum[original_idx] += weight * model_bn.weight.data[pruned_idx]
            bn_bias_sum[original_idx] += weight * model_bn.bias.data[pruned_idx]
            bn_running_mean_sum[original_idx] += weight * model_bn.running_mean[pruned_idx]
            bn_running_var_sum[original_idx] += weight * model_bn.running_var[pruned_idx]
            bn_count[original_idx] += weight

    # Apply weighted averages
    mask = conv_weight_count > 0
    aggregated_conv.weight.data[mask] = conv_weight_sum[mask] / conv_weight_count[mask].unsqueeze(-1).unsqueeze(-1)

    if conv_bias_sum is not None:
        bias_mask = conv_bias_count > 0
        aggregated_conv.bias.data[bias_mask] = conv_bias_sum[bias_mask] / conv_bias_count[bias_mask]

    bn_mask = bn_count > 0
    aggregated_bn.weight.data[bn_mask] = bn_weight_sum[bn_mask] / bn_count[bn_mask]
    aggregated_bn.bias.data[bn_mask] = bn_bias_sum[bn_mask] / bn_count[bn_mask]
    aggregated_bn.running_mean[bn_mask] = bn_running_mean_sum[bn_mask] / bn_count[bn_mask]
    aggregated_bn.running_var[bn_mask] = bn_running_var_sum[bn_mask] / bn_count[bn_mask]


def _aggregate_depth_separable_conv2d(aggregated_block, models, masks, weights, mask_key, block_name):
    """Aggregate a DepthSeparableConv2d block from multiple models."""
    # The depthwise conv adapts to input channels, the pointwise conv is what gets pruned
    aggregated_depthwise_conv = aggregated_block.depthwise[0]
    aggregated_depthwise_bn = aggregated_block.depthwise[1]
    aggregated_pointwise_conv = aggregated_block.pointwise[0]
    aggregated_pointwise_bn = aggregated_block.pointwise[1]

    # Get dimensions
    original_in_channels = aggregated_depthwise_conv.in_channels  # == out_channels for depthwise
    original_out_channels = aggregated_pointwise_conv.out_channels

    # Initialize accumulators for depthwise conv
    # Depthwise conv shape: [in_channels, 1, kernel_h, kernel_w] (groups=in_channels)
    depthwise_weight_sum = torch.zeros_like(aggregated_depthwise_conv.weight.data)
    depthwise_weight_count = torch.zeros(original_in_channels)
    depthwise_bias_sum = (
        torch.zeros_like(aggregated_depthwise_conv.bias.data) if aggregated_depthwise_conv.bias is not None else None
    )
    depthwise_bias_count = torch.zeros(original_in_channels)

    depthwise_bn_weight_sum = torch.zeros_like(aggregated_depthwise_bn.weight.data)
    depthwise_bn_bias_sum = torch.zeros_like(aggregated_depthwise_bn.bias.data)
    depthwise_bn_running_mean_sum = torch.zeros_like(aggregated_depthwise_bn.running_mean)
    depthwise_bn_running_var_sum = torch.zeros_like(aggregated_depthwise_bn.running_var)
    depthwise_bn_count = torch.zeros(original_in_channels)

    # Initialize accumulators for pointwise conv
    pointwise_weight_sum = torch.zeros_like(aggregated_pointwise_conv.weight.data)
    pointwise_weight_count = torch.zeros(original_out_channels, original_in_channels)
    pointwise_bias_sum = (
        torch.zeros_like(aggregated_pointwise_conv.bias.data) if aggregated_pointwise_conv.bias is not None else None
    )
    pointwise_bias_count = torch.zeros(original_out_channels)

    pointwise_bn_weight_sum = torch.zeros_like(aggregated_pointwise_bn.weight.data)
    pointwise_bn_bias_sum = torch.zeros_like(aggregated_pointwise_bn.bias.data)
    pointwise_bn_running_mean_sum = torch.zeros_like(aggregated_pointwise_bn.running_mean)
    pointwise_bn_running_var_sum = torch.zeros_like(aggregated_pointwise_bn.running_var)
    pointwise_bn_count = torch.zeros(original_out_channels)

    # Aggregate from each model
    for model, mask_dict, weight in zip(models, masks, weights):
        if mask_key not in mask_dict:
            raise ValueError(f"Mask key {mask_key} not found for client with {weight} samples.")

        kept_indices = mask_dict[mask_key]["indices_kept"]

        # Get corresponding block from pruned model
        model_block = _get_nested_attr(model, block_name)
        model_depthwise_conv = model_block.depthwise[0]
        model_depthwise_bn = model_block.depthwise[1]
        model_pointwise_conv = model_block.pointwise[0]
        model_pointwise_bn = model_block.pointwise[1]

        # Get previous layer's kept indices for input channel mapping
        prev_kept_indices = _get_previous_layer_indices(block_name, mask_dict)

        # Aggregate depthwise conv (which adapts based on input channels from previous layer)
        if prev_kept_indices is not None:
            for pruned_in_idx, original_in_idx in enumerate(prev_kept_indices):
                # Depthwise conv: each input channel has its own filter
                depthwise_weight_sum[original_in_idx, :, :, :] += (
                    weight * model_depthwise_conv.weight.data[pruned_in_idx, :, :, :]
                )
                depthwise_weight_count[original_in_idx] += weight

                if depthwise_bias_sum is not None and model_depthwise_conv.bias is not None:
                    depthwise_bias_sum[original_in_idx] += weight * model_depthwise_conv.bias.data[pruned_in_idx]
                    depthwise_bias_count[original_in_idx] += weight

                # Depthwise BatchNorm
                depthwise_bn_weight_sum[original_in_idx] += weight * model_depthwise_bn.weight.data[pruned_in_idx]
                depthwise_bn_bias_sum[original_in_idx] += weight * model_depthwise_bn.bias.data[pruned_in_idx]
                depthwise_bn_running_mean_sum[original_in_idx] += (
                    weight * model_depthwise_bn.running_mean[pruned_in_idx]
                )
                depthwise_bn_running_var_sum[original_in_idx] += weight * model_depthwise_bn.running_var[pruned_in_idx]
                depthwise_bn_count[original_in_idx] += weight
        else:
            raise ValueError(f"prev_kept_indices is {None}, but layer {block_name} is not the first one.saul")
            # First layer - all input channels present
            # This shouldn't happen as stem.0 is BasicConv2d
            for original_in_idx in range(original_in_channels):
                depthwise_weight_sum[original_in_idx, :, :, :] += (
                    weight * model_depthwise_conv.weight.data[original_in_idx, :, :, :]
                )
                depthwise_weight_count[original_in_idx] += weight

                if depthwise_bias_sum is not None and model_depthwise_conv.bias is not None:
                    depthwise_bias_sum[original_in_idx] += weight * model_depthwise_conv.bias.data[original_in_idx]
                    depthwise_bias_count[original_in_idx] += weight

                depthwise_bn_weight_sum[original_in_idx] += weight * model_depthwise_bn.weight.data[original_in_idx]
                depthwise_bn_bias_sum[original_in_idx] += weight * model_depthwise_bn.bias.data[original_in_idx]
                depthwise_bn_running_mean_sum[original_in_idx] += (
                    weight * model_depthwise_bn.running_mean[original_in_idx]
                )
                depthwise_bn_running_var_sum[original_in_idx] += (
                    weight * model_depthwise_bn.running_var[original_in_idx]
                )
                depthwise_bn_count[original_in_idx] += weight

        # Aggregate pointwise conv (this is where output pruning happens)
        for pruned_out_idx, original_out_idx in enumerate(kept_indices):
            if prev_kept_indices is not None:
                # Map input channels using previous layer's mask
                for pruned_in_idx, original_in_idx in enumerate(prev_kept_indices):
                    pointwise_weight_sum[original_out_idx, original_in_idx, :, :] += (
                        weight * model_pointwise_conv.weight.data[pruned_out_idx, pruned_in_idx, :, :]
                    )
                    pointwise_weight_count[original_out_idx, original_in_idx] += weight
            else:
                # All input channels present
                pointwise_weight_sum[original_out_idx, :, :, :] += (
                    weight * model_pointwise_conv.weight.data[pruned_out_idx, :, :, :]
                )
                pointwise_weight_count[original_out_idx, :] += weight

            # Aggregate pointwise bias
            if pointwise_bias_sum is not None and model_pointwise_conv.bias is not None:
                pointwise_bias_sum[original_out_idx] += weight * model_pointwise_conv.bias.data[pruned_out_idx]
                pointwise_bias_count[original_out_idx] += weight

            # Aggregate pointwise BatchNorm
            pointwise_bn_weight_sum[original_out_idx] += weight * model_pointwise_bn.weight.data[pruned_out_idx]
            pointwise_bn_bias_sum[original_out_idx] += weight * model_pointwise_bn.bias.data[pruned_out_idx]
            pointwise_bn_running_mean_sum[original_out_idx] += weight * model_pointwise_bn.running_mean[pruned_out_idx]
            pointwise_bn_running_var_sum[original_out_idx] += weight * model_pointwise_bn.running_var[pruned_out_idx]
            pointwise_bn_count[original_out_idx] += weight

    # Apply weighted averages for depthwise conv
    depthwise_mask = depthwise_weight_count > 0
    for ch_idx in range(original_in_channels):
        if depthwise_mask[ch_idx]:
            aggregated_depthwise_conv.weight.data[ch_idx, :, :, :] = (
                depthwise_weight_sum[ch_idx, :, :, :] / depthwise_weight_count[ch_idx]
            )

    if depthwise_bias_sum is not None:
        depthwise_bias_mask = depthwise_bias_count > 0
        aggregated_depthwise_conv.bias.data[depthwise_bias_mask] = (
            depthwise_bias_sum[depthwise_bias_mask] / depthwise_bias_count[depthwise_bias_mask]
        )

    depthwise_bn_mask = depthwise_bn_count > 0
    aggregated_depthwise_bn.weight.data[depthwise_bn_mask] = (
        depthwise_bn_weight_sum[depthwise_bn_mask] / depthwise_bn_count[depthwise_bn_mask]
    )
    aggregated_depthwise_bn.bias.data[depthwise_bn_mask] = (
        depthwise_bn_bias_sum[depthwise_bn_mask] / depthwise_bn_count[depthwise_bn_mask]
    )
    aggregated_depthwise_bn.running_mean[depthwise_bn_mask] = (
        depthwise_bn_running_mean_sum[depthwise_bn_mask] / depthwise_bn_count[depthwise_bn_mask]
    )
    aggregated_depthwise_bn.running_var[depthwise_bn_mask] = (
        depthwise_bn_running_var_sum[depthwise_bn_mask] / depthwise_bn_count[depthwise_bn_mask]
    )

    # Apply weighted averages for pointwise conv
    pointwise_mask = pointwise_weight_count > 0
    aggregated_pointwise_conv.weight.data[pointwise_mask] = pointwise_weight_sum[
        pointwise_mask
    ] / pointwise_weight_count[pointwise_mask].unsqueeze(-1).unsqueeze(-1)

    if pointwise_bias_sum is not None:
        pointwise_bias_mask = pointwise_bias_count > 0
        aggregated_pointwise_conv.bias.data[pointwise_bias_mask] = (
            pointwise_bias_sum[pointwise_bias_mask] / pointwise_bias_count[pointwise_bias_mask]
        )

    pointwise_bn_mask = pointwise_bn_count > 0
    aggregated_pointwise_bn.weight.data[pointwise_bn_mask] = (
        pointwise_bn_weight_sum[pointwise_bn_mask] / pointwise_bn_count[pointwise_bn_mask]
    )
    aggregated_pointwise_bn.bias.data[pointwise_bn_mask] = (
        pointwise_bn_bias_sum[pointwise_bn_mask] / pointwise_bn_count[pointwise_bn_mask]
    )
    aggregated_pointwise_bn.running_mean[pointwise_bn_mask] = (
        pointwise_bn_running_mean_sum[pointwise_bn_mask] / pointwise_bn_count[pointwise_bn_mask]
    )
    aggregated_pointwise_bn.running_var[pointwise_bn_mask] = (
        pointwise_bn_running_var_sum[pointwise_bn_mask] / pointwise_bn_count[pointwise_bn_mask]
    )


def _aggregate_linear_layer(aggregated_linear, models, masks, weights):
    """Aggregate the linear layer from multiple models."""
    original_in_features = aggregated_linear.in_features
    original_out_features = aggregated_linear.out_features

    # Initialize accumulators
    linear_weight_sum = torch.zeros_like(aggregated_linear.weight.data)
    linear_weight_count = torch.zeros(original_out_features, original_in_features)
    linear_bias_sum = torch.zeros_like(aggregated_linear.bias.data) if aggregated_linear.bias is not None else None
    linear_bias_count = torch.zeros(original_out_features)

    # Aggregate from each model
    for model, mask_dict, weight in zip(models, masks, weights):
        # Get the mask for the last conv layer (affects linear layer input)
        # The last layer before fc is conv4.1 (second DepthSeparableConv2d in conv4)
        last_conv_mask_key = "conv4.1.pointwise.0"
        if last_conv_mask_key not in mask_dict:
            raise ValueError(f"Mask key '{last_conv_mask_key}' not found in mask_dict for client with weight {weight}.")

        last_conv_kept_indices = mask_dict[last_conv_mask_key]["indices_kept"]
        model_linear = model.fc

        # The linear layer input is the output of adaptive avg pooling
        # So each kept filter contributes 1 feature to the linear layer
        for pruned_idx, original_idx in enumerate(last_conv_kept_indices):
            linear_weight_sum[:, original_idx] += weight * model_linear.weight.data[:, pruned_idx]
            linear_weight_count[:, original_idx] += weight

        # Aggregate bias
        if linear_bias_sum is not None and model_linear.bias is not None:
            linear_bias_sum += weight * model_linear.bias.data
            linear_bias_count += weight

    # Apply weighted averages
    mask = linear_weight_count > 0
    aggregated_linear.weight.data[mask] = linear_weight_sum[mask] / linear_weight_count[mask]

    if linear_bias_sum is not None:
        bias_mask = linear_bias_count > 0
        aggregated_linear.bias.data[bias_mask] = linear_bias_sum[bias_mask] / linear_bias_count[bias_mask]


def _get_nested_attr(obj, attr_path):
    """Get nested attribute from object using dot-separated path."""
    attrs = attr_path.split(".")
    for attr in attrs:
        obj = getattr(obj, attr)
    return obj


def _get_previous_layer_indices(current_layer_name, mask_dict):
    """
    Get the indices of kept filters from the previous layer.

    This function determines which layer comes before the current layer
    and returns its kept indices from the mask dictionary.
    """
    # Parse the layer name to determine the previous layer
    if current_layer_name == "stem.0":
        # First layer - no previous layer
        return None
    elif current_layer_name == "stem.1":
        # Previous is stem.0 (BasicConv2d)
        prev_key = "stem.0.conv"
    elif current_layer_name.startswith("conv1.0"):
        # Previous is stem.1 (DepthSeparableConv2d)
        prev_key = "stem.1.pointwise.0"
    else:
        # Parse block name like "conv1.1", "conv2.0", etc.
        parts = current_layer_name.split(".")
        block_name = parts[0]  # e.g., "conv1", "conv2"
        block_idx = int(parts[1])  # e.g., 0, 1

        if block_idx == 0:
            # First block in this conv group - previous is from previous group
            if block_name == "conv2":
                prev_key = "conv1.1.pointwise.0"
            elif block_name == "conv3":
                prev_key = "conv2.1.pointwise.0"
            elif block_name == "conv4":
                prev_key = "conv3.5.pointwise.0"
            else:
                # Shouldn't happen, but handle gracefully
                raise ValueError(f"Block name {block_name} not recognized.")
        else:
            # Previous block in same group
            prev_key = f"{block_name}.{block_idx - 1}.pointwise.0"

    if prev_key not in mask_dict:
        raise ValueError(f"Previous layer mask key '{prev_key}' not found in mask_dict.")
    return mask_dict[prev_key]["indices_kept"]
