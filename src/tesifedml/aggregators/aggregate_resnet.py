"""
ResNet Model Aggregation with Pruning Masks

This module provides functionality to aggregate multiple pruned ResNet models using their
pruning masks. It supports weighted aggregation based on client data counts and handles:
- Initial convolutional layer (conv1)
- ResNet layers with BasicBlocks (layer1-4)
- Downsampling layers within BasicBlocks
- Final fully connected layer
- Both BatchNorm2d and GroupNorm2d normalization layers

The aggregation follows the original filter positions, ensuring that each filter in the
aggregated model is the weighted average of all corresponding filters from client models
that retained that specific filter after pruning.
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
from fedml.model.cv.resnet_cifar import BasicBlock, resnet18_cifar


def aggregate_resnet_with_masks(
    models: List[nn.Module],
    masks: List[Dict],
    counters: List[int],
    num_classes: int = 10,
    group_norm: int = 0,
) -> nn.Module:
    """
    Aggregate multiple pruned ResNet models using their pruning masks.

    This function performs weighted aggregation of pruned ResNet models, where each parameter
    is the weighted sum (weighted by counters) of parameters whose mask is not zero.

    Args:
        models: List of pruned ResNet models
        masks: List of mask dictionaries for each model. Each mask dict contains layer-wise masks with:
               - "mask": torch.Tensor - Boolean mask indicating kept filters
               - "indices_kept": np.ndarray - Indices of kept filters in original model
               - "layer_type": str - Type of layer (e.g., "conv")
               - "original_filters": int - Number of filters in original unpruned model
               - "pruned_filters": int - Number of filters kept after pruning
        counters: List of weights for weighted averaging (e.g., number of data samples)
        num_classes: Number of output classes for the classifier
        group_norm: Group normalization parameter (0 means BatchNorm2d)

    Returns:
        Aggregated ResNet model with structure matching the union of all kept filters

    Example:
        >>> model1 = prune_resnet(resnet18_cifar(), rate=0.3)
        >>> model2 = prune_resnet(resnet18_cifar(), rate=0.5)
        >>> aggregated = aggregate_resnet_with_masks(
        ...     [model1, model2],
        ...     [mask1, mask2],
        ...     [100, 50]  # model1 has 100 samples, model2 has 50
        ... )
    """
    assert len(models) == len(masks) == len(counters), "Models, masks, and counters must have same length"
    assert len(models) > 0, "Must provide at least one model"

    # Normalize counters to sum to 1 (for weighted averaging)
    total_count = sum(counters)
    weights = [c / total_count for c in counters]

    # Create a new unpruned model as template for aggregation
    aggregated_model = resnet18_cifar(num_classes=num_classes, group_norm=group_norm)

    # Aggregate initial conv1 layer
    _aggregate_conv_bn(
        aggregated_model.conv1,
        aggregated_model.bn1,
        models,
        masks,
        weights,
        mask_key="conv1",
        prev_mask_key=None,  # First layer has no previous layer
    )

    # Aggregate each ResNet layer (layer1, layer2, layer3, layer4)
    for layer_name in ["layer1", "layer2", "layer3", "layer4"]:
        layer = getattr(aggregated_model, layer_name)

        # Aggregate each BasicBlock in the layer
        for block_idx, block in enumerate(layer):
            block_prefix = f"{layer_name}.{block_idx}"

            # Determine previous layer mask key for input channel mapping
            if layer_name == "layer1" and block_idx == 0:
                # First block of layer1 follows conv1
                prev_mask_key = "conv1"
            elif block_idx == 0:
                # First block of other layers follows last block of previous layer
                prev_layer_num = int(layer_name[-1]) - 1
                prev_layer_name = f"layer{prev_layer_num}"
                # Get number of blocks in previous layer
                prev_layer = getattr(aggregated_model, prev_layer_name)
                prev_block_idx = len(prev_layer) - 1
                prev_mask_key = f"{prev_layer_name}.{prev_block_idx}.conv2"
            else:
                # Other blocks follow previous block in same layer
                prev_mask_key = f"{layer_name}.{block_idx - 1}.conv2"

            _aggregate_basic_block(
                block,
                models,
                masks,
                weights,
                block_prefix,
                prev_mask_key,
            )

    # Aggregate final linear layer
    # The input to fc comes from the last block of layer4
    last_conv_mask_key = "layer4.1.conv2"  # layer4 has 2 blocks (0 and 1) for resnet18
    _aggregate_linear_layer(
        aggregated_model.fc,
        models,
        masks,
        weights,
        last_conv_mask_key,
    )

    return aggregated_model


def _aggregate_basic_block(
    aggregated_block: BasicBlock,
    models: List[nn.Module],
    masks: List[Dict],
    weights: List[float],
    block_prefix: str,
    prev_mask_key: str,
):
    """
    Aggregate a BasicBlock from multiple models.

    Args:
        aggregated_block: The BasicBlock in the aggregated model to fill
        models: List of pruned models
        masks: List of mask dictionaries
        weights: Normalized weights for averaging
        block_prefix: Prefix for mask keys (e.g., "layer1.0")
        prev_mask_key: Mask key of the previous layer for input channel mapping
    """
    # Aggregate conv1
    conv1_mask_key = f"{block_prefix}.conv1"
    _aggregate_conv_bn(
        aggregated_block.conv1,
        aggregated_block.bn1,
        models,
        masks,
        weights,
        mask_key=conv1_mask_key,
        prev_mask_key=prev_mask_key,
    )

    # Aggregate conv2
    conv2_mask_key = f"{block_prefix}.conv2"
    _aggregate_conv_bn(
        aggregated_block.conv2,
        aggregated_block.bn2,
        models,
        masks,
        weights,
        mask_key=conv2_mask_key,
        prev_mask_key=conv1_mask_key,
    )

    # Aggregate downsample if present
    if aggregated_block.downsample is not None:
        downsample_conv = aggregated_block.downsample[0]
        downsample_bn = aggregated_block.downsample[1]

        # Downsample uses same input as block's conv1 and same output as block's conv2
        # We need to aggregate it separately with a special handler
        _aggregate_downsample(
            downsample_conv,
            downsample_bn,
            models,
            masks,
            weights,
            block_prefix,
            conv2_mask_key,  # Output matches conv2
            prev_mask_key,  # Input matches block input
        )


def _aggregate_conv_bn(
    aggregated_conv: nn.Conv2d,
    aggregated_bn: nn.Module,  # Can be BatchNorm2d or GroupNorm2d
    models: List[nn.Module],
    masks: List[Dict],
    weights: List[float],
    mask_key: str,
    prev_mask_key: Optional[str] = None,
):
    """
    Aggregate a Conv2d + BatchNorm2d pair from multiple models.

    Args:
        aggregated_conv: Conv2d layer in aggregated model
        aggregated_bn: BatchNorm2d layer in aggregated model
        models: List of pruned models
        masks: List of mask dictionaries
        weights: Normalized weights for averaging
        mask_key: Key for this layer's mask
        prev_mask_key: Key for previous layer's mask (for input channel mapping)
    """
    original_out_channels = aggregated_conv.out_channels
    original_in_channels = aggregated_conv.in_channels

    # Initialize accumulators
    conv_weight_sum = torch.zeros_like(aggregated_conv.weight.data)
    conv_weight_count = torch.zeros(original_out_channels, original_in_channels, 1, 1)
    conv_bias_sum = torch.zeros_like(aggregated_conv.bias.data) if aggregated_conv.bias is not None else None
    conv_bias_count = torch.zeros(original_out_channels)

    bn_weight_sum = torch.zeros_like(aggregated_bn.weight.data)
    bn_bias_sum = torch.zeros_like(aggregated_bn.bias.data)

    # Handle running stats - they might be None for GroupNorm2d
    bn_running_mean_sum = (
        torch.zeros_like(aggregated_bn.running_mean)
        if hasattr(aggregated_bn, "running_mean") and aggregated_bn.running_mean is not None
        else None
    )
    bn_running_var_sum = (
        torch.zeros_like(aggregated_bn.running_var)
        if hasattr(aggregated_bn, "running_var") and aggregated_bn.running_var is not None
        else None
    )
    bn_count = torch.zeros(original_out_channels)

    # Aggregate from each model
    for model, mask_dict, weight in zip(models, masks, weights):
        if mask_key not in mask_dict:
            continue

        kept_indices = mask_dict[mask_key]["indices_kept"]

        # Get previous layer's kept indices for input channel mapping
        prev_kept_indices = None
        if prev_mask_key is not None and prev_mask_key in mask_dict:
            prev_kept_indices = mask_dict[prev_mask_key]["indices_kept"]

        # Navigate to the corresponding layer in the pruned model
        model_conv, model_bn = _get_conv_bn_from_model(model, mask_key)

        if model_conv is None or model_bn is None:
            continue

        # Aggregate each kept filter
        for pruned_idx, original_idx in enumerate(kept_indices):
            # Aggregate Conv2d weights
            if prev_kept_indices is None:
                # First layer or no previous pruning - all input channels present
                conv_weight_sum[original_idx, :, :, :] += weight * model_conv.weight.data[pruned_idx, :, :, :]
                conv_weight_count[original_idx, :, :, :] += weight
            else:
                # Map input channels using previous layer's mask
                for pruned_in_idx, original_in_idx in enumerate(prev_kept_indices):
                    conv_weight_sum[original_idx, original_in_idx, :, :] += (
                        weight * model_conv.weight.data[pruned_idx, pruned_in_idx, :, :]
                    )
                    conv_weight_count[original_idx, original_in_idx, :, :] += weight

            # Aggregate Conv2d bias (if present)
            if conv_bias_sum is not None and model_conv.bias is not None:
                conv_bias_sum[original_idx] += weight * model_conv.bias.data[pruned_idx]
                conv_bias_count[original_idx] += weight

            # Aggregate BatchNorm parameters
            bn_weight_sum[original_idx] += weight * model_bn.weight.data[pruned_idx]
            bn_bias_sum[original_idx] += weight * model_bn.bias.data[pruned_idx]

            # Aggregate running stats if they exist
            if (
                bn_running_mean_sum is not None
                and hasattr(model_bn, "running_mean")
                and model_bn.running_mean is not None
            ):
                bn_running_mean_sum[original_idx] += weight * model_bn.running_mean[pruned_idx]
            if bn_running_var_sum is not None and hasattr(model_bn, "running_var") and model_bn.running_var is not None:
                bn_running_var_sum[original_idx] += weight * model_bn.running_var[pruned_idx]

            bn_count[original_idx] += weight

    # Apply weighted averages using torch.where to handle broadcasting properly
    # Create a mask for non-zero counts (shape: [out_channels, in_channels, 1, 1])
    mask = conv_weight_count > 0
    # Divide element-wise where mask is True, preserving spatial dimensions
    aggregated_conv.weight.data = torch.where(mask, conv_weight_sum / conv_weight_count, aggregated_conv.weight.data)

    if conv_bias_sum is not None and aggregated_conv.bias is not None:
        bias_mask = conv_bias_count > 0
        aggregated_conv.bias.data[bias_mask] = conv_bias_sum[bias_mask] / conv_bias_count[bias_mask]

    bn_mask = bn_count > 0
    aggregated_bn.weight.data[bn_mask] = bn_weight_sum[bn_mask] / bn_count[bn_mask]
    aggregated_bn.bias.data[bn_mask] = bn_bias_sum[bn_mask] / bn_count[bn_mask]

    # Update running stats if they exist
    if (
        bn_running_mean_sum is not None
        and hasattr(aggregated_bn, "running_mean")
        and aggregated_bn.running_mean is not None
    ):
        aggregated_bn.running_mean[bn_mask] = bn_running_mean_sum[bn_mask] / bn_count[bn_mask]
    if (
        bn_running_var_sum is not None
        and hasattr(aggregated_bn, "running_var")
        and aggregated_bn.running_var is not None
    ):
        aggregated_bn.running_var[bn_mask] = bn_running_var_sum[bn_mask] / bn_count[bn_mask]


def _aggregate_downsample(
    aggregated_conv: nn.Conv2d,
    aggregated_bn: nn.Module,
    models: List[nn.Module],
    masks: List[Dict],
    weights: List[float],
    block_prefix: str,
    output_mask_key: str,
    input_mask_key: Optional[str] = None,
):
    """
    Aggregate a downsample layer (Conv2d + BatchNorm2d) from multiple models.

    This is similar to _aggregate_conv_bn but specifically handles the downsample
    layers within BasicBlocks, which need to be accessed differently in the model structure.

    Args:
        aggregated_conv: Conv2d layer in aggregated model's downsample
        aggregated_bn: BatchNorm2d layer in aggregated model's downsample
        models: List of pruned models
        masks: List of mask dictionaries
        weights: Normalized weights for averaging
        block_prefix: Prefix for the block (e.g., "layer1.0")
        output_mask_key: Mask key determining output channels (conv2's mask)
        input_mask_key: Mask key determining input channels (previous layer's mask)
    """
    original_out_channels = aggregated_conv.out_channels
    original_in_channels = aggregated_conv.in_channels

    # Initialize accumulators
    conv_weight_sum = torch.zeros_like(aggregated_conv.weight.data)
    conv_weight_count = torch.zeros(original_out_channels, original_in_channels, 1, 1)
    conv_bias_sum = torch.zeros_like(aggregated_conv.bias.data) if aggregated_conv.bias is not None else None
    conv_bias_count = torch.zeros(original_out_channels)

    bn_weight_sum = torch.zeros_like(aggregated_bn.weight.data)
    bn_bias_sum = torch.zeros_like(aggregated_bn.bias.data)

    # Handle running stats - they might be None for GroupNorm2d
    bn_running_mean_sum = (
        torch.zeros_like(aggregated_bn.running_mean)
        if hasattr(aggregated_bn, "running_mean") and aggregated_bn.running_mean is not None
        else None
    )
    bn_running_var_sum = (
        torch.zeros_like(aggregated_bn.running_var)
        if hasattr(aggregated_bn, "running_var") and aggregated_bn.running_var is not None
        else None
    )
    bn_count = torch.zeros(original_out_channels)

    # Aggregate from each model
    for model, mask_dict, weight in zip(models, masks, weights):
        if output_mask_key not in mask_dict:
            continue

        output_kept_indices = mask_dict[output_mask_key]["indices_kept"]

        # Get input layer's kept indices
        input_kept_indices = None
        if input_mask_key is not None and input_mask_key in mask_dict:
            input_kept_indices = mask_dict[input_mask_key]["indices_kept"]

        # Navigate to the downsample layer in the pruned model
        parts = block_prefix.split(".")
        layer_name = parts[0]  # e.g., "layer1"
        block_idx = int(parts[1])  # e.g., 0

        layer = getattr(model, layer_name)
        block = layer[block_idx]

        if block.downsample is None:
            continue

        model_conv = block.downsample[0]
        model_bn = block.downsample[1]

        # Aggregate each kept filter
        for pruned_idx, original_idx in enumerate(output_kept_indices):
            # Aggregate Conv2d weights
            if input_kept_indices is None:
                # First layer or no previous pruning - all input channels present
                conv_weight_sum[original_idx, :, :, :] += weight * model_conv.weight.data[pruned_idx, :, :, :]
                conv_weight_count[original_idx, :, :, :] += weight
            else:
                # Map input channels using previous layer's mask
                for pruned_in_idx, original_in_idx in enumerate(input_kept_indices):
                    conv_weight_sum[original_idx, original_in_idx, :, :] += (
                        weight * model_conv.weight.data[pruned_idx, pruned_in_idx, :, :]
                    )
                    conv_weight_count[original_idx, original_in_idx, :, :] += weight

            # Aggregate Conv2d bias (if present)
            if conv_bias_sum is not None and model_conv.bias is not None:
                conv_bias_sum[original_idx] += weight * model_conv.bias.data[pruned_idx]
                conv_bias_count[original_idx] += weight

            # Aggregate BatchNorm parameters
            bn_weight_sum[original_idx] += weight * model_bn.weight.data[pruned_idx]
            bn_bias_sum[original_idx] += weight * model_bn.bias.data[pruned_idx]

            # Aggregate running stats if they exist
            if (
                bn_running_mean_sum is not None
                and hasattr(model_bn, "running_mean")
                and model_bn.running_mean is not None
            ):
                bn_running_mean_sum[original_idx] += weight * model_bn.running_mean[pruned_idx]
            if bn_running_var_sum is not None and hasattr(model_bn, "running_var") and model_bn.running_var is not None:
                bn_running_var_sum[original_idx] += weight * model_bn.running_var[pruned_idx]

            bn_count[original_idx] += weight

    # Apply weighted averages using torch.where to handle broadcasting properly
    # Create a mask for non-zero counts (shape: [out_channels, in_channels, 1, 1])
    mask = conv_weight_count > 0
    # Divide element-wise where mask is True, preserving spatial dimensions
    aggregated_conv.weight.data = torch.where(mask, conv_weight_sum / conv_weight_count, aggregated_conv.weight.data)

    if conv_bias_sum is not None and aggregated_conv.bias is not None:
        bias_mask = conv_bias_count > 0
        aggregated_conv.bias.data[bias_mask] = conv_bias_sum[bias_mask] / conv_bias_count[bias_mask]

    bn_mask = bn_count > 0
    aggregated_bn.weight.data[bn_mask] = bn_weight_sum[bn_mask] / bn_count[bn_mask]
    aggregated_bn.bias.data[bn_mask] = bn_bias_sum[bn_mask] / bn_count[bn_mask]

    # Update running stats if they exist
    if (
        bn_running_mean_sum is not None
        and hasattr(aggregated_bn, "running_mean")
        and aggregated_bn.running_mean is not None
    ):
        aggregated_bn.running_mean[bn_mask] = bn_running_mean_sum[bn_mask] / bn_count[bn_mask]
    if (
        bn_running_var_sum is not None
        and hasattr(aggregated_bn, "running_var")
        and aggregated_bn.running_var is not None
    ):
        aggregated_bn.running_var[bn_mask] = bn_running_var_sum[bn_mask] / bn_count[bn_mask]


def _get_conv_bn_from_model(model: nn.Module, mask_key: str):
    """
    Navigate the model structure to get Conv2d and BatchNorm2d layers based on mask_key.

    Args:
        model: The pruned model
        mask_key: Key like "conv1", "layer1.0.conv1", "layer2.1.conv2", etc.

    Returns:
        Tuple of (Conv2d, BatchNorm2d) or (None, None) if not found
    """
    parts = mask_key.split(".")

    if parts[0] == "conv1":
        # Initial conv layer
        return model.conv1, model.bn1

    elif parts[0].startswith("layer"):
        # Layer block convolution
        # Format: layer1.0.conv1 -> layer1[0].conv1, layer1[0].bn1
        layer_name = parts[0]  # e.g., "layer1"
        block_idx = int(parts[1])  # e.g., 0
        conv_name = parts[2]  # e.g., "conv1" or "conv2"

        layer = getattr(model, layer_name)
        block = layer[block_idx]

        if conv_name == "conv1":
            return block.conv1, block.bn1
        elif conv_name == "conv2":
            return block.conv2, block.bn2

    return None, None


def _aggregate_linear_layer(
    aggregated_linear: nn.Linear,
    models: List[nn.Module],
    masks: List[Dict],
    weights: List[float],
    last_conv_mask_key: str,
):
    """
    Aggregate the final linear layer from multiple models.

    Args:
        aggregated_linear: Linear layer in aggregated model
        models: List of pruned models
        masks: List of mask dictionaries
        weights: Normalized weights for averaging
        last_conv_mask_key: Mask key of the last convolutional layer
    """
    original_in_features = aggregated_linear.in_features
    original_out_features = aggregated_linear.out_features

    # Initialize accumulators
    linear_weight_sum = torch.zeros_like(aggregated_linear.weight.data)
    linear_weight_count = torch.zeros(original_out_features, original_in_features)
    linear_bias_sum = torch.zeros_like(aggregated_linear.bias.data) if aggregated_linear.bias is not None else None
    linear_bias_count = torch.zeros(original_out_features)

    # Aggregate from each model
    for model, mask_dict, weight in zip(models, masks, weights):
        if last_conv_mask_key not in mask_dict:
            raise ValueError(f"{last_conv_mask_key} not present in mask for a client with {weight} samples.")

        last_conv_kept_indices = mask_dict[last_conv_mask_key]["indices_kept"]
        model_linear = model.fc

        # In ResNet, the avgpool reduces spatial dimensions to 1x1
        # So each filter in the last conv layer contributes exactly 1 feature to the linear layer
        # Linear layer input features = number of filters in last conv layer
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
