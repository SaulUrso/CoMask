from typing import Dict, List

import torch
import torch.nn as nn

from ..models.cnn import HARBox_CNN


def aggregate_cnn_with_masks(
    models: List[nn.Module],
    masks: List[Dict],
    counters: List[int],
) -> nn.Module:
    """
    Aggregate multiple pruned HARBox_CNN models using their pruning masks.

    This function performs weighted aggregation of pruned models, where each parameter
    is the weighted sum (weighted by counters) of parameters whose mask is not zero.

    Args:
        models: List of pruned HARBox_CNN models
        masks: List of mask dictionaries for each model. Each mask dict contains:
               - "mask": torch.Tensor - Boolean mask indicating kept filters
               - "indices_kept": np.ndarray - Indices of kept filters in original model
               - "layer_type": str - Type of layer (e.g., "conv")
               - "original_filters": int - Number of filters in original unpruned model
               - "pruned_filters": int - Number of filters kept after pruning
        counters: List of weights for weighted averaging (e.g., number of data samples)

    Returns:
        Aggregated HARBox_CNN model with structure matching the union of all kept filters

    Example:
        >>> model1 = prune_model(HARBox_CNN(), rate=0.3)  # Pruned model 1
        >>> model2 = prune_model(HARBox_CNN(), rate=0.5)  # Pruned model 2
        >>> aggregated = aggregate_cnn_with_masks(
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
    aggregated_model = HARBox_CNN()

    # Iterate over all named modules in the aggregated model
    for name, module in aggregated_model.named_modules():
        if name == "":  # Skip root module
            continue

        # Handle convolutional layers (Sequential blocks)
        if isinstance(module, nn.Sequential) and "conv" in name:
            layer_idx = int(name[-1]) - 1  # conv1 -> 0, conv2 -> 1
            mask_key = f"conv_{layer_idx}"

            conv_layer = module[0]  # Conv2d
            bn_layer = module[1]  # BatchNorm2d

            _aggregate_conv_block(conv_layer, bn_layer, models, masks, weights, mask_key, layer_idx)

        # Handle linear layer
        elif isinstance(module, nn.Linear) and name == "linear":
            _aggregate_linear_layer(module, models, masks, weights)

    return aggregated_model


def _aggregate_conv_block(aggregated_conv, aggregated_bn, models, masks, weights, mask_key, layer_idx):
    """Aggregate a convolutional block (Conv2d + BatchNorm2d) from multiple models."""
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
            continue

        kept_indices = mask_dict[mask_key]["indices_kept"]

        # Get corresponding layer from pruned model
        conv_name = f"conv{layer_idx + 1}"
        model_seq = getattr(model, conv_name)
        model_conv = model_seq[0]
        model_bn = model_seq[1]

        # Get previous layer's kept indices for input channel mapping
        prev_kept_indices = None
        if layer_idx > 0:
            prev_mask_key = f"conv_{layer_idx - 1}"
            prev_kept_indices = mask_dict[prev_mask_key]["indices_kept"]

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
            if conv_bias_sum is not None:
                conv_bias_sum[original_idx] += weight * model_conv.bias.data[pruned_idx]
                conv_bias_count[original_idx] += weight

            # Aggregate BatchNorm parameters
            bn_weight_sum[original_idx] += weight * model_bn.weight.data[pruned_idx]
            bn_bias_sum[original_idx] += weight * model_bn.bias.data[pruned_idx]
            bn_running_mean_sum[original_idx] += weight * model_bn.running_mean[pruned_idx]
            bn_running_var_sum[original_idx] += weight * model_bn.running_var[pruned_idx]
            bn_count[original_idx] += weight

    # Apply weighted averages using vectorized operations where possible
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
        conv2_mask_key = "conv_1"
        if conv2_mask_key not in mask_dict:
            raise ValueError(f"{conv2_mask_key} not present in mask a client with {weight} samples.")

        conv2_kept_indices = mask_dict[conv2_mask_key]["indices_kept"]
        model_linear = model.linear

        # Map linear input features based on conv2 output
        # Each conv2 filter contributes 8*8=64 features to linear layer
        for pruned_conv2_idx, original_conv2_idx in enumerate(conv2_kept_indices):
            pruned_start = pruned_conv2_idx * 64
            pruned_end = pruned_start + 64
            original_start = original_conv2_idx * 64
            original_end = original_start + 64

            linear_weight_sum[:, original_start:original_end] += (
                weight * model_linear.weight.data[:, pruned_start:pruned_end]
            )
            linear_weight_count[:, original_start:original_end] += weight

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
