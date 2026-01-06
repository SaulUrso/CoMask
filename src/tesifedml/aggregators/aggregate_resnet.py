from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

from tesifedml.models.resnet_cifar import resnet18_cifar


def aggregate_resnet_with_masks(
    models: List[nn.Module],
    masks: List[Dict],
    counters: List[int],
    num_classes: int = 10,
) -> nn.Module:
    """
    Aggregate multiple pruned ResNet18 models using their pruning masks.

    This function performs weighted aggregation of pruned models, where each parameter
    is the weighted sum (weighted by counters) of parameters whose mask is not zero.
    Parameters that receive no contribution from any model remain as zeros.

    Args:
        models: List of pruned ResNet18 models
        masks: List of mask dictionaries for each model. Each mask dict contains layer names as keys,
               mapping to dict with:
               - "mask": torch.Tensor - Boolean mask indicating kept filters
               - "indices_kept": np.ndarray - Indices of kept filters in original model
               - "layer_type": str - Type of layer (e.g., "conv")
               - "original_filters": int - Number of filters in original unpruned model
               - "pruned_filters": int - Number of filters kept after pruning
        counters: List of weights for weighted averaging (e.g., number of data samples per client)
        num_classes: Number of output classes

    Returns:
        Aggregated ResNet18 model with structure matching the original unpruned model
    """
    assert len(models) == len(masks) == len(counters), "Models, masks, and counters must have same length"
    assert len(models) > 0, "Must provide at least one model"

    # Normalize counters to sum to 1 (for weighted averaging)
    total_count = sum(counters)
    weights = [c / total_count for c in counters]

    # Create a new unpruned model as template for aggregation
    aggregated_model = resnet18_cifar(num_classes=num_classes)

    # Initialize all parameters to zero
    with torch.no_grad():
        for param in aggregated_model.parameters():
            param.zero_()
        # Also zero out running stats in BatchNorm layers
        for module in aggregated_model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.running_mean.zero_()
                module.running_var.zero_()

    # Aggregate initial conv1 and bn1
    _aggregate_conv_bn(
        aggregated_model.conv1,
        aggregated_model.bn1,
        models,
        masks,
        weights,
        mask_key="conv1",
        conv_attr="conv1",
        bn_attr="bn1",
        get_prev_indices_fn=lambda mask_dict: None,  # First layer, no previous
    )

    # Aggregate each layer (layer1, layer2, layer3, layer4)
    for layer_name in ["layer1", "layer2", "layer3", "layer4"]:
        aggregated_layer = getattr(aggregated_model, layer_name)

        for block_idx, aggregated_block in enumerate(aggregated_layer):
            block_key = f"{layer_name}.{block_idx}"

            # Aggregate conv1 in the block
            _aggregate_conv_bn(
                aggregated_block.conv1,
                aggregated_block.bn1,
                models,
                masks,
                weights,
                mask_key=f"{block_key}.conv1",
                conv_attr=f"{layer_name}.{block_idx}.conv1",
                bn_attr=f"{layer_name}.{block_idx}.bn1",
                get_prev_indices_fn=lambda md, bk=block_key: _get_previous_layer_indices_resnet(f"{bk}.conv1", md),
            )

            # Aggregate conv2 in the block
            _aggregate_conv_bn(
                aggregated_block.conv2,
                aggregated_block.bn2,
                models,
                masks,
                weights,
                mask_key=f"{block_key}.conv2",
                conv_attr=f"{layer_name}.{block_idx}.conv2",
                bn_attr=f"{layer_name}.{block_idx}.bn2",
                get_prev_indices_fn=lambda md, bk=block_key: md[f"{bk}.conv1"]["indices_kept"],
            )

            # Aggregate downsample if present
            if aggregated_block.downsample is not None:
                _aggregate_downsample(
                    aggregated_block.downsample,
                    models,
                    masks,
                    weights,
                    block_key=block_key,
                    layer_name=layer_name,
                    block_idx=block_idx,
                )

    # Aggregate the final fully connected layer
    _aggregate_fc_layer(
        aggregated_model.fc,
        models,
        masks,
        weights,
    )

    return aggregated_model


def _aggregate_conv_bn(
    aggregated_conv: nn.Conv2d,
    aggregated_bn: nn.BatchNorm2d,
    models: List[nn.Module],
    masks: List[Dict],
    weights: List[float],
    mask_key: str,
    conv_attr: str,
    bn_attr: str,
    get_prev_indices_fn,
):
    """Aggregate a Conv2d + BatchNorm2d pair from multiple pruned models."""
    original_out_channels = aggregated_conv.out_channels
    original_in_channels = aggregated_conv.in_channels
    device = aggregated_conv.weight.device

    # Initialize accumulators
    conv_weight_sum = torch.zeros_like(aggregated_conv.weight.data)
    conv_weight_count = torch.zeros(original_out_channels, original_in_channels, device=device)

    bn_weight_sum = torch.zeros_like(aggregated_bn.weight.data)
    bn_bias_sum = torch.zeros_like(aggregated_bn.bias.data)
    bn_running_mean_sum = torch.zeros_like(aggregated_bn.running_mean)
    bn_running_var_sum = torch.zeros_like(aggregated_bn.running_var)
    bn_count = torch.zeros(original_out_channels, device=device)

    # Aggregate from each model
    for model, mask_dict, weight in zip(models, masks, weights):
        kept_indices = mask_dict[mask_key]["indices_kept"]
        kept_indices = np.asarray(kept_indices)
        kept_indices_tensor = torch.from_numpy(kept_indices).long().to(device)

        # Get corresponding conv and bn from pruned model
        model_conv = _get_nested_attr(model, conv_attr)
        model_bn = _get_nested_attr(model, bn_attr)

        # Get previous layer's kept indices for input channel mapping
        prev_kept_indices = get_prev_indices_fn(mask_dict)

        # Aggregate Conv2d weights (vectorized)
        if prev_kept_indices is None:
            # First layer - all input channels present
            conv_weight_sum.index_add_(0, kept_indices_tensor, weight * model_conv.weight.data)
            conv_weight_count[kept_indices, :] += weight
        else:
            # Vectorized: map input channels to original positions using advanced indexing
            prev_kept_indices = np.asarray(prev_kept_indices)
            out_grid, in_grid = np.ix_(kept_indices, prev_kept_indices)
            conv_weight_sum[out_grid, in_grid, :, :] += weight * model_conv.weight.data.cpu().numpy()
            conv_weight_count[out_grid, in_grid] += weight

        # Aggregate BatchNorm parameters (vectorized using index_add_)
        bn_weight_sum.index_add_(0, kept_indices_tensor, weight * model_bn.weight.data)
        bn_bias_sum.index_add_(0, kept_indices_tensor, weight * model_bn.bias.data)
        bn_running_mean_sum.index_add_(0, kept_indices_tensor, weight * model_bn.running_mean)
        bn_running_var_sum.index_add_(0, kept_indices_tensor, weight * model_bn.running_var)
        bn_count[kept_indices] += weight

    # Apply weighted averages (positions with no contribution remain zero)
    mask = conv_weight_count > 0
    aggregated_conv.weight.data[mask] = conv_weight_sum[mask] / conv_weight_count[mask].unsqueeze(-1).unsqueeze(-1)

    bn_mask = bn_count > 0
    aggregated_bn.weight.data[bn_mask] = bn_weight_sum[bn_mask] / bn_count[bn_mask]
    aggregated_bn.bias.data[bn_mask] = bn_bias_sum[bn_mask] / bn_count[bn_mask]
    aggregated_bn.running_mean[bn_mask] = bn_running_mean_sum[bn_mask] / bn_count[bn_mask]
    aggregated_bn.running_var[bn_mask] = bn_running_var_sum[bn_mask] / bn_count[bn_mask]


def _aggregate_downsample(
    aggregated_downsample: nn.Sequential,
    models: List[nn.Module],
    masks: List[Dict],
    weights: List[float],
    block_key: str,
    layer_name: str,
    block_idx: int,
):
    """Aggregate a downsample layer (Conv2d + BatchNorm2d) from multiple pruned models.

    Downsample layers follow the block's input indices for input channels
    and conv2's output indices for output channels.
    """
    aggregated_conv = aggregated_downsample[0]
    aggregated_bn = aggregated_downsample[1]

    original_out_channels = aggregated_conv.out_channels
    original_in_channels = aggregated_conv.in_channels
    device = aggregated_conv.weight.device

    # Initialize accumulators
    conv_weight_sum = torch.zeros_like(aggregated_conv.weight.data)
    conv_weight_count = torch.zeros(original_out_channels, original_in_channels, device=device)

    bn_weight_sum = torch.zeros_like(aggregated_bn.weight.data)
    bn_bias_sum = torch.zeros_like(aggregated_bn.bias.data)
    bn_running_mean_sum = torch.zeros_like(aggregated_bn.running_mean)
    bn_running_var_sum = torch.zeros_like(aggregated_bn.running_var)
    bn_count = torch.zeros(original_out_channels, device=device)

    # Aggregate from each model
    for model, mask_dict, weight in zip(models, masks, weights):
        # Output indices come from conv2 of this block
        out_indices = mask_dict[f"{block_key}.conv2"]["indices_kept"]
        out_indices = np.asarray(out_indices)
        out_indices_tensor = torch.from_numpy(out_indices).long().to(device)

        # Input indices come from the previous layer (block input)
        in_indices = _get_previous_layer_indices_resnet(f"{block_key}.conv1", mask_dict)

        # Get corresponding downsample from pruned model
        model_block = _get_nested_attr(model, f"{layer_name}.{block_idx}")
        model_conv = model_block.downsample[0]
        model_bn = model_block.downsample[1]

        # Aggregate Conv2d weights (vectorized)
        if in_indices is None:
            # First block input from conv1 (shouldn't happen for downsample, but handle it)
            conv_weight_sum.index_add_(0, out_indices_tensor, weight * model_conv.weight.data)
            conv_weight_count[out_indices, :] += weight
        else:
            # Vectorized: map input channels to original positions using advanced indexing
            in_indices = np.asarray(in_indices)
            out_grid, in_grid = np.ix_(out_indices, in_indices)
            conv_weight_sum[out_grid, in_grid, :, :] += weight * model_conv.weight.data.cpu().numpy()
            conv_weight_count[out_grid, in_grid] += weight

        # Aggregate BatchNorm parameters (vectorized using index_add_)
        bn_weight_sum.index_add_(0, out_indices_tensor, weight * model_bn.weight.data)
        bn_bias_sum.index_add_(0, out_indices_tensor, weight * model_bn.bias.data)
        bn_running_mean_sum.index_add_(0, out_indices_tensor, weight * model_bn.running_mean)
        bn_running_var_sum.index_add_(0, out_indices_tensor, weight * model_bn.running_var)
        bn_count[out_indices] += weight

    # Apply weighted averages
    mask = conv_weight_count > 0
    aggregated_conv.weight.data[mask] = conv_weight_sum[mask] / conv_weight_count[mask].unsqueeze(-1).unsqueeze(-1)

    bn_mask = bn_count > 0
    aggregated_bn.weight.data[bn_mask] = bn_weight_sum[bn_mask] / bn_count[bn_mask]
    aggregated_bn.bias.data[bn_mask] = bn_bias_sum[bn_mask] / bn_count[bn_mask]
    aggregated_bn.running_mean[bn_mask] = bn_running_mean_sum[bn_mask] / bn_count[bn_mask]
    aggregated_bn.running_var[bn_mask] = bn_running_var_sum[bn_mask] / bn_count[bn_mask]


def _aggregate_fc_layer(
    aggregated_fc: nn.Linear,
    models: List[nn.Module],
    masks: List[Dict],
    weights: List[float],
):
    """Aggregate the final fully connected layer from multiple pruned models.

    The FC layer input features correspond to the output of layer4.1.conv2 after pooling.
    """
    original_in_features = aggregated_fc.in_features
    original_out_features = aggregated_fc.out_features
    device = aggregated_fc.weight.device

    # Initialize accumulators
    fc_weight_sum = torch.zeros_like(aggregated_fc.weight.data)
    fc_weight_count = torch.zeros(original_out_features, original_in_features, device=device)
    fc_bias_sum = torch.zeros_like(aggregated_fc.bias.data) if aggregated_fc.bias is not None else None
    fc_bias_count = torch.zeros(original_out_features, device=device) if aggregated_fc.bias is not None else None

    # The last conv layer is layer4.1.conv2
    last_conv_mask_key = "layer4.1.conv2"

    for model, mask_dict, weight in zip(models, masks, weights):
        last_conv_indices = mask_dict[last_conv_mask_key]["indices_kept"]
        last_conv_indices = np.asarray(last_conv_indices)

        model_fc = model.fc

        # FC layer: weight shape is [out_features, in_features]
        # After adaptive avg pooling, each kept filter becomes one input feature
        fc_weight_sum[:, last_conv_indices] += weight * model_fc.weight.data
        fc_weight_count[:, last_conv_indices] += weight

        if fc_bias_sum is not None and model_fc.bias is not None:
            fc_bias_sum += weight * model_fc.bias.data
            fc_bias_count += weight

    # Apply weighted averages
    mask = fc_weight_count > 0
    aggregated_fc.weight.data[mask] = fc_weight_sum[mask] / fc_weight_count[mask]

    if fc_bias_sum is not None and fc_bias_count is not None:
        bias_mask = fc_bias_count > 0
        aggregated_fc.bias.data[bias_mask] = fc_bias_sum[bias_mask] / fc_bias_count[bias_mask]


def _get_nested_attr(obj, attr_path: str):
    """Get nested attribute from object using dot-separated path."""
    attrs = attr_path.split(".")
    for attr in attrs:
        if attr.isdigit():
            obj = obj[int(attr)]
        else:
            obj = getattr(obj, attr)
    return obj


def _get_previous_layer_indices_resnet(current_layer_name: str, mask_dict: Dict):
    """
    Get the indices of kept filters from the previous layer in ResNet18.

    Args:
        current_layer_name: Name of the current layer (e.g., "layer1.0.conv1", "layer2.1.conv2")
        mask_dict: Dictionary containing mask information for each layer

    Returns:
        np.ndarray of kept indices from the previous layer, or None if first layer
    """
    if current_layer_name == "conv1":
        # First conv layer - no previous layer (RGB input)
        return None

    # Parse the layer name
    parts = current_layer_name.split(".")

    if parts[0] in ["layer1", "layer2", "layer3", "layer4"]:
        layer_name = parts[0]
        block_idx = int(parts[1])
        conv_name = parts[2]  # "conv1" or "conv2"

        if conv_name == "conv1":
            # conv1 takes input from previous block's output (conv2)
            if block_idx == 0:
                # First block in layer - input from previous layer's last block
                if layer_name == "layer1":
                    # Input from initial conv1
                    return mask_dict["conv1"]["indices_kept"]
                elif layer_name == "layer2":
                    return mask_dict["layer1.1.conv2"]["indices_kept"]
                elif layer_name == "layer3":
                    return mask_dict["layer2.1.conv2"]["indices_kept"]
                elif layer_name == "layer4":
                    return mask_dict["layer3.1.conv2"]["indices_kept"]
            else:
                # Input from previous block's conv2 in same layer
                return mask_dict[f"{layer_name}.{block_idx - 1}.conv2"]["indices_kept"]
        elif conv_name == "conv2":
            # conv2 takes input from conv1 of the same block
            return mask_dict[f"{layer_name}.{block_idx}.conv1"]["indices_kept"]

    raise ValueError(f"Unrecognized layer name: {current_layer_name}")
