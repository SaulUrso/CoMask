from typing import Any

import numpy as np
import torch
import torch.nn as nn

from comask.prune.keywords import KEY_FILTER
from comask.prune.utils import (
    computer_conv_threshold,
    computer_weight,
    create_batchnorm2d,
    create_conv2d,
    create_linear,
    round_to_multiple_of,
)


def prune_conv(
    old_conv2d: nn.Conv2d,
    old_batchnorm2d,
    conv_threshold,
    prune_way,
    in_channels=3,
    in_idx=None,
    minimum_channels=1,
    divisor=1,
    layer_mask=None,
):
    if layer_mask is not None:
        # Use provided mask
        mask = layer_mask["mask"]
        out_idx = np.array(layer_mask["indices_kept"])
        if len(out_idx.shape) == 0:  # Handle single index case
            out_idx = np.array([out_idx])
    else:
        # Calculate mask based on threshold
        weight_copy = computer_weight(old_conv2d.weight, prune_way, (1, 2, 3))

        if (
            len(weight_copy) <= minimum_channels
        ):  # NOTE: present in original code, specifies min number of filter in a layer
            # just specify the indices of the filters maintained, which is the same as no pruning happened
            # Use actual layer size, not minimum_channels, to avoid index out of bounds
            out_idx = np.arange(len(weight_copy))
            mask = torch.ones(len(weight_copy))

        else:
            mask = weight_copy.gt(conv_threshold).float()

            # this just creates an array of the indices of ones in the mask
            # the squeeze is used to make a flat array
            out_idx: np.ndarray[Any, np.dtype[np.signedinteger[Any]]] = np.squeeze(
                np.argwhere(np.asarray(mask.cpu().numpy()))
            )
            if out_idx.size == 1:
                out_idx = np.resize(out_idx, (1,))

            # NOTE: in the original code, they also round the number of filters kept.
            # I kept it here, if you have divisor == 1 (we do) it doesn't do anything
            old_prune_len = len(out_idx)
            new_prune_len = round_to_multiple_of(old_prune_len, divisor)

            if new_prune_len > old_prune_len:  # this is skipped if divisor == 1
                temp_mask = weight_copy.le(conv_threshold).float()
                tmp_idx = np.squeeze(np.argwhere(np.asarray(temp_mask.cpu().numpy())))
                if tmp_idx.size == 1:
                    tmp_idx = np.resize(tmp_idx, (1,))
                res_idx = np.random.choice(tmp_idx, new_prune_len - old_prune_len, replace=False)

                out_idx = np.array(sorted(np.concatenate((out_idx, res_idx))))
                # Update mask to reflect the additional kept filters
                mask = torch.zeros(len(weight_copy))
                mask[out_idx] = 1.0  # type: ignore

    # Number of output channel
    out_filters = len(out_idx)

    # create new conv and copy the weights
    new_conv2d = create_conv2d(old_conv2d, in_channels, out_filters)
    new_batchnorm2d = create_batchnorm2d(old_batchnorm2d, out_filters)

    # NOTE: # notice that you are keeping all input channels, need to remove them if previous layer was pruned
    new_conv2d.weight.data = old_conv2d.weight.data[out_idx.tolist(), :, :, :].clone()
    if old_conv2d.bias is not None:
        new_conv2d.bias.data = old_conv2d.bias.data[out_idx.tolist()].clone()  # type: ignore

    # is None only for input layer, for all others it's the indices of the filters of the previous layer
    # NOTE: if previous layer not pruned -> nothing happens
    # If previous layer pruned -> only keep relevant channels
    if in_idx is not None:
        new_conv2d.weight.data = new_conv2d.weight.data[:, in_idx.tolist(), :, :].clone()

    new_batchnorm2d.weight.data = old_batchnorm2d.weight.data[out_idx.tolist()].clone()
    new_batchnorm2d.bias.data = old_batchnorm2d.bias.data[out_idx.tolist()].clone()
    new_batchnorm2d.running_mean = old_batchnorm2d.running_mean[out_idx.tolist()].clone()
    new_batchnorm2d.running_var = old_batchnorm2d.running_var[out_idx.tolist()].clone()

    named_mask = {
        "mask": mask,
        "layer_type": "conv",
        "original_filters": old_conv2d.out_channels,
        "pruned_filters": out_filters,
        "indices_kept": out_idx,
    }

    return new_conv2d, new_batchnorm2d, out_filters, out_idx, named_mask


def prune_basic_block(
    old_block,
    conv_threshold,
    prune_way,
    in_channels,
    in_idx=None,
    minimum_channels=1,
    divisor=1,
    with_mask=None,
):
    """Prune a BasicBlock while maintaining skip connections"""

    # create mask
    block_masks = {}

    # Prune conv1
    conv1_layer_mask = with_mask["conv1"] if with_mask else None
    new_conv1, new_bn1, out_channels_1, out_idx_1, mask_1 = prune_conv(
        old_block.conv1,
        old_block.bn1,
        conv_threshold,
        prune_way,
        in_channels=in_channels,
        in_idx=in_idx,
        minimum_channels=minimum_channels,
        divisor=divisor,
        layer_mask=conv1_layer_mask,
    )

    # store mast first conv
    block_masks["conv1"] = mask_1

    # Prune conv2 - input channels match conv1 output
    conv2_layer_mask = with_mask["conv2"] if with_mask else None
    new_conv2, new_bn2, out_channels_2, out_idx_2, mask_2 = prune_conv(
        old_block.conv2,
        old_block.bn2,
        conv_threshold,
        prune_way,
        in_channels=out_channels_1,
        in_idx=out_idx_1,
        minimum_channels=minimum_channels,
        divisor=divisor,
        layer_mask=conv2_layer_mask,
    )

    block_masks["conv2"] = mask_2

    # Handle downsample if present
    new_downsample = None
    if old_block.downsample is not None:
        downsample_conv = old_block.downsample[0]
        downsample_bn = old_block.downsample[1]

        # Create new downsample layers with adjusted dimensions
        new_downsample_conv = create_conv2d(downsample_conv, in_channels, out_channels_2)
        new_downsample_bn = create_batchnorm2d(downsample_bn, out_channels_2)

        # Copy weights with proper indexing
        new_downsample_conv.weight.data = downsample_conv.weight.data[out_idx_2.tolist(), :, :, :].clone()
        if downsample_conv.bias is not None:
            assert new_downsample_conv.bias is not None
            new_downsample_conv.bias.data = downsample_conv.bias.data[out_idx_2.tolist()].clone()

        if in_idx is not None:
            new_downsample_conv.weight.data = new_downsample_conv.weight.data[:, in_idx.tolist(), :, :].clone()

        # Copy batchnorm weights
        new_downsample_bn.weight.data = downsample_bn.weight.data[out_idx_2.tolist()].clone()
        new_downsample_bn.bias.data = downsample_bn.bias.data[out_idx_2.tolist()].clone()
        new_downsample_bn.running_mean = downsample_bn.running_mean[out_idx_2.tolist()].clone()
        new_downsample_bn.running_var = downsample_bn.running_var[out_idx_2.tolist()].clone()

        new_downsample = nn.Sequential(new_downsample_conv, new_downsample_bn)

        # Assert downsample conv has correct input and output channels
        assert new_downsample_conv.in_channels == in_channels, (
            f"Downsample conv in_channels mismatch: expected {(len(in_idx) if in_idx is not None else in_channels)}, got {new_downsample_conv.in_channels}"
        )
        assert new_downsample_conv.out_channels == out_channels_2, (
            f"Downsample conv out_channels mismatch: expected {out_channels_2}, got {new_downsample_conv.out_channels}"
        )

    # Create new BasicBlock
    from fedml.model.cv.resnet_cifar import BasicBlock

    # For ResNet, the 'planes' parameter should be the output channels of conv2
    # This ensures identity and main path have matching dimensions
    new_block = BasicBlock(in_channels, out_channels_2, stride=old_block.conv1.stride[0], downsample=new_downsample)

    # Replace layers
    new_block.conv1 = new_conv1
    new_block.bn1 = new_bn1
    new_block.conv2 = new_conv2
    new_block.bn2 = new_bn2
    new_block.relu = nn.ReLU(inplace=True)

    return new_block, out_channels_2, out_idx_2, block_masks


def prune_resnet_features(model, conv_threshold, prune_way, minimum_channels=1, divisor=1, with_mask=None):
    """Prune ResNet feature layers"""
    masks = {}

    # Prune initial conv layer
    conv1_layer_mask = with_mask["conv1"] if with_mask else None
    new_conv1, new_bn1, out_channels, out_idx, mask = prune_conv(
        model.conv1,
        model.bn1,
        conv_threshold,
        prune_way,
        in_channels=3,  # RGB input
        in_idx=None,
        minimum_channels=minimum_channels,
        divisor=divisor,
        layer_mask=conv1_layer_mask,
    )

    model.conv1 = new_conv1
    model.bn1 = new_bn1
    masks["conv1"] = mask

    current_channels = out_channels
    current_idx = out_idx

    # Prune each layer
    for layer_name in ["layer1", "layer2", "layer3", "layer4"]:
        layer = getattr(model, layer_name)
        new_blocks = []

        for block_idx, block in enumerate(layer):
            # Create block-specific mask dictionary
            block_mask_key = f"{layer_name}.{block_idx}"
            block_mask = None
            if with_mask:
                # Extract masks for this specific block
                block_mask = {}
                for conv_name in ["conv1", "conv2"]:
                    mask_key = f"{block_mask_key}.{conv_name}"
                    block_mask[conv_name] = with_mask[mask_key]
                if not block_mask:  # If no masks found for this block
                    raise Exception(f"No masks found for block {block_idx}")

            new_block, current_channels, current_idx, block_masks = prune_basic_block(
                block,
                conv_threshold,
                prune_way,
                current_channels,
                current_idx,
                minimum_channels,
                divisor,
                with_mask=block_mask,
            )
            new_blocks.append(new_block)

            # Store masks
            for conv_name, block_mask in block_masks.items():
                mask_key = f"{layer_name}.{block_idx}.{conv_name}"
                masks[mask_key] = block_mask

        # Replace the layer with new blocks
        setattr(model, layer_name, nn.Sequential(*new_blocks))

    return model, current_channels, current_idx, masks


def prune_resnet_classifier(model, in_channels, in_idx):
    """Adjust final linear layer for pruned features"""
    old_fc = model.fc

    # due to pooling you actually don't have to do anything

    # Create new linear layer with adjusted input size
    new_fc, _ = create_linear(old_fc, in_channels)
    # Copy weights and bias (no pruning for final layer)
    new_fc.weight.data = old_fc.weight.data[:, in_idx].clone()
    new_fc.bias.data = old_fc.bias.data.clone()

    model.fc = new_fc
    return model


def prune(model, percent, prune_way="mean_abs", minimum_channels=1, divisor=1, with_mask=None):
    """Main function to prune ResNet model"""
    # Calculate threshold (only if not using provided mask)
    total_groups = None
    group_pruning_ratio = None
    units_pruned = 0

    if with_mask is None:
        total_groups, threshold, planned_units_pruned = computer_conv_threshold(
            model, percent, prune_type=KEY_FILTER, prune_way=prune_way, ceil=True
        )
        units_pruned = planned_units_pruned
    else:
        # When using provided mask, threshold is not used
        threshold = None
        # Calculate units pruned from mask
        units_pruned = sum(
            mask_info["original_filters"] - mask_info["pruned_filters"] for mask_info in with_mask.values()
        )

    # Calculate original parameters for comparison
    from comask.prune.utils import count_parameters

    total_params_before = count_parameters(model)

    # Prune features
    model, final_channels, final_idx, masks = prune_resnet_features(
        model, threshold, prune_way, minimum_channels, divisor, with_mask
    )

    # Adjust classifier (no pruning, just dimension adjustment)
    model = prune_resnet_classifier(model, final_channels, final_idx)

    # Calculate final statistics
    if with_mask is None:
        new_total_groups, _, _ = computer_conv_threshold(model, percent, prune_type=KEY_FILTER, prune_way=prune_way)
        group_pruning_ratio = (total_groups - new_total_groups) / total_groups if total_groups is not None else None
    else:
        # Recalculate actual units pruned from resulting masks
        units_pruned = sum(mask_info["original_filters"] - mask_info["pruned_filters"] for mask_info in masks.values())

    total_params_after = count_parameters(model)
    param_pruning_ratio = (total_params_before - total_params_after) / total_params_before

    return model, param_pruning_ratio, group_pruning_ratio, threshold, masks, units_pruned
