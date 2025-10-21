import numpy as np
import torch
import torch.nn as nn

from .keywords import KEY_FILTER
from .utils import (
    computer_conv_threshold,
    computer_weight,
    create_batchnorm2d,
    create_conv2d,
    create_linear,
    round_to_multiple_of,
)


def prune_pointwise_conv(
    old_conv2d: nn.Conv2d,
    old_batchnorm2d,
    conv_threshold,
    prune_way,
    in_channels=3,
    in_idx=None,
    minimum_channels=1,
    divisor=1,
):
    """Prune pointwise convolution (1x1 conv)"""
    weight_copy = computer_weight(old_conv2d.weight, prune_way, (1, 2, 3))

    if len(weight_copy) <= minimum_channels:
        out_idx = np.arange(minimum_channels)
        mask = torch.ones(len(weight_copy))
    else:
        mask = weight_copy.gt(conv_threshold).float()
        out_idx = np.squeeze(np.argwhere(np.asarray(mask.cpu().numpy())))
        if out_idx.size == 1:
            out_idx = np.resize(out_idx, (1,))

        old_prune_len = len(out_idx)
        new_prune_len = round_to_multiple_of(old_prune_len, divisor)

        if new_prune_len > old_prune_len:
            temp_mask = weight_copy.le(conv_threshold).float()
            tmp_idx = np.squeeze(np.argwhere(np.asarray(temp_mask.cpu().numpy())))
            if tmp_idx.size == 1:
                tmp_idx = np.resize(tmp_idx, (1,))
            res_idx = np.random.choice(tmp_idx, new_prune_len - old_prune_len, replace=False)

            out_idx = np.array(sorted(np.concatenate((out_idx, res_idx))))
            mask = torch.zeros(len(weight_copy))
            mask[out_idx] = 1.0

    out_filters = len(out_idx)

    # Create new layers
    new_conv2d = create_conv2d(old_conv2d, in_channels, out_filters)
    new_batchnorm2d = create_batchnorm2d(old_batchnorm2d, out_filters)

    # Copy weights
    new_conv2d.weight.data = old_conv2d.weight.data[out_idx.tolist(), :, :, :].clone()
    if old_conv2d.bias is not None:
        new_conv2d.bias.data = old_conv2d.bias.data[out_idx.tolist()].clone()

    if in_idx is not None:
        new_conv2d.weight.data = new_conv2d.weight.data[:, in_idx.tolist(), :, :].clone()

    # Copy batchnorm parameters
    new_batchnorm2d.weight.data = old_batchnorm2d.weight.data[out_idx.tolist()].clone()
    new_batchnorm2d.bias.data = old_batchnorm2d.bias.data[out_idx.tolist()].clone()
    new_batchnorm2d.running_mean = old_batchnorm2d.running_mean[out_idx.tolist()].clone()
    new_batchnorm2d.running_var = old_batchnorm2d.running_var[out_idx.tolist()].clone()

    return new_conv2d, new_batchnorm2d, out_filters, out_idx, mask


def adapt_depthwise_conv(old_conv2d: nn.Conv2d, old_batchnorm2d, new_channels, in_idx):
    """Adapt depthwise convolution to match new channel count (no pruning, just adaptation)"""
    # Assert that this is indeed a depthwise convolution
    assert old_conv2d.groups == old_conv2d.in_channels == old_conv2d.out_channels, (
        f"Expected depthwise conv with groups=in_channels=out_channels, "
        f"got groups={old_conv2d.groups}, in_channels={old_conv2d.in_channels}, out_channels={old_conv2d.out_channels}"
    )

    # Assert weight shape is correct for depthwise conv
    expected_weight_shape = (old_conv2d.out_channels, 1, old_conv2d.kernel_size[0], old_conv2d.kernel_size[1])
    assert old_conv2d.weight.data.shape == expected_weight_shape, (
        f"Expected depthwise conv weight shape {expected_weight_shape}, got {old_conv2d.weight.data.shape}"
    )

    # For depthwise conv, input_channels == output_channels == groups
    new_conv2d = create_conv2d(old_conv2d, new_channels, new_channels, old_groups=new_channels)
    new_batchnorm2d = create_batchnorm2d(old_batchnorm2d, new_channels)

    # Copy weights for kept channels only
    new_conv2d.weight.data = old_conv2d.weight.data[in_idx.tolist(), :, :, :].clone()

    # Assert new weight shape is correct
    expected_new_weight_shape = (new_channels, 1, old_conv2d.kernel_size[0], old_conv2d.kernel_size[1])
    assert new_conv2d.weight.data.shape == expected_new_weight_shape, (
        f"Expected new depthwise conv weight shape {expected_new_weight_shape}, got {new_conv2d.weight.data.shape}"
    )

    if old_conv2d.bias is not None:
        new_conv2d.bias.data = old_conv2d.bias.data[in_idx.tolist()].clone()

    # Copy batchnorm parameters
    new_batchnorm2d.weight.data = old_batchnorm2d.weight.data[in_idx.tolist()].clone()
    new_batchnorm2d.bias.data = old_batchnorm2d.bias.data[in_idx.tolist()].clone()
    new_batchnorm2d.running_mean = old_batchnorm2d.running_mean[in_idx.tolist()].clone()
    new_batchnorm2d.running_var = old_batchnorm2d.running_var[in_idx.tolist()].clone()

    return new_conv2d, new_batchnorm2d


def prune_basic_conv2d(
    old_block,
    conv_threshold,
    prune_way,
    in_channels,
    in_idx=None,
    minimum_channels=1,
    divisor=1,
):
    """Prune a BasicConv2d block"""
    new_conv, new_bn, out_channels, out_idx, mask = prune_pointwise_conv(
        old_block.conv,
        old_block.bn,
        conv_threshold,
        prune_way,
        in_channels=in_channels,
        in_idx=in_idx,
        minimum_channels=minimum_channels,
        divisor=divisor,
    )

    # Create new BasicConv2d block
    from tesifedml.models.mobilenet import BasicConv2d

    new_block = BasicConv2d(in_channels, out_channels, old_block.conv.kernel_size[0])
    new_block.conv = new_conv
    new_block.bn = new_bn
    new_block.relu = nn.ReLU(inplace=True)

    return new_block, out_channels, out_idx, mask


def prune_depth_separable_conv2d(
    old_block,
    conv_threshold,
    prune_way,
    in_channels,
    in_idx=None,
    minimum_channels=1,
    divisor=1,
):
    """Prune a DepthSeparableConv2d block - only prune the pointwise conv"""
    # Assert that depthwise conv has correct structure
    depthwise_conv = old_block.depthwise[0]
    assert isinstance(depthwise_conv, nn.Conv2d), f"Expected Conv2d in depthwise, got {type(depthwise_conv)}"
    assert depthwise_conv.groups == depthwise_conv.in_channels == depthwise_conv.out_channels, (
        f"Depthwise conv should have groups=in_channels=out_channels, "
        f"got groups={depthwise_conv.groups}, in_channels={depthwise_conv.in_channels}, out_channels={depthwise_conv.out_channels}"
    )

    # Assert that pointwise conv is 1x1
    pointwise_conv = old_block.pointwise[0]
    assert isinstance(pointwise_conv, nn.Conv2d), f"Expected Conv2d in pointwise, got {type(pointwise_conv)}"
    assert pointwise_conv.kernel_size == (1, 1), f"Pointwise conv should be 1x1, got {pointwise_conv.kernel_size}"

    # Adapt depthwise convolution (no pruning, just channel adjustment)
    new_depthwise_conv, new_depthwise_bn = adapt_depthwise_conv(
        old_block.depthwise[0], old_block.depthwise[1], in_channels, in_idx
    )

    # Assert adapted depthwise conv has correct properties
    assert (
        new_depthwise_conv.groups == new_depthwise_conv.in_channels == new_depthwise_conv.out_channels == in_channels
    ), (
        f"Adapted depthwise conv should have groups=in_channels=out_channels={in_channels}, "
        f"got groups={new_depthwise_conv.groups}, in_channels={new_depthwise_conv.in_channels}, out_channels={new_depthwise_conv.out_channels}"
    )

    # Prune pointwise convolution
    new_pointwise_conv, new_pointwise_bn, out_channels, out_idx, mask = prune_pointwise_conv(
        old_block.pointwise[0],
        old_block.pointwise[1],
        conv_threshold,
        prune_way,
        in_channels=in_channels,
        in_idx=in_idx,
        minimum_channels=minimum_channels,
        divisor=divisor,
    )

    # Create new DepthSeparableConv2d block
    from tesifedml.models.mobilenet import DepthSeperabelConv2d

    new_block = DepthSeperabelConv2d(
        in_channels,
        out_channels,
        old_block.depthwise[0].kernel_size[0],
        stride=old_block.depthwise[0].stride,
        padding=old_block.depthwise[0].padding,
        bias=False,
    )

    # Replace the internal layers
    new_block.depthwise = nn.Sequential(new_depthwise_conv, new_depthwise_bn, nn.ReLU(inplace=True))

    new_block.pointwise = nn.Sequential(new_pointwise_conv, new_pointwise_bn, nn.ReLU(inplace=True))

    return new_block, out_channels, out_idx, mask


def prune_features(model, conv_threshold, prune_way, minimum_channels=1, divisor=1):
    """Prune MobileNet features"""
    masks = {}
    layer_idx = 0

    # Start with input channels (grayscale)
    in_channels = 1
    in_idx = None

    # Prune stem (BasicConv2d + DepthSeparableConv2d)
    # Prune BasicConv2d in stem
    new_stem_0, in_channels, in_idx, mask = prune_basic_conv2d(
        model.stem[0],
        conv_threshold,
        prune_way,
        in_channels=in_channels,
        in_idx=in_idx,
        minimum_channels=minimum_channels,
        divisor=divisor,
    )
    masks["stem.0.conv"] = {
        "mask": mask,
        "layer_type": "conv",
        "original_filters": model.stem[0].conv.out_channels,
        "pruned_filters": in_channels,
        "indices_kept": in_idx,
    }

    # Prune DepthSeparableConv2d in stem
    new_stem_1, in_channels, in_idx, mask = prune_depth_separable_conv2d(
        model.stem[1],
        conv_threshold,
        prune_way,
        in_channels=in_channels,
        in_idx=in_idx,
        minimum_channels=minimum_channels,
        divisor=divisor,
    )
    masks["stem.1.pointwise.0"] = {
        "mask": mask,
        "layer_type": "conv",
        "original_filters": model.stem[1].pointwise[0].out_channels,
        "pruned_filters": in_channels,
        "indices_kept": in_idx,
    }

    model.stem = nn.Sequential(new_stem_0, new_stem_1)

    # Prune conv1, conv2, conv3, conv4
    for conv_block_name in ["conv1", "conv2", "conv3", "conv4"]:
        conv_block = getattr(model, conv_block_name)
        new_blocks = []

        for block_idx, block in enumerate(conv_block):
            new_block, in_channels, in_idx, mask = prune_depth_separable_conv2d(
                block,
                conv_threshold,
                prune_way,
                in_channels=in_channels,
                in_idx=in_idx,
                minimum_channels=minimum_channels,
                divisor=divisor,
            )
            new_blocks.append(new_block)

            # Store mask
            mask_key = f"{conv_block_name}.{block_idx}.pointwise.0"
            masks[mask_key] = {
                "mask": mask,
                "layer_type": "conv",
                "original_filters": block.pointwise[0].out_channels,
                "pruned_filters": in_channels,
                "indices_kept": in_idx,
            }

        setattr(model, conv_block_name, nn.Sequential(*new_blocks))

    return model, in_channels, in_idx, masks


def prune_classifier(model, in_channels, in_idx):
    """Adjust classifier for pruned features"""
    old_fc = model.fc

    # Create new linear layer with adjusted input size
    new_fc, _ = create_linear(old_fc, in_channels)

    # Copy weights (only for kept channels)
    new_fc.weight.data = old_fc.weight.data[:, in_idx].clone()
    new_fc.bias.data = old_fc.bias.data.clone()

    model.fc = new_fc
    return model


def prune(model, percent, prune_way="mean_abs", minimum_channels=1, divisor=1):
    """Main function to prune MobileNet"""
    # Calculate threshold
    total, total_params_before, threshold = computer_conv_threshold(
        model, percent, prune_type=KEY_FILTER, prune_way=prune_way
    )

    # Prune features
    model, final_channels, final_idx, masks = prune_features(model, threshold, prune_way, minimum_channels, divisor)

    # Adjust classifier
    model = prune_classifier(model, final_channels, final_idx)

    # Calculate final statistics
    new_total, total_params_after, _ = computer_conv_threshold(
        model, percent, prune_type=KEY_FILTER, prune_way=prune_way
    )

    param_pruning_ratio = (total_params_before - total_params_after) / total_params_before

    return model, param_pruning_ratio, threshold, masks
