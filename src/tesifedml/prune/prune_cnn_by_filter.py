import numpy as np
import torch
import torch.nn as nn

from .keywords import KEY_FILTER
from .utils import (
    computer_conv_threshold,
    computer_weight,
    count_parameters,
    create_batchnorm2d,
    create_conv2d,
    create_linear,
    round_to_multiple_of,
    set_module_list,
)

# TODO: the conv assumes to have a bias, but you may want to remove it i guess (or not, since you already did the experiments)


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
        weight_copy = computer_weight(old_conv2d.weight, prune_way, (1, 2, 3))

        if (
            len(weight_copy) <= minimum_channels
        ):  # NOTE: present in original code, specifies min number of filter in a layer
            # just specify the indices of the filters maintained, which is the same as no pruning hapened
            out_idx = np.arange(minimum_channels)
            mask = torch.ones(len(weight_copy))

        else:
            mask = weight_copy.gt(conv_threshold).float()

            # this just creates an array of the indices of ones in the mask
            # the squeeze is used to make a flat array
            out_idx = np.squeeze(np.argwhere(np.asarray(mask.cpu().numpy())))
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
                mask[out_idx] = 1.0

    # Number of output channel
    out_filters = len(out_idx)

    # create new conv and copy the weights
    new_conv2d = create_conv2d(old_conv2d, in_channels, out_filters)
    new_batchnorm2d = create_batchnorm2d(old_batchnorm2d, out_filters)

    # NOTE: # notice that you are keeping all input channels, need to remove them if previous layer was pruned
    new_conv2d.weight.data = old_conv2d.weight.data[out_idx.tolist(), :, :, :].clone()
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

    return new_conv2d, new_batchnorm2d, out_filters, out_idx, mask


def prune_features(module_list, conv_threshold, prune_way, minimum_channels=1, divisor=1, with_mask=None):
    new_module_list = list()
    masks = {}  # Changed from list to dict
    idx = 0
    layer_idx = 0  # Track layer index for naming

    # NOTE: for HARBox_CNN, input channels is 1
    in_channels = 1
    in_idx = None

    while idx < len(module_list):
        if isinstance(module_list[idx], nn.Sequential):
            # Handle sequential blocks containing Conv2d + BatchNorm2d
            seq_block = module_list[idx]
            conv_layer = seq_block[0]  # Conv2d
            bn_layer = seq_block[1]  # BatchNorm2d

            assert isinstance(conv_layer, nn.Conv2d), f"Expected Conv2d, got {type(conv_layer)}"
            assert isinstance(bn_layer, nn.BatchNorm2d), f"Expected BatchNorm2d, got {type(bn_layer)}"

            # Get layer mask if provided
            layer_name = f"conv_{layer_idx}"
            layer_mask = None
            if with_mask is not None:
                print(conv_layer)
                # Find the corresponding mask by looking through all mask keys
                for mask_key, mask_data in with_mask.items():
                    if mask_key == layer_name:
                        layer_mask = mask_data
                        break

                assert layer_mask is not None

            new_conv2d, new_batchnorm2d, in_channels, in_idx, mask = prune_conv(
                conv_layer,
                bn_layer,
                conv_threshold,
                prune_way,
                in_channels=in_channels,
                in_idx=in_idx,
                minimum_channels=minimum_channels,
                divisor=divisor,
                layer_mask=layer_mask,
            )

            # Store mask information with layer name
            masks[layer_name] = {
                "mask": mask,
                "layer_type": "conv",
                "original_filters": conv_layer.out_channels,
                "pruned_filters": in_channels,
                "indices_kept": in_idx,
            }
            layer_idx += 1

            # Create new sequential block
            new_seq_block = nn.Sequential(new_conv2d, new_batchnorm2d)
            new_module_list.append(new_seq_block)
            # NOTE: hte way i iterate, in the list i get a reference to the sequential,
            # and then a reference to each of its submodules, that is why i need to replicate them
            new_module_list.append(new_conv2d)
            new_module_list.append(new_batchnorm2d)
            idx += 3

        elif isinstance(module_list[idx], nn.Conv2d):
            # Handle standalone Conv2d layers (if any)
            raise Exception(f"found module {module_list[idx]} as conv alone, but there should not be any")

        elif isinstance(module_list[idx], nn.MaxPool2d):
            new_module_list.append(module_list[idx])
            idx += 1
        else:
            # Skip other layers (like Linear, which will be handled separately)
            new_module_list.append(module_list[idx])
            idx += 1

    return new_module_list, in_channels, in_idx, masks


def prune_linear(old_linear: nn.Linear, threshold, prune_way, in_cols, in_idx, min_cols=1, divisor=1):
    weight_copy = computer_weight(old_linear.weight, prune_way=prune_way, dimension=1)

    if len(weight_copy) <= min_cols:  # NOTE: present in original code, specifies min number of filter in a layer
        # just specify the indices of the filters maintained, which is the same as no pruning hapened
        out_idx = np.arange(min_cols)

    else:
        mask = weight_copy.gt(threshold).float()
        # print(f"MASK: {mask}")

        out_idx = np.squeeze(np.argwhere(np.asarray(mask.cpu().numpy())))
        if out_idx.size == 1:
            out_idx = np.resize(out_idx, (1,))

        # This was copied from the filter pruning code. Since the divisor for me is always
        # == 1 i did not test this specific part, but it should work nonetheless
        old_prune_len = len(out_idx)
        new_prune_len = round_to_multiple_of(old_prune_len, divisor)

        if new_prune_len > old_prune_len:  # this is skipped if divisor == 1
            mask = weight_copy.le(threshold).float()
            tmp_idx = np.squeeze(np.argwhere(np.asarray(mask.cpu().numpy())))
            if tmp_idx.size == 1:
                tmp_idx = np.resize(tmp_idx, (1,))
            res_idx = np.random.choice(tmp_idx, new_prune_len - old_prune_len, replace=False)

            out_idx = np.array(sorted(np.concatenate((out_idx, res_idx))))

        # print(out_idx)
        # print(new_prune_len)

    # Number of output channel
    out_cols = len(out_idx)

    new_linear, _ = create_linear(old_linear, in_cols, out_cols)

    new_linear.weight.data = old_linear.weight.data[out_idx.tolist(), :].clone()
    new_linear.bias.data = old_linear.bias.data[out_idx.tolist()].clone()

    new_linear.weight.data = new_linear.weight.data[:, in_idx.tolist()].clone()

    return new_linear, out_cols, out_idx


def prune_classifier(module_list, in_channels, in_idx):
    new_module_list = list()

    # For HARBox_CNN: last conv layer has 32 channels and output size is 8x8
    # So flattened size is 32 * 8 * 8 = 2048
    in_idx_fc = torch.arange(32 * 8 * 8).reshape(32, 8, 8)[in_idx, :, :].reshape(-1)

    # For HARBox_CNN, there's only one linear layer (no hidden layers to prune)
    # So we just create the final layer with pruned inputs
    old_linear = module_list[0]
    new_linear, _ = create_linear(old_linear, in_channels)
    new_linear.weight.data = old_linear.weight.data[:, in_idx_fc].clone()
    new_linear.bias.data = old_linear.bias.data.clone()
    new_module_list.append(new_linear)

    return new_module_list


def prune(model, percent, prune_way="mean_abs", minimum_channels=1, divisor=1, with_mask=None):
    # Calculate threshold (only if not using provided mask)
    total_groups = None
    if with_mask is None:
        total_groups, threshold = computer_conv_threshold(model, percent, prune_type=KEY_FILTER, prune_way=prune_way)
    else:
        threshold = None

    total_params_before = count_parameters(model)

    feature_name_list = list()
    feature_module_list = list()
    classifier_name_list = list()
    classifier_module_list = list()

    for name, module in model.named_modules():
        if "linear" in name:  # Linear layer -> classifier
            classifier_name_list.append(f"{name}")
            classifier_module_list.append(module)
        elif "conv" in name:  # Convolutional blocks -> features
            feature_name_list.append(f"{name}")
            feature_module_list.append(module)
        elif name == "":
            continue

    new_module_list, in_channels, in_idx, masks = prune_features(
        feature_module_list,
        conv_threshold=threshold,
        prune_way=prune_way,
        minimum_channels=minimum_channels,
        divisor=divisor,
        with_mask=with_mask,
    )

    # # Associate actual layer names with masks
    # named_masks = {}
    # conv_layer_count = 0
    # for name in feature_name_list:
    #     if "conv" in name and name.endswith(".0"):  # Only for actual conv layers (not sequential or batchnorm)
    #         mask_key = f"conv_{conv_layer_count}"
    #         named_masks[name] = masks[mask_key]
    #         conv_layer_count += 1

    assert len(new_module_list) == len(feature_module_list) == len(feature_name_list)
    set_module_list(model, feature_name_list, feature_module_list, new_module_list)

    # For HARBox_CNN: in_channels * 8 * 8 (since conv2 output is 8x8)
    new_module_list = prune_classifier(
        classifier_module_list,
        in_channels * 8 * 8,
        in_idx,
    )
    assert len(new_module_list) == len(classifier_module_list) == len(classifier_name_list)
    set_module_list(model, classifier_name_list, classifier_module_list, new_module_list)

    # Calculate final statistics
    group_pruning_ratio = None
    if with_mask is None:
        new_total_groups, _ = computer_conv_threshold(model, percent, prune_type=KEY_FILTER, prune_way=prune_way)

        group_pruning_ratio = (total_groups - new_total_groups) / total_groups

    total_params_after = count_parameters(model)

    # Calculate parameter-based pruning ratio
    param_pruning_ratio = (total_params_before - total_params_after) / total_params_before

    return model, param_pruning_ratio, group_pruning_ratio, threshold, masks
