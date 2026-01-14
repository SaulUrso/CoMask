import numpy as np
import torch
import torch.nn as nn

from .keywords import KEY_FILTER
from .utils import (
    computer_conv_threshold,
    computer_weight,
    create_conv2d,
    create_linear,
    round_to_multiple_of,
    set_module_list,
)


def prune_conv(
    old_conv2d: nn.Conv2d, conv_threshold, prune_way, in_channels=3, in_idx=None, minimum_channels=1, divisor=1
):
    weight_copy = computer_weight(old_conv2d.weight, prune_way, (1, 2, 3))
    print(f"Layer: f{old_conv2d}")
    print(weight_copy)

    if (
        len(weight_copy) <= minimum_channels
    ):  # NOTE: present in original code, specifies min number of filter in a layer
        # just specify the indices of the filters maintained, which is the same as no pruning hapened
        out_idx = np.arange(minimum_channels)

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
            mask = weight_copy.le(conv_threshold).float()
            tmp_idx = np.squeeze(np.argwhere(np.asarray(mask.cpu().numpy())))
            if tmp_idx.size == 1:
                tmp_idx = np.resize(tmp_idx, (1,))
            res_idx = np.random.choice(tmp_idx, new_prune_len - old_prune_len, replace=False)

            out_idx = np.array(sorted(np.concatenate((out_idx, res_idx))))

        print(out_idx)
        print(new_prune_len)

    # Number of output channel
    out_filters = len(out_idx)

    # create new conv and copy the weights
    new_conv2d = create_conv2d(old_conv2d, in_channels, out_filters)
    # NOTE: # notice that you are keeping all input channels, need to remove them if previous layer was pruned
    new_conv2d.weight.data = old_conv2d.weight.data[out_idx.tolist(), :, :, :].clone()
    new_conv2d.bias.data = old_conv2d.bias.data[out_idx.tolist()].clone()  # type: ignore

    # is None only for input layer, for all others it's the indices of the filters of the previous layer
    # NOTE: if previous layer not pruned -> nothing happens
    # If previous layer pruned -> only keep relevant channels
    if in_idx is not None:
        print(f"Weights before in_idx assignment: {new_conv2d.weight.data.shape}")
        new_conv2d.weight.data = new_conv2d.weight.data[:, in_idx.tolist(), :, :].clone()
        print(f"Weights after in_idx assignment: {new_conv2d.weight.data.shape}")

    print(f"NEW CONV LAYER: {new_conv2d}")
    return new_conv2d, out_filters, out_idx


def prune_features(module_list, conv_threshold, prune_way, minimum_channels=1, divisor=1):
    new_module_list = list()
    idx = 0

    # NOTE: purely depends on model, for our case is fine.
    # This is only used for the input, as the channels change one you prune the previous layer
    in_channels = 3

    in_idx = None

    while idx < len(module_list):
        if isinstance(module_list[idx], nn.Conv2d):
            # in this case, we only advance by one, as the CNN_WEB of fedML only has the conv
            # due to them using the relu as a functional operation, and not a layer,
            # and the CNN does not use batch normalization
            new_conv2d, in_channels, in_idx = prune_conv(
                module_list[idx],
                conv_threshold,
                prune_way,
                in_channels=in_channels,
                in_idx=in_idx,
                minimum_channels=minimum_channels,
                divisor=divisor,
            )
            new_module_list.append(new_conv2d)
            idx += 1

        elif isinstance(module_list[idx], nn.MaxPool2d):
            new_module_list.append(module_list[idx])
            idx += 1

    return new_module_list, in_channels, in_idx


def prune_linear(old_linear: nn.Linear, threshold, prune_way, in_cols, in_idx, min_cols=1, divisor=1):
    weight_copy = computer_weight(old_linear.weight, prune_way=prune_way, dimension=1)


    if len(weight_copy) <= min_cols:  # NOTE: present in original code, specifies min number of filter in a layer
        # just specify the indices of the filters maintained, which is the same as no pruning hapened
        out_idx = np.arange(min_cols)

    else:
        mask = weight_copy.gt(threshold).float()
        print(f"MASK: {mask}")

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

        print(out_idx)
        print(new_prune_len)

    # Number of output channel
    out_cols = len(out_idx)

    new_linear, _ = create_linear(old_linear, in_cols, out_cols)

    new_linear.weight.data = old_linear.weight.data[out_idx.tolist(), :].clone()
    new_linear.bias.data = old_linear.bias.data[out_idx.tolist()].clone()

    new_linear.weight.data = new_linear.weight.data[:, in_idx.tolist()].clone()

    print(f"LINEAR WEIGHT SHAPE{new_linear.weight.data.shape}")

    return new_linear, out_cols, out_idx


def prune_classifier(module_list, in_channels, in_idx, threshold, prune_way="mean_abs", minimum_channels=1, divisor=1):
    # TODO: change to allow for any dimension, the arange is not good like this
    new_module_list = list()

    # Creating first fc layer. Remember that everything gets reshaped, so we get the indices of the weights
    # associated to the conv filters in the original form and then flatten array for fc layer
    # new_idx.shape = [in_channels * 5 * 5]
    # in_idx.shape = [in_channels]

    in_idx_fc = torch.arange(16 * 5 * 5).reshape(16, 5, 5)[in_idx, :, :].reshape(-1)

    print("IND idx")
    print(in_idx)
    print("NEW_IDX")
    print(in_idx_fc)

    idx = 0

    # exclude last layer from pruning
    while idx < len(module_list) - 1:
        if isinstance(module_list[idx], nn.Linear):
            new_linear, in_channels, in_idx_fc = prune_linear(
                module_list[idx],
                threshold,
                prune_way,
                in_channels,
                in_idx_fc,
                min_cols=minimum_channels,
                divisor=divisor,
            )
            new_module_list.append(new_linear)
            idx += 1

        else:
            raise ValueError(f"Module {module_list[idx]} not supported")

    # create last layer, which is not pruned
    old_last: nn.Linear = module_list[-1]
    new_last, _ = create_linear(old_last, in_channels)
    new_last.weight.data = old_last.weight.data[:, in_idx_fc].clone()
    new_last.bias.data = old_last.bias.data.clone()
    new_module_list.append(new_last)

    return new_module_list


def prune(model, percent, prune_way="mean_abs", minimum_channels=1, divisor=1):
    total, threshold = computer_conv_threshold(model, percent, prune_type=KEY_FILTER, prune_way=prune_way)

    # model = list(model.children())[0] # not necessary for fedML lenet, as it does not have a wrapper
    feature_name_list = list()
    feature_module_list = list()
    classifier_name_list = list()
    classifier_module_list = list()

    for name, module in model.named_modules():
        if "fc" in name:  # fully connected layer -> classifier
            classifier_name_list.append(f"{name}")
            classifier_module_list.append(module)

        elif name == "":
            continue

        else:  # NOTE: assuming either batchnorm,conv,pool. RELU is used as func, so no layer.
            # TODO: change in case you use lstms, but for CNN is okay.
            feature_name_list.append(f"{name}")
            feature_module_list.append(module)

    print(feature_name_list)
    print(feature_module_list)
    print("-----")
    print(classifier_name_list)
    print(classifier_module_list)

    new_module_list, in_channels, in_idx = prune_features(
        feature_module_list,
        conv_threshold=threshold,
        prune_way=prune_way,
        minimum_channels=minimum_channels,
        divisor=divisor,
    )

    assert len(new_module_list) == len(feature_module_list) == len(feature_name_list)
    set_module_list(model, feature_name_list, feature_module_list, new_module_list)

    print("NEW MODEL AFTER FILTER PRUNING:")
    print(model)

    # now we need to readapt classifier
    # TODO: change this to allow for any filter dimension.
    new_module_list = prune_classifier(
        classifier_module_list,
        in_channels * 5 * 5,
        in_idx,
        threshold=threshold,
        prune_way=prune_way,
        minimum_channels=minimum_channels,
        divisor=divisor,
    )
    assert len(new_module_list) == len(classifier_module_list) == len(classifier_name_list)
    set_module_list(model, classifier_name_list, classifier_module_list, new_module_list)

    new_total, _ = computer_conv_threshold(model, percent, prune_type=KEY_FILTER, prune_way=prune_way)
    return model, 1 - (1.0 * new_total / total), threshold
