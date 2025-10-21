# main.py


import torch
from fedml.model.cv.resnet_cifar import resnet18_cifar

from tesifedml.prune import prune_resnet_by_filter
from tesifedml.prune.profile import compute_model_time, computer_flops_and_params
from tesifedml.prune.utils import combine_mask


def prune_filter(model, prune_way="mean_abs", pruned_rate=0.2, minimum_channels=1, divisor=1):
    pruned_model, true_pruned_ratio, threshold, masks = prune_resnet_by_filter.prune(
        model, pruned_rate, prune_way=prune_way, minimum_channels=minimum_channels, divisor=divisor
    )

    print(pruned_model)
    print("pruned ratio:", true_pruned_ratio)
    print("threshold:", threshold)

    # print("\nPruning Masks:")
    # for name, mask in masks.items():
    #     print(f"{name}: {mask}")
    #     print(
    #         f"{name} - Kept filters: {mask['mask'].sum().item()}/{len(mask['mask'])} ({mask['mask'].sum().item() / len(mask['mask']) * 100:.1f}%)"
    #     )

    return masks


if __name__ == "__main__":
    # Create LeNet5 model instance
    torch.manual_seed(42)
    model = resnet18_cifar()

    # Print the model structure
    print("resnet Model Structure:")
    print(model)

    print("BEFORE pruning")
    # computer_flops_and_params(model, data_shape=(2, 3, 32, 32))
    # compute_model_time((2, 3, 32, 32), model, torch.device("cpu"))

    mask_1 = prune_filter(model, prune_way="group_lasso", pruned_rate=0.2, divisor=8, minimum_channels=3)
    computer_flops_and_params(model, data_shape=(2, 3, 32, 32))
    compute_model_time((2, 3, 32, 32), model, torch.device("cpu"))

    mask_2 = prune_filter(model, prune_way="group_lasso", pruned_rate=0.2, divisor=8, minimum_channels=3)
    computer_flops_and_params(model, data_shape=(2, 3, 32, 32))
    compute_model_time((2, 3, 32, 32), model, torch.device("cpu"))

    true_mask_1 = combine_mask(mask_1, mask_2)
    # print("\nCombined Mask (relative to original model):")
    # # for name, mask in true_mask_1.items():
    # #     print(f"{name}: {mask}")
    # #     original_size = mask["original_filters"]
    # #     final_kept = mask["pruned_filters"]
    # #     print(f"{name} - Final kept filters: {final_kept}/{original_size} ({final_kept / original_size * 100:.1f}%)")
    # #     print(f"{name} - Original indices kept: {mask['indices_kept']}")
    # #     print()
