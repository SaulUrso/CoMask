# main.py


import torch

from tesifedml.models.cnn import HARBox_CNN
from tesifedml.prune import prune_cnn_by_filter
from tesifedml.prune.profile import compute_model_time, computer_flops_and_params


def prune_filter(model, prune_way="mean_abs", pruned_rate=0.2, minimum_channels=1, divisor=1):
    pruned_model, true_pruned_ratio, threshold, masks = prune_cnn_by_filter.prune(
        model, pruned_rate, prune_way=prune_way, minimum_channels=minimum_channels, divisor=divisor
    )

    print(pruned_model)
    print("pruned ratio:", true_pruned_ratio)
    print("threshold:", threshold)

    print("\nPruning Masks:")
    for mask in masks:
        print(f": {mask}")
        print(f" - Kept filters: {mask.sum().item()}/{len(mask)} ({mask.sum().item() / len(mask) * 100:.1f}%)")

    return masks


if __name__ == "__main__":
    # Create LeNet5 model instance
    torch.manual_seed(42)
    model = HARBox_CNN()

    # Print the model structure
    print("LeNet5 Model Structure:")
    print(model)

    mask_1 = prune_filter(model, prune_way="group_lasso", pruned_rate=0.05)
    computer_flops_and_params(model, data_shape=(2, 1, 30, 30))
    compute_model_time((2, 1, 30, 30), model, torch.device("cpu"))

    mask_2 = prune_filter(model, prune_way="group_lasso", pruned_rate=0.05)
    computer_flops_and_params(model, data_shape=(2, 1, 30, 30))
    compute_model_time((2, 1, 30, 30), model, torch.device("cpu"))
