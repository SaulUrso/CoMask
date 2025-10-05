# main.py


import torch
from fedml.model.cv.cnn import CNN_WEB

from tesifedml.prune import prune_lenet_by_filter_and_channel_v2
from tesifedml.prune.profile import compute_model_time, computer_flops_and_params


def prune_filter(model, prune_way="mean_abs", pruned_rate=0.2, minimum_channels=1, divisor=1):
    pruned_model, true_pruned_ratio, threshold = prune_lenet_by_filter_and_channel_v2.prune(
        model, pruned_rate, prune_way=prune_way, minimum_channels=minimum_channels, divisor=divisor
    )

    print(pruned_model)
    print("pruned ratio:", true_pruned_ratio)
    print("threshold:", threshold)


if __name__ == "__main__":
    # Create LeNet5 model instance
    torch.manual_seed(42)
    model = CNN_WEB()

    # Set all weights of the first filter in the first conv layer to zero
    # with torch.no_grad():
    #     model.conv1.weight[0].zero_()
    #     if model.conv1.bias is not None:
    #         model.conv1.bias[0] = 0.0

    #     model.conv2.weight[0].zero_()
    #     if model.conv2.bias is not None:
    #         model.conv2.bias[0] = 0.0

    # Print the model structure
    print("LeNet5 Model Structure:")
    print(model)

    prune_filter(model, prune_way="group_lasso", pruned_rate=0.5)
    computer_flops_and_params(model, data_shape=(1, 3, 32, 32))
    compute_model_time((1, 3, 32, 32), model, torch.device("cpu"))
