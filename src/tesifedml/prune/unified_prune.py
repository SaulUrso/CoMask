from ..models.cnn import HARBox_CNN
from ..models.mobilenet import MobileNet
from fedml.model.cv.resnet_cifar import ResNet
from tesifedml.models.resnet_cifar import ResNet as ResNet2

from . import prune_cnn_by_filter
from . import prune_mobilenet_by_filter
from . import prune_resnet_by_filter


def prune_model(model, percent, prune_way="mean_abs", minimum_channels=1, divisor=1, with_mask=None):
    """
    Unified pruning function that automatically detects model type and applies appropriate pruning.

    Args:
        model: The model to prune (HARBox_CNN, MobileNet, or ResNet)
        percent: Pruning percentage or absolute number of groups to prune
        prune_way: Pruning method ("mean_abs", etc.)
        minimum_channels: Minimum channels to keep in each layer
        divisor: Divisor for rounding channel numbers
        with_mask: Optional pre-computed mask for pruning

    Returns:
        tuple: (pruned_model, param_pruning_ratio, group_pruning_ratio, threshold, masks, units_pruned)
    """
    if isinstance(model, HARBox_CNN):
        return prune_cnn_by_filter.prune(
            model, percent, prune_way, minimum_channels, divisor, with_mask
        )
    elif isinstance(model, MobileNet):
        return prune_mobilenet_by_filter.prune(
            model, percent, prune_way, minimum_channels, divisor, with_mask
        )
    elif isinstance(model, ResNet) or isinstance(model,ResNet2):
        return prune_resnet_by_filter.prune(
            model, percent, prune_way, minimum_channels, divisor, with_mask
        )
    else:
        raise ValueError(
            f"Unsupported model type: {type(model)}. Supported types are HARBox_CNN, MobileNet, and ResNet"
        )