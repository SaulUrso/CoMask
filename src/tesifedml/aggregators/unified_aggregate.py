from typing import Dict, List

from fedml.model.cv.resnet_cifar import ResNet
from torch import nn

from ..models.cnn import HARBox_CNN
from ..models.mobilenet import MobileNet
from .aggregate_cnn import aggregate_cnn_with_masks
from .aggregate_mobilenet import aggregate_mobilenet_with_masks
from .aggregate_resnet import aggregate_resnet_with_masks


def aggregate_model(
    models: List[nn.Module],
    masks: List[Dict],
    counters: List[int],
    **kwargs,
) -> nn.Module:
    if isinstance(models[0], HARBox_CNN):
        agg_model = aggregate_cnn_with_masks(models, masks, counters, **kwargs)

    elif isinstance(models[0], MobileNet):
        agg_model = aggregate_mobilenet_with_masks(models, masks, counters, **kwargs)

    elif isinstance(models[0], ResNet):
        agg_model = aggregate_resnet_with_masks(models, masks, counters, **kwargs)

    else:
        raise ValueError("Model istance is not supported for aggregation")

    return agg_model
