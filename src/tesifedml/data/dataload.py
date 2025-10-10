from typing import List

import torch
import torchvision.transforms as T
from datasets import DatasetDict, load_dataset


def load_cifar10():
    dataset = load_dataset("uoft-cs/cifar10")

    assert isinstance(dataset, DatasetDict)

    # Define the torchvision transforms
    transform_tv = T.Compose(
        [
            T.ToTensor(),  # Convert PIL Image or numpy array to tensor
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalization
        ]
    )

    def transforms_tv(examples):
        # Apply the torchvision transforms
        pixel_values = [
            transform_tv(image.convert("RGB")) for image in examples["img"]
        ]  # Ensure image is in RGB format
        return {"pixel_values": pixel_values, "label": examples["label"]}
    
    return dataset, transforms_tv


def collate_fn(examples):
    images = []
    labels = []
    for example in examples:
        images.append((example["pixel_values"]))
        labels.append(example["label"])

    return CustomImageBatch(images, labels)


class CustomImageBatch:
    def __init__(self, images: List[torch.Tensor], labels: List[int]):
        self.images = torch.stack(images)
        self.labels = torch.tensor(labels, dtype=torch.int64)

    def pin_memory(self):
        self.images = self.images.pin_memory()
        self.labels = self.labels.pin_memory()
        return self

    def to(self, device):
        self.images = self.images.to(device)
        self.labels = self.labels.to(device)
        return self

    def __iter__(self):
        return iter((self.images, self.labels))

    def __len__(self):
        return len(self.labels)
