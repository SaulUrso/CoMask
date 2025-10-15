import os
import sys
from typing import List

import numpy as np
import torch
import torchvision.transforms as T
from datasets import Dataset, DatasetDict, load_dataset
from sklearn.model_selection import train_test_split

from .data_pre import NUM_OF_TOTAL_USERS, load_data

# Add the path to access data_pre.py functions
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../large_scale_HARBox"))


def load_my_data(dataset_name, **kwargs):
    if dataset_name == "cifar10":
        dataset, trs = load_cifar10()
    elif dataset_name == "harbox":
        dataset, trs = load_harbox()
    elif dataset_name == "femnist":
        dataset, trs = load_femnist(**kwargs)
    else:
        raise ValueError(f"{dataset_name} is an unknown dataset.")

    return dataset, trs


def load_cifar10():
    dataset = load_dataset("uoft-cs/cifar10")

    assert isinstance(dataset, DatasetDict)

    # Define the torchvision transforms
    # TODO: change transform
    transform_tv = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.247, 0.243, 0.261),
            ),
        ]
    )

    def transforms_tv(examples):
        # Apply the torchvision transforms
        pixel_values = [
            transform_tv(image.convert("RGB")) for image in examples["img"]
        ]  # Ensure image is in RGB format
        return {"pixel_values": pixel_values, "label": examples["label"]}

    return dataset, transforms_tv


def load_femnist(test_only_users=None, test_only_user_seed=42):
    dataset = load_dataset("flwrlabs/femnist")

    assert isinstance(dataset, DatasetDict)

    # TODO: add normalization
    # TODO: in partition group them together because some clients have too low data, go check papers to see how they do it

    transform_tv = T.Compose(
        [
            T.ToTensor(),
        ]
    )

    def transforms_tv(examples):
        # FEMNIST is 28 x 28 greyscale images (only 1 channel, but you can replicate if need 3, ofc it's useless)
        pixel_values = [transform_tv(image.convert("L")) for image in examples["image"]]
        return {"pixel_values": pixel_values, "label": examples["label"]}

    # adding label column (called character originally)
    dataset["train"] = dataset["train"].rename_column("character", "label")

    # Handle test_only users if specified
    if test_only_users is not None and test_only_users > 0:
        if isinstance(test_only_users, int):
            # Get unique writer_ids from train set
            train_writer_ids = list(set(dataset["train"]["writer_id"]))

            # Randomly sample test_only_users number of writer IDs
            np.random.seed(test_only_user_seed)
            test_only_writer_ids = set(
                np.random.choice(train_writer_ids, size=min(test_only_users, len(train_writer_ids)), replace=False)
            )

            # Split train dataset
            train_indices = []
            test_only_indices = []

            for i, writer_id in enumerate(dataset["train"]["writer_id"]):
                if writer_id in test_only_writer_ids:
                    test_only_indices.append(i)
                else:
                    train_indices.append(i)

            # Create new splits
            new_train = dataset["train"].select(train_indices)
            test_only = dataset["train"].select(test_only_indices)

            # Create new DatasetDict with test_only split
            dataset = DatasetDict({"train": new_train, "test_only": test_only})
        else:
            raise ValueError("test_only_users must be None or an integer")

    # Now create train/test splits from remaining train data
    # Group remaining train data by writer_id
    print("iterating writer")
    writer_data = {}
    for i, writer_id in enumerate(dataset["train"]["writer_id"]):
        if writer_id not in writer_data:
            writer_data[writer_id] = []
        writer_data[writer_id].append(i)

    # Create train/test splits
    train_indices = []
    test_indices = []

    print("iterating train-test split")
    for writer_id, indices in writer_data.items():
        if len(indices) >= 100:
            # Split 80/20 for train/test
            indices_array = np.array(indices)
            # labels = [dataset["train"]["character"][i] for i in indices]

            train_idx, test_idx = train_test_split(
                indices_array, test_size=0.2, random_state=42
            )  # TODO: discuss stratify=labels
            train_indices.extend(train_idx.tolist())
            test_indices.extend(test_idx.tolist())
        else:
            # All data goes to train (less than 100 samples)
            train_indices.extend(indices)

    dataset_dict = {}
    if train_indices:
        dataset_dict["train"] = dataset["train"].select(train_indices)
    if test_indices:
        dataset_dict["test"] = dataset["train"].select(test_indices)
    if "test_only" in dataset:
        dataset_dict["test_only"] = dataset["test_only"]

    dataset = DatasetDict(dataset_dict)

    return dataset, transforms_tv


def _load_user_data(user_id):
    """Load data for a single user."""
    try:
        x_coll, y_coll, _, _ = load_data(user_id)
        return x_coll, y_coll
    except Exception as e:
        print(f"Warning: Could not load data for user {user_id}: {e}")
        return None, None


def _split_user_data(x_data, y_data, user_id_0indexed):
    """Split a user's data into train/test (80/20)."""
    if x_data.shape[0] < 5:
        # If user has very little data, put all in train
        return {
            "train": (x_data, y_data, np.full(x_data.shape[0], user_id_0indexed)),
            "test": (np.empty((0, x_data.shape[1])), np.empty(0), np.empty(0, dtype=int)),
        }

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42, stratify=y_data)

    train_user_ids = np.full(x_train.shape[0], user_id_0indexed)
    test_user_ids = np.full(x_test.shape[0], user_id_0indexed)

    return {"train": (x_train, y_train, train_user_ids), "test": (x_test, y_test, test_user_ids)}


def _collect_user_data(test_only_user_ids):
    """Collect and organize data from all users."""

    data_splits = {
        "train": {"x": [], "y": [], "users": []},
        "test": {"x": [], "y": [], "users": []},
        "test_only": {"x": [], "y": [], "users": []},
    }

    for user_id in range(1, NUM_OF_TOTAL_USERS + 1):
        x_data, y_data = _load_user_data(user_id)

        if x_data is None or x_data.shape[0] == 0:
            continue

        user_id_0indexed = user_id - 1

        if user_id_0indexed in test_only_user_ids:
            # Test-only user
            data_splits["test_only"]["x"].append(x_data)
            data_splits["test_only"]["y"].append(y_data)
            data_splits["test_only"]["users"].append(np.full(x_data.shape[0], user_id_0indexed))
        else:
            # Regular user - split into train/test
            splits = _split_user_data(x_data, y_data, user_id_0indexed)

            for split_name in ["train", "test"]:
                x_split, y_split, user_ids_split = splits[split_name]
                if x_split.shape[0] > 0:
                    data_splits[split_name]["x"].append(x_split)
                    data_splits[split_name]["y"].append(y_split)
                    data_splits[split_name]["users"].append(user_ids_split)

    return data_splits


def _combine_split_data(split_data):
    """Combine data arrays for a single split."""
    if not split_data["x"]:
        return np.empty((0, 30, 30)), np.empty(0), np.empty(0, dtype=int)

    combined_x = np.concatenate(split_data["x"], axis=0)
    combined_y = np.concatenate(split_data["y"], axis=0)
    combined_users = np.concatenate(split_data["users"], axis=0)

    # Reshape from 900 to 30x30
    combined_x_reshaped = combined_x.reshape(-1, 30, 30)

    return combined_x_reshaped, combined_y, combined_users


def _create_dataset_dict(data_splits):
    """Create DatasetDict from organized data splits."""
    dataset_dict = {}

    for split_name in ["train", "test", "test_only"]:
        x_data, y_data, user_data = _combine_split_data(data_splits[split_name])

        if x_data.shape[0] > 0:  # Only create dataset if there's data
            dataset = Dataset.from_dict(
                {
                    "features": x_data,
                    "label": y_data.astype(int),
                    "user_id": user_data.astype(int),
                }
            )

            dataset_dict[split_name] = dataset

    return DatasetDict(dataset_dict)


def load_harbox(test_only_users=None, test_only_user_seed=42):
    """
    Load HARBox dataset using functions from data_pre.py.

    Args:
        test_only_users: None or integer representing number of users to use only for testing
        test_only_user_seed: Random seed for sampling test-only users

    Returns:
        dataset: DatasetDict with train, test, and optionally test_only splits
        transforms_tv: Transform function to apply to examples
    """
    # Convert test_only_users to 0-indexed set
    test_only_user_ids = set()
    if test_only_users is not None and test_only_users > 0:
        if isinstance(test_only_users, int):
            # Randomly sample test_only_users number of user IDs
            np.random.seed(test_only_user_seed)
            all_user_ids = list(range(NUM_OF_TOTAL_USERS))  # 0-indexed
            sampled_ids = np.random.choice(all_user_ids, size=min(test_only_users, NUM_OF_TOTAL_USERS), replace=False)
            test_only_user_ids = set(sampled_ids)
        else:
            raise ValueError("test_only_users must be None or an integer")

    # Collect and organize data from all users
    data_splits = _collect_user_data(test_only_user_ids)

    # Check if any data was loaded
    if not any(data_splits[split]["x"] for split in ["train", "test"]):
        raise ValueError("No data could be loaded from any user")

    # Create DatasetDict
    dataset = _create_dataset_dict(data_splits)

    # TODO: CHANGE IF YOU USE SOME CLIENTS FOR SPLITS
    transform_tv = T.Compose([T.ToTensor(), T.Normalize(-1.807077, 17.650785)])

    def transforms_tv(examples):
        pixel_values = [transform_tv(np.array(image, dtype=np.float32)) for image in examples["features"]]
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


def combine_batches(batches):
    if batches is None:
        return None
    full_x = torch.from_numpy(np.asarray([])).float()
    full_y = torch.from_numpy(np.asarray([])).long()
    for batched_x, batched_y in batches:
        full_x = torch.cat((full_x, batched_x), 0)
        full_y = torch.cat((full_y, batched_y), 0)
    assert len(full_x) > 0
    assert len(full_y) > 0
    return [(full_x, full_y)]
