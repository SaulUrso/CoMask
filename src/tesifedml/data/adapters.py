"""
Framework Adapters for Your Federated Learning Setup
Adapts your HuggingFace-based partitions to Flower and FedML
"""

from typing import List, Optional, Tuple

import numpy as np
from datasets import Dataset as HFDataset
from torch.utils.data import DataLoader

import wandb

from .dataload import collate_fn, combine_batches

# Import your existing functions

# class FlowerAdapter:
#     """Adapter for Flower FL framework"""

#     @staticmethod
#     def create_client_datasets(
#         partitions: Dict[int, List[int]],
#         full_dataset: HFDataset,
#         transform_fn
#     ) -> Dict[int, HFDatasetWrapper]:
#         """
#         Create client datasets for Flower.

#         Args:
#             partitions: Dict mapping client_id -> list of indices
#             full_dataset: Full HuggingFace dataset
#             transform_fn: Your transforms_tv function from dataload.py

#         Returns:
#             Dict mapping client_id -> PyTorch-compatible dataset
#         """
#         client_datasets = {}

#         for client_id, indices in partitions.items():
#             # Select subset using HF's select method
#             client_hf_dataset = full_dataset.select(indices)
#             # Wrap it for PyTorch compatibility
#             client_datasets[client_id] = HFDatasetWrapper(client_hf_dataset, transform_fn)

#         return client_datasets

#     @staticmethod
#     def create_client_dataloaders(
#         partitions: Dict[int, List[int]],
#         full_dataset: HFDataset,
#         transform_fn,
#         batch_size: int = 32,
#         shuffle: bool = True
#     ) -> Dict[int, DataLoader]:
#         """
#         Create DataLoaders for each client (ready to use in Flower).

#         Returns:
#             Dict mapping client_id -> DataLoader
#         """
#         client_datasets = FlowerAdapter.create_client_datasets(
#             partitions, full_dataset, transform_fn
#         )

#         client_dataloaders = {}
#         for client_id, dataset in client_datasets.items():
#             client_dataloaders[client_id] = DataLoader(
#                 dataset,
#                 batch_size=batch_size,
#                 shuffle=shuffle,
#                 collate_fn=collate_fn
#             )

#         return client_dataloaders

#     @staticmethod
#     def get_partition_fn(partitions: Dict[int, List[int]], full_dataset: HFDataset, transform_fn):
#         """
#         Create a partition function for Flower's simulation API.

#         Usage in Flower:
#             partition_fn = FlowerAdapter.get_partition_fn(partitions, train_dataset, transforms_tv)
#             trainloaders, valloaders, testloader = partition_fn(batch_size=32)
#         """
#         def partition_fn(batch_size: int = 32):
#             trainloaders = FlowerAdapter.create_client_dataloaders(
#                 partitions, full_dataset, transform_fn, batch_size=batch_size, shuffle=True
#             )
#             # Return as list for Flower
#             return list(trainloaders.values()), [], None  # train, val, test

#         return partition_fn


class FedMLAdapter:
    """Adapter for FedML framework"""

    @staticmethod
    def log_label_distribution(
        client_datasets: List[HFDataset], global_dataset: HFDataset, partition_name: str = "train"
    ):
        """
        Log label distribution for each client to wandb.

        Args:
            client_datasets: List of client HuggingFace datasets
            global_dataset: Global HuggingFace dataset containing labels
            partition_name: Name prefix for wandb logging (e.g., "train", "test")
        """
        labels = np.array(global_dataset["label"])

        # Log global distribution
        wandb.log(
            {
                f"{partition_name}/global_label_distribution": wandb.Histogram(labels.tolist()),
            }
        )

        # Calculate and log per-client distributions
        for client_id, client_dataset in enumerate(client_datasets):
            if client_dataset is None:
                continue
            client_labels = np.array(client_dataset["label"])

            # Log individual client distribution
            wandb.log(
                {
                    f"{partition_name}/client_{client_id}_label_distribution": wandb.Histogram(client_labels.tolist()),
                }
            )

    @staticmethod
    def create_fedml_data_structure(
        train_partitions: List[HFDataset],
        train_dataset: HFDataset,
        test_dataset: HFDataset,
        transform_fn,
        test_partitions: Optional[List[HFDataset]] = None,
        batch_size: int = 32,
        num_workers: int = 0,
        log_distributions: bool = True,
        validation_split: Optional[float] = None,
    ) -> Tuple:
        """
        Create FedML's expected data structure.

        Args:
            train_partitions: List of client training datasets
            train_dataset: Global training dataset
            test_dataset: Global test dataset
            transform_fn: Transform function to apply to datasets
            test_partitions: Optional list of client test datasets
            batch_size: Batch size for dataloaders
            num_workers: Number of workers for dataloaders
            log_distributions: Whether to log label distributions to wandb
            validation_split: Optional portion of training data to use as validation (0.0-1.0)

        Returns:
            Tuple of (train_data_num, test_data_num, train_data_global,
                     test_data_global, train_data_local_num_dict,
                     train_data_local_dict, test_data_local_dict, class_num,
                     val_data_local_dict) when validation_split is provided, or
            Tuple of (train_data_num, test_data_num, train_data_global,
                     test_data_global, train_data_local_num_dict,
                     train_data_local_dict, test_data_local_dict, class_num)
                     when validation_split is None
        """
        # Number of classes
        class_num = len(np.unique(train_dataset["label"]))

        full_batch = True if batch_size < 0 else False
        if full_batch:
            batch_size = 1024

        # Wrap full datasets
        wrapped_train_dataset = train_dataset.with_transform(transform_fn)
        wrapped_test_dataset = test_dataset.with_transform(transform_fn)

        # Global data loaders
        train_data_global = DataLoader(
            wrapped_train_dataset,  # type: ignore
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=num_workers,
        )

        test_data_global = DataLoader(
            wrapped_test_dataset,  # type: ignore
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=num_workers,
        )

        # Local data for each client
        train_data_local_num_dict = {}
        train_data_local_dict = {}
        test_data_local_dict = {}
        val_data_local_dict = {}

        null_count = 0
        null_idexes = []

        for client_id, client_train_dataset in enumerate(train_partitions):
            # Handle train/validation split if requested
            if validation_split is not None:
                client_train_dataset = client_train_dataset.shuffle(seed=client_id)
                dataset_size = len(client_train_dataset)
                val_size = int(dataset_size * validation_split)
                train_size = dataset_size - val_size
                
                # Split the dataset indices
                indices = list(range(dataset_size))
                train_indices = indices[:train_size]
                val_indices = indices[train_size:]
                
                # Create train and validation subsets
                client_train_split = client_train_dataset.select(train_indices)
                client_val_split = client_train_dataset.select(val_indices)
                
                # Set transforms
                client_train_split = client_train_split.with_transform(transform_fn)
                client_val_split = client_val_split.with_transform(transform_fn)
                
                # Store the actual training size after split
                train_data_local_num_dict[client_id] = len(client_train_split)
                
                # Create training dataloader
                train_data_local_dict[client_id] = DataLoader(
                    client_train_split,  # type: ignore
                    batch_size=batch_size,
                    shuffle=True,
                    collate_fn=collate_fn,
                )
                
                # Create validation dataloader
                val_data_local_dict[client_id] = DataLoader(
                    client_val_split,  # type: ignore
                    batch_size=batch_size,
                    shuffle=False,
                    collate_fn=collate_fn,
                )
            else:
                # No validation split - use entire dataset for training
                client_train_dataset = client_train_dataset.with_transform(transform_fn)
                train_data_local_num_dict[client_id] = len(client_train_dataset)
                train_data_local_dict[client_id] = DataLoader(
                    client_train_dataset,  # type: ignore
                    batch_size=batch_size,
                    shuffle=True,
                    collate_fn=collate_fn,
                )
                # No validation data when validation_split is None
                val_data_local_dict[client_id] = None

            # Create local test dataset for this client
            if test_partitions is not None and test_partitions[client_id] is not None:
                client_test_dataset = test_partitions[client_id]
                client_test_dataset = client_test_dataset.with_transform(transform_fn)

                test_data_local_dict[client_id] = DataLoader(
                    client_test_dataset,  # type: ignore
                    batch_size=batch_size,
                    shuffle=False,
                    collate_fn=collate_fn,
                )
            else:
                null_count += 1
                null_idexes.append(client_id)
                test_data_local_dict[client_id] = None

        print(f"NULL_COUNT: {null_count}")
        print(f"NULL_idexes: {null_idexes}")

        if full_batch:
            # train_data_global = combine_batches(train_data_global)
            # test_data_global = combine_batches(test_data_global)
            train_data_local_dict = {
                cid: combine_batches(train_data_local_dict[cid]) for cid in train_data_local_dict.keys()
            }
            test_data_local_dict = {
                cid: combine_batches(test_data_local_dict[cid]) if test_data_local_dict[cid] is not None else None
                for cid in test_data_local_dict.keys()
            }
            if validation_split is not None:
                val_data_local_dict = {
                    cid: combine_batches(val_data_local_dict[cid]) if val_data_local_dict[cid] is not None else None
                    for cid in val_data_local_dict.keys()
                }

            # assert_one_batch(train_data_global, "train_data_global")
            # assert_one_batch(test_data_global, "test_data_global")
            for cid, loader in train_data_local_dict.items():
                assert_one_batch(loader, f"train_data_local_dict[{cid}]")
            for cid, loader in test_data_local_dict.items():
                assert_one_batch(loader, f"test_data_local_dict[{cid}]")
            if validation_split is not None:
                for cid, loader in val_data_local_dict.items():
                    assert_one_batch(loader, f"val_data_local_dict[{cid}]")

        # Log label distributions if requested
        if log_distributions:
            FedMLAdapter.log_label_distribution(train_partitions, train_dataset, "train")
            if test_partitions is not None:
                FedMLAdapter.log_label_distribution(test_partitions, test_dataset, "test")

        # Return tuple with validation data if validation split was requested
        if validation_split is not None:
            return (
                len(train_dataset),  # train_data_num
                len(test_dataset),  # test_data_num
                train_data_global,  # train_data_global
                test_data_global,  # test_data_global
                train_data_local_num_dict,  # train_data_local_num_dict
                train_data_local_dict,  # train_data_local_dict
                test_data_local_dict,  # test_data_local_dict
                class_num,  # class_num
                val_data_local_dict,  # val_data_local_dict
            )
        else:
            return (
                len(train_dataset),  # train_data_num
                len(test_dataset),  # test_data_num
                train_data_global,  # train_data_global
                test_data_global,  # test_data_global
                train_data_local_num_dict,  # train_data_local_num_dict
                train_data_local_dict,  # train_data_local_dict
                test_data_local_dict,  # test_data_local_dict
                class_num,  # class_num
            )


def assert_one_batch(dataloader, name):
    if dataloader is None and ("test_data_local" in name or "val_data_local" in name):
        return
    assert len(dataloader) == 1, f"{name} does not have exactly one batch (found {len(dataloader)})"
