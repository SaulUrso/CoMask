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

        Returns:
            Tuple of (train_data_num, test_data_num, train_data_global,
                     test_data_global, train_data_local_num_dict,
                     train_data_local_dict, test_data_local_dict, class_num)
        """
        # Number of classes
        class_num = len(np.unique(train_dataset["label"]))

        full_batch = True if batch_size < 0 else False
        if full_batch:
            batch_size = 128

        # Wrap full datasets
        wrapped_train_dataset = train_dataset.with_transform(transform_fn)
        wrapped_test_dataset = test_dataset.with_transform(transform_fn)

        # Global data loaders
        train_data_global = DataLoader(
            wrapped_train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=num_workers
        )

        test_data_global = DataLoader(
            wrapped_test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_workers
        )

        # Local data for each client
        train_data_local_num_dict = {}
        train_data_local_dict = {}
        test_data_local_dict = {}

        for client_id, client_train_dataset in enumerate(train_partitions):
            # Set transform on the client training dataset
            client_train_dataset = client_train_dataset.with_transform(transform_fn)

            train_data_local_num_dict[client_id] = len(client_train_dataset)
            train_data_local_dict[client_id] = DataLoader(
                client_train_dataset,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=collate_fn,
                num_workers=num_workers,
            )

            # Create local test dataset for this client
            if test_partitions is not None:
                client_test_dataset = test_partitions[client_id]
                client_test_dataset = client_test_dataset.with_transform(transform_fn)

                test_data_local_dict[client_id] = DataLoader(
                    client_test_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    collate_fn=collate_fn,
                    num_workers=num_workers,
                )
            else:
                # Fallback to global test data if no partition exists for this client
                test_data_local_dict[client_id] = test_data_global

        if full_batch:
            train_data_global = combine_batches(train_data_global)
            test_data_global = combine_batches(test_data_global)
            train_data_local_dict = {
                cid: combine_batches(train_data_local_dict[cid]) for cid in train_data_local_dict.keys()
            }
            test_data_local_dict = {
                cid: combine_batches(test_data_local_dict[cid]) for cid in test_data_local_dict.keys()
            }

        # Log label distributions if requested
        if log_distributions:
            FedMLAdapter.log_label_distribution(train_partitions, train_dataset, "train")
            if test_partitions is not None:
                FedMLAdapter.log_label_distribution(test_partitions, test_dataset, "test")

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
