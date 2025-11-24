import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from datasets import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as _TDataset
from typing_extensions import Literal


def partition(dataset, partition_method, method_name=None, client_num=None, feature_col=None, **kwargs):
    if partition_method == "natural":
        assert feature_col is not None, "Feature col must be declared for natural partition"
        return partition_dataset_natural(dataset, feature_column=feature_col)
    elif partition_method == "label":
        assert method_name is not None
        assert client_num is not None
        return partition_dataset_with_labels(method_name=method_name, dataset=dataset, num_clients=client_num,**kwargs)

    else:
        raise ValueError(f"{partition_method} is not a valid partition method.")


def dirichlet_partition(
    y: np.ndarray,
    num_clients: int,
    data_split_alpha: float = 0.5,
    seed: int = 0,
    min_require_size: int = 10,
    self_balancing: bool = True,
) -> Dict[int, List[int]]:
    """Partitions dataset indices among clients using a Dirichlet distribution to
    simulate non-IID data splits.

    Args:
        y (np.ndarray): array containing the classes of the samples of the dataset
        num_clients (int): number of partitions to create
        data_split_alpha (float, optional): alpha od dirilichet distribution. Defaults
        to 0.5.
        seed (int, optional): Seed used for the random number generator. Defaults to 0.
        min_require_size (int, optional): Minimum amount of samples present in each
        client. Defaults to 10.
        self_balancing (bool, optional): Whether to balance the partitioning of samples
        among the clients. Defaults to True.

    Raises:
        ValueError: When trials done for creating the partitioning never manage to
        respect the minimum number of samples per client. Trieas at most 10 times

    Returns:
        Dict[int, List[int]]: A dictionary mapping each client index (int) to a list of
        dataset sample indices (List[int]) assigned to that client.
    """
    min_size: int = 0
    K: int = len(np.unique(y))

    N: int = y.shape[0]
    rng = np.random.default_rng(seed)
    client_data_indices: Dict[int, List[int]] = {}
    idx_batch: List[List[int]] = [[] for _ in range(num_clients)]
    trial: int = 0
    while min_size < min_require_size:
        idx_batch = [[] for _ in range(num_clients)]

        # first sample proportions in order to make it independent of sample amount for random number
        class_proportions = []
        for k in range(K):
            proportions: np.ndarray[Any, np.dtype[np.float64]] = rng.dirichlet(
                np.repeat(data_split_alpha, num_clients)
            )
            # print(proportions)
            class_proportions.append(proportions)

        for k in range(K):
            idx_k = np.where(y == k)[0]
            proportions: np.ndarray[Any, np.dtype[np.float64]] = class_proportions[k]
            rng.shuffle(idx_k)

            ## Balance
            if self_balancing:
                proportions = np.array([p * (len(idx_j) < N / num_clients) for p, idx_j in zip(proportions, idx_batch)])

            proportions = proportions / proportions.sum()

            sample_prop: np.ndarray[tuple, np.dtype[np.int_]] = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, sample_prop))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

        trial += 1

        if trial >= 10:
            raise ValueError(f"Max number of attempts {trial} reached, try a different alpha.")

        # TODO: hotfix to ensure test and train have same distribution. Fix later.
        if min_size < min_require_size:
            raise ValueError(f"Min size of {min_size} < {min_require_size} , try a different alpha.")

    for j in range(num_clients):
        rng.shuffle(idx_batch[j])
        client_data_indices[j] = idx_batch[j]

    return client_data_indices


def uniform_partition(
    y: np.ndarray,
    num_clients: int,
    seed=0,
) -> Dict[int, List[int]]:
    """Partitions dataset indices among clients uniformly at random, simulating an
    I.I.D. scenario

    Args:
        y (np.ndarray): array containing the classes of the samples of the dataset
        num_clients (int): number of partitions to create
        seed (int, optional): Seed used for the random number generator. Defaults to 0.
    Returns:
        Dict[int, List[int]]: A dictionary mapping each client index (int) to a list of
        dataset sample indices (List[int]) assigned to that client.
    """
    num_classes = len(np.unique(y))
    client_data_indices: Dict[int, List[int]] = defaultdict(list)

    # Collect sample indices by class
    class_indices = {c: np.where(y == c)[0] for c in range(num_classes)}

    rng = np.random.default_rng(seed)

    for c, indices in class_indices.items():
        rng.shuffle(indices)  # Shuffle indices for each class
        splits = np.array_split(indices, num_clients)  # Split indices among clients
        for client_id, split in enumerate(splits):
            client_data_indices[client_id].extend(split)

    return client_data_indices


def class_partition(
    y: np.ndarray,
    num_clients: int,
    classes: int = 2,
    seed: int = 0,
) -> Dict[int, List[int]]:
    """Generate Non-IID data partition where each client is assigned a fixed number of
    classes.

    Args:
        y (np.ndarray): array containing the classes of the samples of the dataset.
        num_clients (int): number of partitions to create.
        classes (int, optional): Number of classes each client should. Defaults to 2.
        seed (int, optional): _description_. Defaults to 0.

    Returns:
        Dict[int, List[int]]: A dictionary mapping each client index (int) to a list of
        dataset sample indices (List[int]) assigned to that client.
    """
    K: int = len(np.unique(y))
    client_data_indices: Dict[int, List[int]] = defaultdict(list)
    rng = np.random.default_rng(seed)

    if classes == 10:
        client_data_indices = {i: [] for i in range(num_clients)}
        for i in range(10):
            idx_k = np.where(y == i)[0]
            rng.shuffle(idx_k)
            split = np.array_split(idx_k, num_clients)
            for j in range(num_clients):
                client_data_indices[j] = list(np.append(client_data_indices[j], split[j]))
    else:
        times = [0 for i in range(K)]
        contain = []
        for i in range(num_clients):
            current = [i % K]
            times[i % K] += 1
            j = 1
            while j < classes:
                ind = rng.integers(low=0, high=K)
                if ind not in current:
                    j = j + 1
                    current.append(ind)
                    times[ind] += 1
            contain.append(current)
        client_data_indices = {i: [] for i in range(num_clients)}
        for i in range(K):
            if times[i] > 0:
                idx_k = np.where(y == i)[0]
                rng.shuffle(idx_k)
                split = np.array_split(idx_k, times[i])
                ids = 0
                for j in range(num_clients):
                    if i in contain[j]:
                        client_data_indices[j] = list(np.append(client_data_indices[j], split[ids]))
                        ids += 1

    return client_data_indices


def shard_partition(
    y: np.ndarray,
    num_clients: int,
    shards_per_client: int = 2,
    seed=0,
) -> Dict[int, List[int]]:
    """Partitions data indices into shards and assigns them to clients for federated
    learning.

    This function sorts the labels, splits them into shards, shuffles the shards, and
    assigns a specified number of shards to each client.

    Args:
        y (np.ndarray): array containing the classes of the samples of the dataset.
        num_clients (int): Number of clients to partition the data among.
        shards_per_client (int, optional): Number of shards assigned to each client.
        Defaults to 2.
        seed (int, optional): Random seed for reproducibility. Defaults to 0.
    Returns:
        Dict[int, List[int]]: A dictionary mapping each client index (int) to a list of
        dataset sample indices (List[int]) assigned to that client.
    """

    N = len(y)

    client_data_indices: Dict[int, List[int]] = {i: [] for i in range(num_clients)}
    sorted_indices = np.argsort(y)

    num_shards = num_clients * shards_per_client
    shard_size = N // num_shards

    shards = [sorted_indices[i * shard_size : (i + 1) * shard_size] for i in range(num_shards)]

    rng = np.random.default_rng(seed)
    rng.shuffle(shards)

    for i in range(num_clients):
        assigned_shards = shards[i * shards_per_client : (i + 1) * shards_per_client]
        client_data_indices[i] = np.concatenate(assigned_shards).tolist()

    return client_data_indices


PARTITIONING_METHODS = {
    "uniform": uniform_partition,
    "shard": shard_partition,
    "dirichlet": dirichlet_partition,
    "class": class_partition,
}

PartitioningMethod = Literal["uniform", "shard", "dirichlet", "class"]


def partition_labels(
    method_name: PartitioningMethod, y: np.ndarray, num_clients: int, **kwargs
) -> Dict[int, List[int]]:
    """Partitions labels among clients using the specified partitioning method.

    This function delegates the partitioning of the label array `y` among `num_clients`
    to the method specified by `method_name`. Additional keyword arguments can be passed
    to customize the partitioning behavior.

    Args:
        method_name (PartitioningMethod): The partitioning strategy to use.
        y (np.ndarray): array containing the classes of the samples of the dataset.
        num_clients (int): number of partitions to create.
        **kwargs: Additional keyword arguments passed to the partitioning method.

    Returns:
        Dict[int, List[int]]: A dictionary mapping each client index (int) to a list of
        dataset sample indices (List[int]) assigned to that client.

    Raises:
        KeyError: If `method_name` is not found in PARTITIONING_METHODS.
    """
    return PARTITIONING_METHODS[method_name](y, num_clients, **kwargs)


def partition_dataset_with_labels(
    method_name: PartitioningMethod,
    dataset: Dataset,
    num_clients: int,
    label_column: str = "label",
    **kwargs,
) -> List[Dataset]:
    if label_column in dataset.column_names:
        labels = np.array(dataset[label_column])
    else:
        raise ValueError("Cannot extract labels from dataset")

    # Partition
    client_data_indices = partition_labels(method_name, labels, num_clients, **kwargs)
    client_datasets = [dataset.select(indices) for indices in client_data_indices.values()]

    return client_datasets


def create_dataloaders(
    client_datasets: List[Dataset], batch_size: int = 32, shuffle: bool = True, **dataloader_kwargs
) -> List[DataLoader]:
    """
    Create PyTorch DataLoaders from HuggingFace datasets.
    """
    dataloaders = []

    for dataset in client_datasets:
        # Set the format to PyTorch tensors
        # This tells HF dataset to return torch tensors instead of Python objects
        # dataset.set_format(type="torch")

        # Create DataLoader
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, **dataloader_kwargs)  # type: ignore
        dataloaders.append(dataloader)

    return dataloaders


def create_dataloaders_2(
    client_datasets: List[_TDataset], batch_size: int = 32, shuffle: bool = True, **dataloader_kwargs
) -> List[DataLoader]:
    """
    Create PyTorch DataLoaders from HuggingFace datasets.
    """
    dataloaders = []

    for dataset in client_datasets:
        # Set the format to PyTorch tensors
        # This tells HF dataset to return torch tensors instead of Python objects
        # dataset.set_format(type="torch")

        # Create DataLoader
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, **dataloader_kwargs)  # type: ignore
        dataloaders.append(dataloader)

    return dataloaders


def split_train_validation(
    client_data_list: List[Dataset], validation_size: float, seed: int = 42
) -> Tuple[List[Dataset], List[Dataset]]:
    """
    Split each client's HuggingFace dataset into training and validation sets using native HF method.

    Args:
        client_data_list (List[Dataset]): List of client HuggingFace datasets
        validation_size (float): Proportion of the dataset to use for validation (0.0 to 1.0)
        seed (int): Random seed for reproducibility

    Returns:
        Tuple[List[Dataset], List[Dataset]]: training datasets and validation datasets for each client
    """
    train_datasets = []
    val_datasets = []

    for client_data in client_data_list:
        split_data = client_data.train_test_split(
            test_size=validation_size,
            seed=seed,
            shuffle=True,
        )

        train_datasets.append(split_data["train"])
        val_datasets.append(split_data["test"])  # HF calls it 'test' but it's our validation

    return train_datasets, val_datasets


def create_personalized_test_sets(
    client_data_list: List[Dataset],
    test_set: Dataset,
    label_column: str = "label",
) -> Tuple[List[Dataset], List[np.ndarray]]:
    """Create a personalized test set for each client based on classes present in their
    training data.

    Args:
        client_data_list (List[Dataset]): List of client training datasets
        test_set: The global test dataset
        label_column (str): Name of the label column in the datasets

    Returns:
        Tuple[List[Dataset], List[np.ndarray]]: Tuple containing:
            - List of personalized test datasets for each client
            - List of classes present in each client's data
    """

    client_test_sets = []
    client_class_list = []

    for client_data in client_data_list:
        # obtain the classes of the client from its dataset
        classes = np.unique(client_data[label_column])
        client_class_list.append(classes)

        # Filter test set indices for the client's classes
        client_test_indices = [idx for idx, label in enumerate(test_set[label_column]) if label in classes]

        # Create personalized test dataset for this client
        client_test_set = test_set.select(client_test_indices)
        client_test_sets.append(client_test_set)

    return client_test_sets, client_class_list


def create_natural_test_sets(
    client_datasets: List[Dataset],
    test_set: Dataset,
    feature_column: str,
) -> List[Dataset]:
    """Create personalized test sets for clients based on natural partitioning feature values.

    For each client dataset (created from partition_dataset_natural), this function creates
    a corresponding test set containing all test samples that have the same feature values
    as the client's training data.

    Args:
        client_datasets (List[Dataset]): List of client datasets from partition_dataset_natural
        test_set (Dataset): The global test dataset
        feature_column (str): Name of the feature column used for natural partitioning

    Returns:
        List[Dataset]: List of personalized test datasets for each client

    Raises:
        ValueError: If feature_column is not found in datasets
    """
    if feature_column not in test_set.column_names:
        raise ValueError(f"Feature column '{feature_column}' not found in test set columns: {test_set.column_names}")

    client_test_sets = []
    test_feature_values = np.array(test_set[feature_column])

    for client_idx, client_data in enumerate(client_datasets):
        if feature_column not in client_data.column_names:
            raise ValueError(f"Feature column '{feature_column}' not found in client {client_idx} dataset columns: {client_data.column_names}")
        
        # Get unique feature values from this client's training data
        client_feature_values = np.unique(client_data[feature_column])

        assert len(client_feature_values) == 1
        
        # Find test samples that match any of the client's feature values
        matching_indices = []
        for value in client_feature_values:
            indices = np.where(test_feature_values == value)[0]
            matching_indices.extend(indices)
        
        assert len(matching_indices) == len(list(set(matching_indices)))
        
        # Create personalized test dataset for this client
        if matching_indices:
            client_test_set = test_set.select(matching_indices)
        else:
            # If no matching samples found, create empty dataset with same structure
            client_test_set = None
        
        client_test_sets.append(client_test_set)

    # print(f"✓ Created {len(client_test_sets)} personalized test sets based on '{feature_column}'")
    # print(f"✓ Test samples per client: min={min([len(ds) for ds in client_test_sets if ds is not None])}, "
    #       f"max={max([len(ds) for ds in client_test_sets if ds is not None])}, "
    #       f"mean={np.mean([len(ds) for ds in client_test_sets if ds is not None]):.1f}")

    return client_test_sets


def create_and_save_partition(
    dataset: Dataset,
    num_clients: int,
    method: PartitioningMethod,
    save_path: str,
    label_column: str = "label",
    **partition_kwargs,
) -> Dict[int, List[int]]:
    """
    Create partition using your existing partition_labels function and save indices.

    Args:
        dataset: HuggingFace dataset
        num_clients: Number of clients
        method: Partitioning method ("uniform", "shard", "dirichlet", "class")
        save_path: Path to save partition indices
        label_column: Name of label column in dataset
        **partition_kwargs: Additional args for partition method (e.g., data_split_alpha, seed)

    Returns:
        Dictionary mapping client_id -> list of indices
    """
    # Extract labels
    labels = np.array(dataset[label_column])

    # Use YOUR partition_labels function
    client_indices = partition_labels(method, labels, num_clients, **partition_kwargs)

    # Save indices
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(client_indices, f)

    # print(f"✓ Created {method} partition with {num_clients} clients")
    # print(f"✓ Saved to {save_path}")

    return client_indices


def load_partition_indices(load_path: str) -> Dict[int, List[int]]:
    """Load saved partition indices"""
    with open(load_path, "rb") as f:
        partitions = pickle.load(f)
    # print(f"✓ Loaded partition from {load_path}")
    return partitions


def partition_dataset_natural(
    dataset: Dataset,
    feature_column: str,
    min_samples_per_client: int = 10,
) -> List[Dataset]:
    """Partition dataset naturally based on feature values (e.g., writer ID for FEMNIST).

    Each unique value in the specified feature column becomes a separate client partition.
    This is useful for datasets like FEMNIST where natural client boundaries exist
    (e.g., different writers).

    Args:
        dataset (Dataset): The HuggingFace dataset to partition
        feature_column (str): Name of the feature column to partition by
        max_clients (int, optional): Maximum number of clients. If None, uses all unique values.
        min_samples_per_client (int, optional): Minimum samples per client. Clients with fewer
            samples are filtered out. Defaults to 1.

    Returns:
        List[Dataset]: List of client datasets, one per unique feature value

    Raises:
        ValueError: If feature_column is not in dataset or no valid clients found
    """
    if feature_column not in dataset.column_names:
        raise ValueError(f"Feature column '{feature_column}' not found in dataset columns: {dataset.column_names}")

    # Get unique feature values
    feature_values = np.array(dataset[feature_column])
    unique_values = np.unique(feature_values)
    # print(f"UNIQUE values = {len(unique_values)}")

    # Group indices by feature value
    client_data_indices = {}
    for value in unique_values:
        indices = np.where(feature_values == value)[0].tolist()
        if len(indices) < min_samples_per_client:
            raise ValueError(f"Expected {min_samples_per_client} samples but found {len(indices)} samples.")
        client_data_indices[value] = indices

    # Create client datasets
    client_datasets = []
    for value in sorted(client_data_indices.keys()):
        indices = client_data_indices[value]
        client_dataset = dataset.select(indices)
        client_datasets.append(client_dataset)

    # print(f"✓ Created natural partition with {len(client_datasets)} clients based on '{feature_column}'")
    # print(
    #     f"✓ Samples per client: min={min([len(ds) for ds in client_datasets])}, "
    #     f"max={max([len(ds) for ds in client_datasets])}, "
    #     f"mean={np.mean([len(ds) for ds in client_datasets]):.1f}"
    # )

    return client_datasets



