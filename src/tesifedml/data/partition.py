from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np
from datasets import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as _TDataset
from typing_extensions import Literal


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
    np.random.seed(seed)
    client_data_indices: Dict[int, List[int]] = {}
    idx_batch: List[List[int]] = [[] for _ in range(num_clients)]
    trial: int = 0
    while min_size < min_require_size:
        idx_batch = [[] for _ in range(num_clients)]
        for k in range(K):
            idx_k = np.where(y == k)[0]
            np.random.shuffle(idx_k)
            proportions: np.ndarray[Any, np.dtype[np.float64]] = np.random.dirichlet(
                np.repeat(data_split_alpha, num_clients)
            )

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

    for j in range(num_clients):
        np.random.shuffle(idx_batch[j])
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

    np.random.seed(seed)

    for c, indices in class_indices.items():
        np.random.shuffle(indices)  # Shuffle indices for each class
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
    np.random.seed(seed)

    if classes == 10:
        client_data_indices = {i: [] for i in range(num_clients)}
        for i in range(10):
            idx_k = np.where(y == i)[0]
            np.random.shuffle(idx_k)
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
                ind = np.random.randint(0, K)
                if ind not in current:
                    j = j + 1
                    current.append(ind)
                    times[ind] += 1
            contain.append(current)
        client_data_indices = {i: [] for i in range(num_clients)}
        for i in range(K):
            if times[i] > 0:
                idx_k = np.where(y == i)[0]
                np.random.shuffle(idx_k)
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

    np.random.seed(seed)
    np.random.shuffle(shards)

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
