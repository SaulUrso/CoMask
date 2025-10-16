import copy
from typing import Dict, Iterable

import numpy as np
import torch
from datasets import Dataset
from sklearn.cluster import AffinityPropagation, AgglomerativeClustering


def perform_clustering_task2vec(client_data_list, args):
    backbone = args.backbone


def perform_clustering_svd(client_data_list, args):
    U_clients = []

    required_fields = ["n_basis", "client_num_in_total", "preference"]
    for field in required_fields:
        if not hasattr(args, field):
            raise AttributeError(f"args is missing required field: '{field}'")

    K = args.n_basis  # 5 by default

    for idx, train_ds_local in enumerate(client_data_list):  # for each client dataset
        idxs_local = np.arange(len(train_ds_local))
        labels_local = np.array([train_ds_local[i]["label"] for i in range(len(train_ds_local))])

        # Sort Labels Train
        idxs_labels_local = np.vstack((idxs_local, labels_local))
        idxs_labels_local = idxs_labels_local[:, idxs_labels_local[1, :].argsort()]
        idxs_local = idxs_labels_local[0, :]
        labels_local = idxs_labels_local[1, :]

        uni_labels, cnt_labels = np.unique(labels_local, return_counts=True)

        print(f"Client {idx} - Labels: {uni_labels}, Counts: {cnt_labels}")

        cnt = 0
        U_temp = []
        for j in range(len(uni_labels)):  # for each label
            # obtain subset of the class - get pixel_values for this label's samples
            class_samples = []
            for sample_idx in idxs_local[cnt : cnt + cnt_labels[j]]:

                pixel_values = train_ds_local[int(sample_idx)]["pixel_values"]
                if isinstance(pixel_values, torch.Tensor):
                    pixel_values = pixel_values.numpy()
                class_samples.append(pixel_values.flatten())

            local_ds1 = np.array(class_samples).T

            # always true in practice
            if K > 0:
                # do svd only on subset (we only care about U, the first matrix of the svd)
                u1_temp, _, _ = np.linalg.svd(local_ds1, full_matrices=False)
                u1_temp = u1_temp / np.linalg.norm(u1_temp, ord=2, axis=0)
                U_temp.append(u1_temp[:, :K])

            cnt += cnt_labels[j]

        U_clients.append(copy.deepcopy(np.hstack(U_temp)))
        print(f"Client {idx} - Shape of U: {U_clients[-1].shape}")

    ###################################### Clustering
    sim_mat = -calculating_adjacency(range(args.client_num_in_total), U_clients)
    np.fill_diagonal(sim_mat, 0)  # Set diagonal to 0 for self-similarity

    preference = args.preference if isinstance(args.preference, int) else None

    cluster_labels, cluster_centers = perform_clustering(sim_mat, method="affinity", preference=preference)

    return cluster_labels, cluster_centers, sim_mat


def flatten(items):
    """Yield items from any nested iterable; see Reference."""
    for x in items:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            for sub_x in flatten(x):
                yield sub_x
        else:
            yield x


def calculating_adjacency(clients_idxs, U):
    nclients = len(clients_idxs)

    sim_mat = np.zeros([nclients, nclients])
    for idx1 in range(nclients):
        for idx2 in range(nclients):
            # print(idx1)
            # print(U)
            # print(idx1)
            U1 = copy.deepcopy(U[clients_idxs[idx1]])
            U2 = copy.deepcopy(U[clients_idxs[idx2]])

            # sim_mat[idx1,idx2] = np.where(np.abs(U1.T@U2) > 1e-2)[0].shape[0]
            # sim_mat[idx1,idx2] = 10*np.linalg.norm(U1.T@U2 - np.eye(15), ord='fro')
            # sim_mat[idx1,idx2] = 100/np.pi*(np.sort(np.arccos(U1.T@U2).reshape(-1))[0:4]).sum()
            mul = np.clip(U1.T @ U2, a_min=-1.0, a_max=1.0)
            sim_mat[idx1, idx2] = np.min(np.arccos(mul)) * 180 / np.pi

    return sim_mat


def perform_clustering(matrix, method="affinity", preference=None):
    """
    Perform clustering on the similarity matrix.

    Args:
        matrix (np.ndarray): Distance matrix (if method = "hierarchical") or similarity matrix (if method = "Affinity")
        method (str): Clustering method ('hierarchical' or 'affinity').
        kwargs: Additional arguments for the clustering method.

    Returns:
        np.ndarray: Cluster labels for each client.
        np.ndarray (optional): Cluster centers for affinity propagation.
    """
    if method == "hierarchical":
        clustering = AgglomerativeClustering(
            n_clusters=2,
            metric="precomputed",
            linkage="average",
        )
        return clustering.fit_predict(matrix)
    elif method == "affinity":
        clustering = AffinityPropagation(
            affinity="precomputed", random_state=42, preference=preference, verbose=True, max_iter=1000
        )
        labels = clustering.fit_predict(matrix)
        return labels, clustering.cluster_centers_indices_
    else:
        raise ValueError(f"Unsupported clustering method: {method}")


def ideal_clusters(client_classes):
    """
    Cluster clients based on their training classes. Clients with the exact same
    classes are grouped into the same cluster.

    Args:
        client_classes (List[Set[int]]): A list where each element is a set of class labels
                                         for a client's training data.

    Returns:
        List[List[int]]: A list of clusters, where each cluster is a list of client indices.
    """
    clusters = {}
    for client_idx, classes in enumerate(client_classes):
        classes_tuple = tuple(sorted(classes))  # Use sorted tuple as a hashable key
        if classes_tuple not in clusters:
            clusters[classes_tuple] = []
        clusters[classes_tuple].append(client_idx)

    return list(clusters.values())


class RemappedSubset:
    """
    Custom dataset wrapper that applies a remapping of class labels.

    Attributes:
        original_dataset (Dataset): The original dataset.
        indices (List[int]): Indices of the subset.
        class_mapping (Dict[int, int]): Mapping from original class labels to new labels.
    """

    def __init__(self, client_partition: Dataset, class_mapping: Dict[int, int]):
        self.dataset = client_partition
        self.class_mapping = class_mapping

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        sample = self.dataset[idx]
        sample["label"] = self.class_mapping[sample["label"]]
        return sample
