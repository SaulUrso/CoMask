import contextlib
import copy
from typing import Dict, Iterable

import joblib
import numpy as np
import torch
from datasets import Dataset
from joblib import Parallel, delayed
from sklearn.cluster import AffinityPropagation, AgglomerativeClustering
from tqdm import tqdm


def perform_clustering_svd(client_data_list, args, output_path="/disc/homes/urso/adjacency_matrix.npz"):
    U_clients = []

    required_fields = ["n_basis", "client_num_in_total", "preference"]
    for field in required_fields:
        if not hasattr(args, field):
            raise AttributeError(f"args is missing required field: '{field}'")

    K = args.n_basis  # 5 by default in experiments


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

    n_clusters = getattr(args, "n_clusters", None)

    if isinstance(n_clusters, int) and n_clusters > 0:
        # Automatically tune AffinityPropagation's `preference` so the clustering
        # yields exactly `n_clusters` clusters (closest achievable otherwise).
        cluster_labels, cluster_centers, _ = find_preference_for_n_clusters(sim_mat, n_clusters)
    else:
        preference = args.preference if isinstance(args.preference, int) else None
        cluster_labels, cluster_centers = perform_clustering(sim_mat, method="affinity", preference=preference)

    return cluster_labels, cluster_centers, sim_mat


def find_preference_for_n_clusters(sim_mat, target, random_state=42, n_iter=40):
    """
    Find the AffinityPropagation `preference` value that yields `target` clusters
    via bisection.

    AffinityPropagation does not let you fix the number of clusters directly, but
    the cluster count grows (approximately) monotonically with `preference`: more
    negative preference -> fewer clusters, less negative -> more clusters. We
    bisect the preference over the range of off-diagonal similarity values until
    we hit the target count (or get as close as possible).

    Args:
        sim_mat (np.ndarray): Precomputed similarity matrix (diagonal zeroed).
        target (int): Desired number of clusters.
        random_state (int): Seed for AffinityPropagation (deterministic results).
        n_iter (int): Maximum number of bisection probes.

    Returns:
        tuple: (cluster_labels, cluster_centers_indices, used_preference) for the
        clustering whose count is closest to `target`.
    """
    # Initial search bounds: range of off-diagonal similarity values. More
    # negative preference -> fewer clusters; less negative -> more clusters.
    off_diag = sim_mat[~np.eye(sim_mat.shape[0], dtype=bool)]
    low, high = float(off_diag.min()), float(off_diag.max())
    span = high - low if high > low else abs(high) + 1.0

    def cluster_at(pref):
        labels, centers = perform_clustering(sim_mat, method="affinity", preference=pref)
        return labels, centers, len(set(labels))

    best = None  # (abs_diff, count, preference, labels, centers)

    def consider(pref, labels, centers, count):
        nonlocal best
        diff = abs(count - target)
        if best is None or diff < best[0]:
            best = (diff, count, pref, labels, centers)

    # Expand the bounds outward until they bracket the target count. The
    # conventional [min, max] similarity range does not always reach very small
    # (or very large) cluster counts, so push `low` further negative while it
    # still yields too many clusters, and `high` higher while it yields too few.
    for _ in range(20):
        labels, centers, count = cluster_at(low)
        consider(low, labels, centers, count)
        if count <= target:
            break
        low -= span
    for _ in range(20):
        labels, centers, count = cluster_at(high)
        consider(high, labels, centers, count)
        if count >= target:
            break
        high += span

    for _ in range(n_iter):
        mid = (low + high) / 2.0
        labels, centers, count = cluster_at(mid)
        consider(mid, labels, centers, count)

        if count == target:
            print(f"  - Found preference={mid:.4f} yielding exactly {target} clusters")
            return labels, centers, mid

        if count < target:
            # Too few clusters -> need a less negative (higher) preference.
            low = mid
        else:
            # Too many clusters -> need a more negative (lower) preference.
            high = mid

    _, count, pref, labels, centers = best
    print(
        f"  - WARNING: could not reach exactly {target} clusters; "
        f"using closest preference={pref:.4f} yielding {count} clusters"
    )
    return labels, centers, pref


def flatten(items):
    """Yield items from any nested iterable."""
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
    if method == "hierarchical": # Never used in the experiments
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


def calculating_adjacency_parallel(clients_idxs, U): #use this for efficiency on FEMNIST
    nclients = len(clients_idxs)

    def compute_similarity(U1, U2):
        mul = np.clip(U1.T @ U2, a_min=-1.0, a_max=1.0)
        return np.min(np.arccos(mul)) * 180 / np.pi

    print("starting parallel")
    # Use joblib to parallelize the computation of the upper triangular part of the similarity matrix (excluding diagonal)
    with tqdm_joblib(tqdm(desc="My calculation", total=((nclients * (nclients - 1)) // 2))):
        results = list(
            Parallel(n_jobs=-1, backend="loky", pre_dispatch="40*n_jobs")(
                delayed(compute_similarity)(U[clients_idxs[idx1]], U[clients_idxs[idx2]])
                for idx1 in range(nclients)
                for idx2 in range(idx1 + 1, nclients)  # Skip diagonal by starting at idx1 + 1
            )
        )

    # Initialize the similarity matrix
    sim_mat = np.zeros((nclients, nclients))

    # Fill the upper triangular part of the matrix
    idx = 0
    for i in range(nclients):
        for j in range(i + 1, nclients):  # Skip diagonal
            sim_mat[i, j] = results[idx]
            sim_mat[j, i] = results[idx]  # Copy to the lower triangular part
            idx += 1

    return sim_mat


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()
