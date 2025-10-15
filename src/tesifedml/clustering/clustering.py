import copy
from typing import Iterable

import numpy as np
import torch
from sklearn.cluster import AffinityPropagation, AgglomerativeClustering


def perform_clustering_svd(client_data_list, args):
    U_clients = []

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
    sim_mat = -calculating_adjacency(range(args.num_clients), U_clients)
    np.fill_diagonal(sim_mat, 0)  # Set diagonal to 0 for self-similarity

    cluster_labels, cluster_centers = perform_clustering(sim_mat, method="affinity", preference=args.preference)

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


def hierarchical_clustering(A, thresh=1.5, linkage="maximum"):
    """
    Hierarchical Clustering Algorithm. It is based on single linkage, finds the minimum element and merges
    rows and columns replacing the minimum elements. It is working on adjacency matrix.

    :param: A (adjacency matrix), thresh (stopping threshold)
    :type: A (np.array), thresh (int)

    :return: clusters
    """
    label_assg = {i: i for i in range(A.shape[0])}

    step = 0
    while A.shape[0] > 1:
        np.fill_diagonal(A, -np.NINF)
        # print(f'step {step} \n {A}')
        step += 1
        ind = np.unravel_index(np.argmin(A, axis=None), A.shape)

        if A[ind[0], ind[1]] > thresh:
            print("Breaking HC")
            break
        else:
            np.fill_diagonal(A, 0)
            if linkage == "maximum":
                Z = np.maximum(A[:, ind[0]], A[:, ind[1]])
            elif linkage == "minimum":
                Z = np.minimum(A[:, ind[0]], A[:, ind[1]])
            elif linkage == "average":
                Z = (A[:, ind[0]] + A[:, ind[1]]) / 2

            A[:, ind[0]] = Z
            A[:, ind[1]] = Z
            A[ind[0], :] = Z
            A[ind[1], :] = Z
            A = np.delete(A, (ind[1]), axis=0)
            A = np.delete(A, (ind[1]), axis=1)

            if type(label_assg[ind[0]]) is list:
                label_assg[ind[0]].append(label_assg[ind[1]])
            else:
                label_assg[ind[0]] = [label_assg[ind[0]], label_assg[ind[1]]]

            label_assg.pop(ind[1], None)

            temp = []
            for k, v in label_assg.items():
                if k > ind[1]:
                    kk = k - 1
                    vv = v
                else:
                    kk = k
                    vv = v
                temp.append((kk, vv))

            label_assg = dict(temp)

    clusters = []
    for k in label_assg.keys():
        if type(label_assg[k]) is list:
            clusters.append(list(flatten(label_assg[k])))
        elif type(label_assg[k]) is int:
            clusters.append([label_assg[k]])

    return clusters


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
