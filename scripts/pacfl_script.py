"""
PACFL baseline training script.

Same data/partition plumbing as cluster_svd_script.py, but clusters clients once via
agglomerative hierarchical clustering on the min-principal-angle distance matrix (PACFL) and
trains with PACFLClusterAPI (uniform random client sampling + per-cluster FedAvg).
"""

import fedml
from fedml import FedMLRunner
from fedml.model.cv.resnet_cifar import resnet18_cifar
from torch.utils.data import DataLoader

import wandb

# Import your custom adapter and data loading functions
from comask.clustering.clustering import perform_clustering_svd_hc
from comask.data.adapters import FedMLAdapter
from comask.data.dataload import collate_fn, load_my_data
from comask.data.partition import create_natural_test_sets, partition
from comask.models.cnn import HARBox_CNN
from comask.models.mobilenet import MobileNet
from comask.servers.pacfl_trainer import PACFLClusterAPI
from comask.utils import initialize_and_override_config

if __name__ == "__main__":
    initialize_and_override_config()

    args = fedml.init()

    device = fedml.device.get_device(args)
    test_only_users = getattr(args, "test_only_users", None)

    hf_dataset, transforms_tv = load_my_data(args.dataset, test_only_users=test_only_users)
    train_dataset = hf_dataset["train"]
    test_dataset = hf_dataset["test"]

    separate_test_set = hf_dataset["test_only"] if test_only_users is not None else None

    num_clients = args.client_num_in_total

    method_name = getattr(args, "method_name", "dirichlet")
    feature_col = getattr(args, "feature_col", None)

    train_partitions = partition(
        train_dataset, args.partition_method, method_name=method_name, client_num=num_clients, feature_col=feature_col
    )

    if args.partition_method == "natural" and feature_col is not None:
        test_partitions = create_natural_test_sets(train_partitions, test_dataset, feature_column=feature_col)
    else:
        test_partitions = partition(
            test_dataset,
            args.partition_method,
            method_name=method_name,
            client_num=num_clients,
            feature_col=feature_col,
        )

    args.client_num_in_total = len(train_partitions)

    client_num_per_round = getattr(args, "client_num_per_round", None)
    if client_num_per_round is not None:
        client_num_per_round = float(client_num_per_round)
        if client_num_per_round < 1:
            absolute_clients = max(1, int(round(args.client_num_in_total * client_num_per_round)))
            args.client_num_per_round = absolute_clients
            print(
                f"  - client_num_per_round interpreted as fraction ({client_num_per_round}) -> {absolute_clients} clients"
            )

    dataset = FedMLAdapter.create_fedml_data_structure(
        train_partitions=train_partitions,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        transform_fn=transforms_tv,
        test_partitions=test_partitions,
        batch_size=args.batch_size,
        num_workers=getattr(args, "num_workers", 0),
        log_distributions=False,
    )

    # create dataloader for separate test if exists
    if separate_test_set is not None:
        separate_test_set.set_transform(transforms_tv)
        if args.batch_size == -1:
            args.batch_size = 128

        sep_test_loader = DataLoader(
            separate_test_set,  # type:ignore
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=getattr(args, "num_workers", 0),
        )

        args.sep_test_loader = sep_test_loader

    # Extract number of classes for model output dimension
    output_dim = len(set(train_dataset["label"]))

    print("✓ Loaded custom partitioned data:")
    print(f"  - Number of clients: {len(train_partitions)}")
    print(f"  - Training samples: {dataset[0]}")
    print(f"  - Test samples: {dataset[1]}")
    print(f"  - Output dimension (classes): {output_dim}")

    single_cluster = getattr(args, "single_cluster", False)
    if isinstance(single_cluster, str):
        single_cluster = single_cluster.lower() in {"1", "true", "yes", "y", "on"}

    if single_cluster:
        cluster_labels = [0] * len(train_partitions)
        args.cluster_indexes = cluster_labels
        print("  - single_cluster=True, skipping clustering and assigning all clients to cluster 0")
    else:
        # PACFL one-shot clustering: hierarchical cut on the principal-angle distance matrix
        cluster_labels, _, _ = perform_clustering_svd_hc(
            [client_data.with_transform(transforms_tv) for client_data in train_partitions], args
        )
        args.cluster_indexes = cluster_labels

    n_clusters_found = len(set(cluster_labels))
    print(f"  - Number of clusters: {n_clusters_found}")

    # Log the resulting cluster count so the beta sweep is interpretable.
    if getattr(args, "enable_wandb", False) and wandb.run is not None:
        wandb.log({"Clustering/NumClusters": n_clusters_found})
        wandb.run.summary["Clustering/NumClusters"] = n_clusters_found

    if args.model == "resnet18":
        model = resnet18_cifar()
    elif args.model == "har_cnn":
        model = HARBox_CNN()
    elif args.model == "mobilenet":
        model = MobileNet(class_num=62)
    else:
        model = fedml.model.create(args, output_dim)

    fedml_runner = FedMLRunner(
        args, device, dataset, model, algorithm_flow=PACFLClusterAPI(args, device, dataset, model)
    )  # type: ignore
    fedml_runner.run()
