"""
FedML training script adapted to use the custom data partitioning
"""

import fedml
from fedml import FedMLRunner
from tesifedml.models.resnet_cifar import resnet18_cifar
from torch.utils.data import DataLoader

# Import your custom adapter and data loading functions
from tesifedml.clustering.clustering import perform_clustering_svd
from tesifedml.data.adapters import FedMLAdapter
from tesifedml.data.dataload import collate_fn, load_my_data
from tesifedml.data.partition import create_natural_test_sets, partition
from tesifedml.models.cnn import HARBox_CNN
from tesifedml.models.mobilenet import MobileNet
from tesifedml.servers.cluster_pruning_trainer import PruneClusterAPI
from tesifedml.utils import initialize_and_override_config

if __name__ == "__main__":
    initialize_and_override_config()

    args = fedml.init()

    # if wandb.run is not None:
    #     for key, value in wandb.config.items():
    #         if hasattr(args, key):
    #             setattr(args, key, value)
    #             print(f"Sweep override: {key} = {value}")

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

    # no need to partition the test_only dataset

    validation_split = getattr(args, "validation_split", None)

    dataset = FedMLAdapter.create_fedml_data_structure(
        train_partitions=train_partitions,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        transform_fn=transforms_tv,
        test_partitions=test_partitions,
        batch_size=args.batch_size,
        num_workers=getattr(args, "num_workers", 0),
        log_distributions=False,
        validation_split=validation_split,
    )

    # create dataloader for separate test if exists
    if separate_test_set is not None:
        separate_test_set.set_transform(transforms_tv)
        full_batch = False
        if args.batch_size == -1:
            full_batch = True
            args.batch_size = 128

        sep_test_loader = DataLoader(
            separate_test_set,  # type:ignore
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=getattr(args, "num_workers", 0),
        )

        # if full_batch:
        #     sep_test_loader = combine_batches(sep_test_loader)

        args.sep_test_loader = sep_test_loader

    # Extract number of classes for model output dimension
    output_dim = len(set(train_dataset["label"]))

    print("✓ Loaded custom partitioned data:")
    print(f"  - Number of clients: {len(train_partitions)}")
    print(f"  - Training samples: {dataset[0]}")
    print(f"  - Test samples: {dataset[1]}")
    print(f"  - Output dimension (classes): {output_dim}")

    # start clustering
    cluster_labels, cluster_centers, sim_mat = perform_clustering_svd(
        [client_data.with_transform(transforms_tv) for client_data in train_partitions], args
    )
    args.cluster_indexes = cluster_labels

    print(f"  - Number of clusters: {len(set(cluster_labels))}")

    if args.model == "resnet18":
        model = resnet18_cifar()

    elif args.model == "har_cnn":
        model = HARBox_CNN()

    elif args.model == "mobilenet":
        model = MobileNet(class_num=62)
    else:
        model = fedml.model.create(args, output_dim)

    # # this should work, in reality all arguments but last are ignored
    fedml_runner = FedMLRunner(
        args,
        device,
        dataset,
        model,
        algorithm_flow=PruneClusterAPI(args, device, dataset, model),  # type: ignore
    )
    fedml_runner.run()
