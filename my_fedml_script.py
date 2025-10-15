"""
FedML training script adapted to use the custom data partitioning
"""

import fedml
from fedml import FedMLRunner
from fedml.model.cv.resnet_cifar import resnet18_cifar
from torch.utils.data import DataLoader

# Import your custom adapter and data loading functions
from tesifedml.data.adapters import FedMLAdapter
from tesifedml.data.dataload import collate_fn, combine_batches, load_my_data
from tesifedml.data.partition import create_natural_test_sets, partition
from tesifedml.models.cnn import HARBox_CNN
from tesifedml.models.mobilenet import MobileNet

if __name__ == "__main__":
    # run = wandb.init()
    # assert run is not None

    # i = sys.argv.index("--cf")

    # # Load base config
    # with open(sys.argv[i + 1], "r") as f:
    #     config = yaml.safe_load(f)

    # # Override with sweep params
    # for key, value in wandb.config.items():
    #     if isinstance(value, dict):
    #         nest_conf = config[key]
    #         for nest_key, nest_value in value.items():
    #             nest_conf[nest_key] = nest_value
    #             print(f"Sweep setting: {nest_key} = {nest_value}")

    # # Save modified config
    # sweep_config_path = f"sweep_config_{run.id}.yaml"
    # print(sweep_config_path)
    # with open(sweep_config_path, "w") as f:
    #     yaml.dump(config, f)

    # # find and replace the filename
    # if "--cf" in sys.argv:
    #     i = sys.argv.index("--cf")
    #     if i + 1 < len(sys.argv):
    #         sys.argv[i + 1] = sweep_config_path

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

    dataset = FedMLAdapter.create_fedml_data_structure(
        train_partitions=train_partitions,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        transform_fn=transforms_tv,
        test_partitions=test_partitions,
        batch_size=args.batch_size,
        num_workers=getattr(args, "num_workers", 0),
        log_distributions=True,
    )

    # create dataloader for separate test if exists
    if separate_test_set is not None:
        separate_test_set.set_transform(transforms_tv)
        full_batch = False
        if args.batch_size == -1:
            full_batch = True
            args.batch_size = 32

        sep_test_loader = DataLoader(
            separate_test_set,  # type:ignore
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=getattr(args, "num_workers", 0),
        )

        if full_batch:
            sep_test_loader = combine_batches(sep_test_loader)

        args.sep_test_loader = sep_test_loader

    # Extract number of classes for model output dimension
    output_dim = len(set(train_dataset["label"]))

    print("✓ Loaded custom partitioned data:")
    print(f"  - Number of clients: {len(train_partitions)}")
    print(f"  - Training samples: {dataset[0]}")
    print(f"  - Test samples: {dataset[1]}")
    print(f"  - Output dimension (classes): {output_dim}")

    if args.model == "resnet18":
        model = resnet18_cifar()

    elif args.model == "har_cnn":
        model = HARBox_CNN()

    elif args.model == "mobilenet":
        model = MobileNet(class_num=62)
    else:
        model = fedml.model.create(args, output_dim)

    # start training
    fedml_runner = FedMLRunner(args, device, dataset, model)
    fedml_runner.run()
