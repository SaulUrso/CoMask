"""
FedML training script adapted to use the custom data partitioning
"""

import sys

import fedml
import yaml
from fedml import FedMLRunner
from fedml.model.cv.resnet_cifar import resnet18_cifar

import wandb

# Import your custom adapter and data loading functions
from tesifedml.data.adapters import FedMLAdapter
from tesifedml.data.dataload import load_my_data
from tesifedml.data.partition import partition
from tesifedml.models.cnn import HARBox_CNN
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

    hf_dataset, transforms_tv = load_my_data(args.dataset)
    train_dataset = hf_dataset["train"]
    test_dataset = hf_dataset["test"]

    num_clients = args.client_num_in_total

    method_name = getattr(args, "method_name", "dirichlet")

    train_partitions = partition(
        train_dataset, args.partition_method, method_name=method_name, client_num=num_clients, feature_col="user_id"
    )

    test_partitions = partition(
        test_dataset, args.partition_method, method_name=method_name, client_num=num_clients, feature_col="user_id"
    )


    args.client_num_in_total = len(train_partitions)

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

    # Extract number of classes for model output dimension
    output_dim = len(set(train_dataset["label"]))

    print("✓ Loaded custom partitioned data:")
    print(f"  - Number of clients: {len(train_partitions)}")
    print(f"  - Training samples: {dataset[0]}")
    print(f"  - Test samples: {dataset[1]}")
    print(f"  - Output dimension (classes): {output_dim}")

    # load model
    try:
        model = fedml.model.create(args, output_dim)
    except Exception:
        if args.model == "resnet18":
            model = resnet18_cifar()

        elif args.model == "har_cnn":
            model = HARBox_CNN()
        else:
            raise Exception("Model not recognized")

    # start training
    fedml_runner = FedMLRunner(args, device, dataset, model)
    fedml_runner.run()
