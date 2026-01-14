# CoMask: Collaborative Masking in cluster-based Decentralized Learning

## Repository Structure

```
Tesi-FedML/
├── configs/                          # Configuration files for different experiments
│   ├── cifar_config.yaml            # CIFAR dataset configuration
│   ├── femnist_config.yaml          # FEMNIST dataset configuration
│   ├── har_config.yaml              # HAR dataset configuration
│   ├── scaffold_femnist_config.yaml # SCAFFOLD algorithm config for FEMNIST
│   └── scaffold_har_config.yaml     # SCAFFOLD algorithm config for HAR
│
├── scripts/                          # Executable scripts for running experiments
│   ├── cluster_prune_script.py      # Cluster-based pruning experiments
│   ├── cluster_svd_script.py        # SVD-based clustering experiments
│   ├── my_fedml_script.py           # Main FedML experiment runner
│   └── personal_pruning_script.py   # Personal pruning experiments
│
├── src/                              # Source code directory
│   └── comask/                   # Main package implementation
│
├── sweeps_configs/                   # W&B sweep configurations
│   ├── fedavg_sweep_*.yaml          # FedAvg hyperparameter sweeps
│   ├── hermes_sweep_*.yaml          # HERMES algorithm sweeps
│   ├── scaffold_*_sweep.yaml        # SCAFFOLD algorithm sweeps
│   └── sweep_*_prune_*.yaml         # CoMask sweep varying pruning percetage
│   └── sweep_*_cons.yaml            # CoMask sweep varying voting percentage
│
├── pyproject.toml                    # Python project configuration
└── README.md                         # This file
```

