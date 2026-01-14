# CoMask: Collaborative Masking in cluster-based Decentralized Learning

## Initialize the environment

This project uses [uv](https://github.com/astral-sh/uv) for Python package management.

### 1. Install uv

If you haven't already, install uv:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Create the environment

Sync the project dependencies:
```bash
uv sync
```

This will create a virtual environment and install all required packages specified in [pyproject.toml](pyproject.toml).

### 3. Configure Weights & Biases

Create a `.env` file in the project root with your W&B API key:
```bash
echo "WANDB_API_KEY=your_api_key_here" > .env
```

You can find your API key at https://wandb.ai/authorize

### 4. Running commands

All commands should be executed with the `uv run` prefix to use the project environment and load environment variables:
```bash
uv run --env-file .env -- <command>
```

For example:
```bash
uv run --env-file .env -- python scripts/my_fedml_script.py --cf configs/femnist_config.yaml
```


## Reproducing the experiments

### Running Weights & Biases Sweeps

This project uses W&B sweeps for hyperparameter tuning. Sweep configurations are located in the `sweeps_configs/` directory.

#### Creating a Sweep

1. Choose or create a sweep configuration file (e.g., `sweeps_configs/fedavg_sweep_femnist.yaml`)

2. Initialize the sweep:
   ```bash
   wandb sweep sweeps_configs/fedavg_sweep_femnist.yaml
   ```
   
   This will output a sweep ID in the format: `<entity>/<project>/<sweep_id>`

#### Running a Sweep Agent

Start one agent to execute the sweep runs:

```bash
wandb agent <entity>/<project>/<sweep_id>
```

### Additional notes on partitioning reproducibility

Due to differences in hardware architecture, the clustering procedure may produce a different number of clusters than reported in the original experiments. The original experiments were executed on a Jetson Orin AGX (an ARM-based device), and reproducing them on x86_64 architectures may result in variations in the clustering results for CIFAR-10 and HARBox datasets. This is due to architecture-specific differences in floating-point computations and numerical precision, which result in obtaining different client embeddings.


### Additional notes on HARBox

In order to reproduce the experiments on HARBox, you need to download the HARBox Dataset from [here](https://github.com/xmouyang/FL-Datasets-for-HAR) and put the large_scale_HARBox folder in this project main directory.



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
│   └── comask/                      # Main package implementation
│
├── sweeps_configs/                   # W&B sweep configurations
│   ├── fedavg_sweep_*.yaml          # FedAvg hyperparameter sweeps
│   ├── hermes_sweep_*.yaml          # HERMES algorithm sweeps
│   ├── scaffold_*_sweep.yaml        # SCAFFOLD algorithm sweeps
│   └── sweep_*_prune_*.yaml         # CoMask sweep varying pruning percentage
│   └── sweep_*_cons.yaml            # CoMask sweep varying voting percentage
│
├── pyproject.toml                    # Python project configuration
└── README.md                         # This file
```
