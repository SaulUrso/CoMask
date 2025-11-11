import sys

import yaml

import wandb


def initialize_and_override_config():
    run = wandb.init()
    assert run is not None

    i = sys.argv.index("--cf")

    # Load base config
    with open(sys.argv[i + 1], "r") as f:
        config = yaml.safe_load(f)

    # Override with sweep params
    prune_percent = None
    consolidation_percentage = None

    for key, value in wandb.config.items():
        if isinstance(value, dict):
            nest_conf = config[key]
            for nest_key, nest_value in value.items():
                if nest_key == "prune_percent":
                    prune_percent = nest_value
                elif nest_key == "consolidation_percentage":
                    consolidation_percentage = nest_value
                nest_conf[nest_key] = nest_value
                print(f"Sweep setting: {nest_key} = {nest_value}")

    if prune_percent != consolidation_percentage and prune_percent is not None and consolidation_percentage is not None:
        print(
            f"Don't do this run as prune_percent={prune_percent} and consolidation_percentage={consolidation_percentage}"
        )
        exit(0)

    # Save modified config
    sweep_config_path = f"sweep_config_{run.id}.yaml"
    print(sweep_config_path)
    with open(sweep_config_path, "w") as f:
        yaml.dump(config, f)

    # find and replace the filename
    if "--cf" in sys.argv:
        i = sys.argv.index("--cf")
        if i + 1 < len(sys.argv):
            sys.argv[i + 1] = sweep_config_path
