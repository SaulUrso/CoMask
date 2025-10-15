import sys
import wandb
import yaml

def initialize_and_override_config():
    run = wandb.init()
    assert run is not None

    i = sys.argv.index("--cf")

    # Load base config
    with open(sys.argv[i + 1], "r") as f:
        config = yaml.safe_load(f)

    # Override with sweep params
    for key, value in wandb.config.items():
        if isinstance(value, dict):
            nest_conf = config[key]
            for nest_key, nest_value in value.items():
                nest_conf[nest_key] = nest_value
                print(f"Sweep setting: {nest_key} = {nest_value}")

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