"""
Expand a W&B sweep YAML into concrete, fully-merged config files.

On network-isolated compute nodes the W&B sweep agent cannot run (no internet),
so `wandb.config` is empty and grid parameters never get applied. This script
does the grid expansion offline: it takes the base config + a sweep YAML and
writes one merged config per grid combination, ready to pass to the training
script with `--cf`. Run it on the login node (or inside the job — it needs no
network).

Usage:
    python scripts/prepare_offline_configs.py \
        --sweep sweeps_configs/fedavg_dir_0.1_cifar_grid.yaml \
        --base  configs/cifar_config.yaml \
        --manifest out/_manifest.txt

Prints the generated config paths (one per line) and, if --manifest is given,
writes the same list there for a runner loop to consume.
"""

import argparse
import itertools
import os
from copy import deepcopy

import yaml


def expand_grid(sweep_params):
    """Turn the nested sweep `parameters` block into a list of override dicts.

    Expected structure:
        {section: {"parameters": {key: {"values": [...]}|{"value": x}}}}
    Returns (combinations, leaves) where each combination maps
    (section, key) -> chosen value, and leaves lists (section, key, values).
    """
    leaves = []  # (section, key, [values])
    for section, sect_body in sweep_params.items():
        params = (sect_body or {}).get("parameters", {})
        for key, spec in params.items():
            if "values" in spec:
                values = spec["values"]
            elif "value" in spec:
                values = [spec["value"]]
            else:
                raise ValueError(f"param {section}.{key} has neither 'values' nor 'value'")
            leaves.append((section, key, values))

    keys = [(s, k) for s, k, _ in leaves]
    value_lists = [v for _, _, v in leaves]
    combinations = [dict(zip(keys, combo)) for combo in itertools.product(*value_lists)]
    return combinations, leaves


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep", required=True, help="sweep YAML to expand")
    ap.add_argument("--base", required=True, help="base config YAML to merge into")
    ap.add_argument("--out-dir", default="configs/generated", help="where to write merged configs")
    ap.add_argument("--manifest", default=None, help="optional file to write the list of generated configs")
    args = ap.parse_args()

    with open(args.sweep) as f:
        sweep = yaml.safe_load(f)
    with open(args.base) as f:
        base = yaml.safe_load(f)

    sweep_name = sweep.get("name") or os.path.splitext(os.path.basename(args.sweep))[0]
    project = sweep.get("project")
    entity = sweep.get("entity")

    combinations, leaves = expand_grid(sweep.get("parameters", {}))
    varying = {(s, k) for s, k, vals in leaves if len(vals) > 1}

    out_root = os.path.join(args.out_dir, sweep_name)
    os.makedirs(out_root, exist_ok=True)

    generated = []
    for idx, combo in enumerate(combinations):
        cfg = deepcopy(base)
        for (section, key), value in combo.items():
            cfg.setdefault(section, {})
            cfg[section][key] = value

        # Build a descriptive, unique run name.
        if varying:
            parts = [f"{k}={combo[(s, k)]}" for (s, k) in sorted(varying)]
            run_name = f"{sweep_name}__" + "_".join(parts)
        elif len(combinations) == 1:
            run_name = sweep_name
        else:
            run_name = f"{sweep_name}_{idx:03d}"

        tracking = cfg.setdefault("tracking_args", {})
        tracking["enable_wandb"] = True
        if project:
            tracking["wandb_project"] = project
        if entity:
            tracking["wandb_entity"] = entity
        tracking["run_name"] = run_name

        out_path = os.path.join(out_root, f"combo_{idx:03d}.yaml")
        with open(out_path, "w") as f:
            yaml.dump(cfg, f, sort_keys=False)
        generated.append(out_path)
        print(out_path)

    if args.manifest:
        os.makedirs(os.path.dirname(args.manifest) or ".", exist_ok=True)
        with open(args.manifest, "w") as f:
            f.write("\n".join(generated) + "\n")


if __name__ == "__main__":
    main()
