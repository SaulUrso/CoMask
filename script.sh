#!/bin/bash

CONFIG_DIR="./configs"

for filename in "$CONFIG_DIR"/*; do
    if [[ -f "$filename" ]]; then
        uv run my_fedml_script.py --cf "$filename"
    fi
done