#!/bin/bash

# Get the directory of the current script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default Apptainer image and script path relative to the script's location
DEFAULT_IMAGE="/net/software/lab/chai/chai_apptainer/chai.sif"
DEFAULT_SCRIPT="$SCRIPT_DIR/predict.py"

# Initialize variables with default values
IMAGE="$DEFAULT_IMAGE"
SCRIPT="$DEFAULT_SCRIPT"
OTHER_ARGS=()

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --image) IMAGE="$2"; shift ;;  # If --image is provided, use the specified path
        --script) SCRIPT="$2"; shift ;;  # If --script is provided, use the specified path
        *) OTHER_ARGS+=("$1") ;;  # Add all other arguments to OTHER_ARGS array
    esac
    shift
done

# Run the command with Apptainer
apptainer run --slurm --nv "$IMAGE" "$SCRIPT" "${OTHER_ARGS[@]}"
