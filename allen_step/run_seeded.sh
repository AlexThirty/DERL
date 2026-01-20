#!/bin/bash

# Check if the correct number of arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <name> <device>"
    exit 1
fi

# Assign input parameters to variables
NAME=$1
DEVICE=$2

# Print the input parameters
echo "Name: $NAME"
echo "Device: $DEVICE"

# Define an array of modes
MODES=("Derivative" "Output" "PINN" "Sobolev" "PINN+Output")
SEEDS=(0.001 0.01 0.05 0.1)

# Loop through each mode and run the script

for SEED in "${SEEDS[@]}"; do
    echo "Running with seed: $SEED"
    python train_seeded.py --name "$NAME" --device "$DEVICE" --mode Sobolev --step "$SEED" &
done
