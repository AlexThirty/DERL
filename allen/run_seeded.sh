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
SEEDS=(30 42 2025)

# Loop through each mode and run the script
for MODE in "${MODES[@]}"; do
    echo "Running with mode: $MODE"
    for SEED in "${SEEDS[@]}"; do
        echo "Running with seed: $SEED"
        python train_seeded.py --name "$NAME" --device "$DEVICE" --mode "$MODE" --seed "$SEED" &
    done
done