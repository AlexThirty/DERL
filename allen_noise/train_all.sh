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

# Loop through each mode and run the script
for MODE in "${MODES[@]}"; do
    echo "Running with mode: $MODE"
    python train.py --name "$NAME" --device "$DEVICE" --mode "$MODE" &
done