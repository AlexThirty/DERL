#!/bin/bash

# Check if the correct number of arguments are provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <device1>"
    exit 1
fi

# Assign input parameters to variables
DEVICE=$1

# Print the input parameters
echo "Device 1: $DEVICE"

python generate.py

#python train_phase0.py --device "$DEVICE"
# Define an array of modes
#MODES=("Derivative" "Output" "PINN" "Sobolev" "Forgetting")

MODES=("PINN")
# Loop through each mode and run the script
for MODE in "${MODES[@]}"; do
    echo "Running with mode: $MODE"
    python train_phase1.py --name extend --device "$DEVICE" --mode "$MODE"
    python train_phase2.py --name extend --device "$DEVICE" --mode "$MODE"
done

#for MODE in "${MODES[@]}"; do
#    echo "Running with mode: $MODE"
#    python train_phase1.py --name new --device "$DEVICE" --mode "$MODE"
#    python train_phase2.py --name new --device "$DEVICE" --mode "$MODE"
#done

#python train_pinn_full.py --device "$DEVICE"

#python test_joint.py --device "$DEVICE" --name new
python test_joint.py --device "$DEVICE" --name extend