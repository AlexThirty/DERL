#!/bin/bash

device=$1

name=("new" "extend")
mode=("Derivative" "Sobolev" "Output")
#python3 train_pinn.py --device $device &
#python3 train_phase0.py --mode $mode --device $device

#echo "Training phase 1, name $n"
#python3 train_phase1.py --name new --device $device --mode Derivative &
#python3 train_phase1.py --name extend --device $device --mode Derivative

#python3 train_phase1.py --name new --device $device --mode Sobolev &
#python3 train_phase1.py --name extend --device $device --mode Sobolev

#python3 train_phase1.py --name new --device $device --mode Output &
#python3 train_phase1.py --name extend --device $device --mode Output


#python3 train_phase2.py --name new --device $device --mode Derivative &
#python3 train_phase2.py --name extend --device $device --mode Derivative

#python3 train_phase2.py --name extend --device $device --mode Sobolev &
#python3 train_phase2.py --name new --device $device --mode Sobolev

#python3 train_phase2.py --name new --device $device --mode Output &
#python3 train_phase2.py --name extend --device $device --mode Output

python3 train_phase1.py --name extend --device $device --mode PINN
python3 train_phase2.py --name extend --device $device --mode PINN


done
