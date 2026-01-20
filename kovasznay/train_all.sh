#!/bin/bash

python train.py --device cuda:3 --mode Derivative --no-grid &
python train.py --device cuda:3 --mode Output --no-grid &
python train.py --device cuda:2 --mode Sobolev --no-grid &
python train.py --device cuda:2 --mode PINN+Output --no-grid &