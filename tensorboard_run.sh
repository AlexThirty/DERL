#!/bin/bash
myabsfile=$(get_abs_filename "./$0")

tensorboard --logdir=$1