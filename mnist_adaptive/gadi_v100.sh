#!/bin/bash
#PBS -q gpuvolta
#PBS -j oe
#PBS -l walltime=01:00:00,mem=120GB
#PBS -l wd
#PBS -l ncpus=24
#PBS -l ngpus=2
#PBS -l storage=scratch/um09+scratch/c07+scratch/aw81
#PBS -M u7321615@anu.edu.au
#PBS -m abe
#

WORKDIR=./mnist_adaptive
RESULTDIR=$WORKDIR/results

# Configure benchmark
BATCH_START=256
BATCH_END=8192
SLICE_SIZE=1024
LEARNING_RATE=0.005
EPOCHS=10
RESULT_NAME=lma_{$BATCH_START}_{$BATCH_END}.csv

legate --gpus 1 $WORKDIR/bench.py \
    --batch-start $BATCH_START --batch-end $BATCH_END --slice-size $SLICE_SIZE --epochs $EPOCHS \
    -o $RESULTDIR/$RESULT_NAME