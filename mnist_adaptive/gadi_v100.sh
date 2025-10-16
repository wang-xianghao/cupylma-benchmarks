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
BATCH_START=200
BATCH_END=60000
BATCH_SLOPE=1.5
SLICE_SIZE=1024
LEARNING_RATE=1.0
EPOCHS=30
RESULT_NAME=lma_{$BATCH_START}_{$BATCH_END}.csv

legate --gpus 1 $WORKDIR/bench.py \
    --batch-start $BATCH_START --batch-end $BATCH_END --batch-slope=$BATCH_SLOPE --slice-size $SLICE_SIZE --epochs $EPOCHS \
    --learning-rate $LEARNING_RATE \
    -o $RESULTDIR/$RESULT_NAME