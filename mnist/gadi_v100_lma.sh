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

WORKDIR=./mnist/
RESULTDIR=$WORKDIR/results

# Configure benchmark
BATCH_SIZE=4096
SLICE_SIZE=1024
LEARNING_RATE=0.005
EPOCHS=5
RESULT_NAME=lma_$BATCH_SIZE.csv

legate --gpus 1 $WORKDIR/bench.py --optim lma \
    --batch_size $BATCH_SIZE --lr $LEARNING_RATE --slice_size $SLICE_SIZE --epochs $EPOCHS \
    --out $RESULTDIR/$RESULT_NAME