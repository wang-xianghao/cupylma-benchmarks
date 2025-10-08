#!/bin/bash
#PBS -q gpuvolta
#PBS -j oe
#PBS -l walltime=00:30:00,mem=120GB
#PBS -l wd
#PBS -l ncpus=24
#PBS -l ngpus=2
#PBS -l storage=scratch/um09+scratch/c07+scratch/aw81
#PBS -M u7321615@anu.edu.au
#PBS -m abe
#

# lma
legate --gpus 1 mnist/bench.py --optim lma --batch_size 1024 --lr 0.1 --slice_size 256 --epochs 2 \
    --out mnist/results/lma_32.csv