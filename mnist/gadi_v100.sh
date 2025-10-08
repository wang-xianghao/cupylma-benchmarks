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
legate --gpus 1 mnist/bench.py --optim lma --batch_size 32 --slice_size 32 --epochs 20 \
    --out mnist_bench/lma_32.csv

# adam
legate --gpus 1 mnist/bench.py --optim adam --batch_size 32 --epochs 100 \
    --out mnist_bench/adam_32.csv