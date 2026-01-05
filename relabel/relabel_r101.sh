#!/bin/bash


NOW=$(date '+%F-%H-%M-%S')
PARTITION="RTXA6000"
RUNVIA=python
NODES=1
NTASKS=1
CPUS_PER_TASK=24
GPUS_PER_TASK=1
MEM=64G

export LOGLEVEL=INFO
        
srun -K \
    --job-name="PRISM Relabeling ImNet 10" \
    --partition=RTXA6000,A100-PCI \
    --nodes=$NODES \
    --ntasks=$NTASKS \
    --cpus-per-task=$CPUS_PER_TASK \
    --gpus-per-task=$GPUS_PER_TASK \
    --gpu-bind=none \
    --mem=$MEM \
    --time=3-00:00 \
    --container-image=/enroot/nvcr.io_nvidia_pytorch_23.12-py3.sqsh \
    --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds,/ds-sds:/ds-sds,"`pwd`":"`pwd`" \
    --container-workdir="`pwd`" \
    python generate_soft_label_r101.py \
    -b 100 \
    -j 8 \
    --epochs 300 \
    --fkd-seed 42 \
    --input-size 224 \
    --min-scale-crops 0.25 \
    --max-scale-crops 1 \
    --use-fp16 --candidate-number 4 \
    --fkd-path FKD_prism_r101 \
    --mode 'fkd_save' \
    --mix-type 'cutmix' \
    --data /netscratch/bmoser/sre2l/recover/syn_data/full_r101_10
