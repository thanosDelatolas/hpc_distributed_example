#!/bin/sh
#BSUB -q gpuv100
#BSUB -J single_node
### -- specify that the cores must be on the same host --
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=2:mode=exclusive_process"
#BSUB -W 02:00
#BSUB -R "rusage[mem=10GB]"
#BSUB -o single_node_%J.out
#BSUB -e single_node_%J.err

module load python3/3.10.12
module load cuda/11.8
source ~/dist-training/bin/activate

torchrun --standalone --nproc_per_node=2 train.py