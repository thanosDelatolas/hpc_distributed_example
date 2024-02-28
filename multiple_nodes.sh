#!/bin/sh
### General options
#BSUB -q gpuv100
#BSUB -R "select[gpu32gb]"
#BSUB -J multiple_nodes
#BSUB -n 16
#BSUB -R "span[ptile=8]"
#BSUB -gpu "num=2:mode=exclusive_process"
#BSUB -W 02:00
#BSUB -R "rusage[mem=10GB]"
#BSUB -o multiple_nodes_%J.out
#BSUB -e multiple_nodes_%J.err

# Get the list of nodes-addresses
List=$(cat $LSB_DJOB_HOSTFILE | uniq )

# Set the port for the rendezvous protocol
PORT=29400 

module load python3/3.10.12
module load cuda/11.8
source ~/dist-training/bin/activate

echo "Running on nodes $List"
blaunch -z "$List" torchrun  --nnodes=2 --nproc_per_node=2 \
    --rdzv_id=$LSB_JOBID --rdzv_backend=c10d \
    --rdzv_endpoint=$HOSTNAME:$PORT \
    train.py