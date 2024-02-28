# Distributed Training on HPC
This repo presents a simple tutorial for distributed training in HPC.
The `single_node.sh` trains the network with two gpus in the same node 
while the `multiple_nodes.sh` trains the same network with two GPUs at each node.

## Resources
* [DTU tutorial](https://www.hpc.dtu.dk/?page_id=4800)
* [PyTorch tutorial](https://pytorch.org/tutorials/intermediate/ddp_series_multinode.html)
* [PyTorch Distributed Data Parallel](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)

## Installation
```bash
module load python3/3.10.12
module load cuda/11.8
python3 -m venv ~/dist-training
source ~/dist-training/bin/activate
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```