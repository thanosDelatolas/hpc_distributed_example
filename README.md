# Distributed Training on HPC
This repo presents a simple tutorial for distributed training in HPC.
The `single_node.sh` trains the network with two gpus in the same node 
while the `multiple_nodes.sh` trains the same network with two GPUs at each node.

## Notes
* Crucial parameters in **torchrun**: 
    * `--nnodes=number of nodes` 
    * `--nproc_per_node=number of GPUs per node`
* Crucial parameters in the **multiple_nodes.sh** script:
    * `#BSUB -n 16`: Define the number of total CPU cores. Must be at least 4 for each GPU
    * `#BSUB -R "span[ptile=8]"`: The number of cores in **each** node
    * Note that we don't explicity ask for a number of nodes but the number of nodes is calculated as the number of total cores devided by the number of cores in each node.
    * Unlike the script **single_node.sh**, the command `#BSUB -R "span[hosts=1]"` is omitted as we want more than one hosts. 
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