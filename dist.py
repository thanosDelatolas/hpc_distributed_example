import os
import numpy as np

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import torchvision
import torchvision.transforms as transforms


def multi_node_setup():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def cleanup():
    dist.destroy_process_group()


def create_data_loaders(global_rank):
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
        download=True, transform=transform
    )
    if global_rank == 0:
        print(f'Train dataset size: {len(train_dataset)}')
        
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    
    def worker_init_fn(worker_id): 
        return np.random.seed(torch.initial_seed()%(2**31) + worker_id + global_rank*100)

    train_loader = DataLoader(train_dataset, batch_size=4, sampler=train_sampler,
        worker_init_fn=worker_init_fn, drop_last=True, pin_memory=True
    )

    val_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
        download=True, transform=transform
    )
    val_sampler = DistributedSampler(val_dataset, shuffle=False)

    val_loader = DataLoader(val_dataset, batch_size=4, sampler=val_sampler)

    return train_loader, val_loader