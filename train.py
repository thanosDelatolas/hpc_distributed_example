import os 
import random
from tqdm import tqdm

# torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision


# distributed imports
from dist import multi_node_setup, cleanup, create_data_loaders
from torch.nn.parallel import DistributedDataParallel as DDP

# others
import matplotlib.pyplot as plt
import numpy as np


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def seed_everything(seed=29102910):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

os.makedirs('assets', exist_ok=True)

def imshow(img, global_rank):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.figure()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig(f'assets/samples_gpu_{global_rank}', bbox_inches='tight')
    plt.show()


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train(local_rank, global_rank, world_size):
    
    train_loader, val_loader = create_data_loaders(global_rank)
    if global_rank == 0:
        """
        The main idea is that for each GPU (global_rank),
        A thread is created.

        NOTE: this will only be executed once.
        I usually log in only one GPU and save my best model.
        """
        print(f'Validation loader: {len(val_loader)}')
    print(f'Training loader at GPU:{global_rank} with size: {len(train_loader)}')

    # show some images
    dataiter = iter(train_loader)
    images, labels = next(dataiter)
    imshow(torchvision.utils.make_grid(images), global_rank)

    net = Net()
    net.to(local_rank)

    # model to distributed mode 
    net = DDP(net, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False) 

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    best_val_loss = 1e10
    for epoch in range(50): 
        train_loader.sampler.set_epoch(epoch) # crucial for randomness!
        net.train()
        train_loss = 0.0

        # NOTE that the number of iteration depends on the number of GPUs ;)
        for images, labels in tqdm(train_loader, f'[EPOCH {epoch+1}/50]'): 
            images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)
            
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        
        net.eval()
        total = 0
        correct = 0
        val_loss = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)

                outputs = net(images)
                val_loss += criterion(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
        val_loss /= len(val_loader)
        
        if val_loss <= best_val_loss and global_rank == 0:
            best_val_loss = val_loss
            # NOTE: to access the non distributed model you use net.module
            torch.save(net.module.state_dict(), 'model.pth')

        """
        NOTE: this command is not necessary now.
        You use it if you want to wait for all GPUs to be there and then continue
        """
        torch.cuda.synchronize()

def main():
    """
    local rank: the GPU number in each node
    global_rank: the GPU number across all nodes
    world_size: number of GPUs
    """
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ['WORLD_SIZE'])
    multi_node_setup()
    print(f'local_rank: {local_rank}, global_rank: {global_rank}, world_size:{world_size}')

    torch.cuda.set_device(local_rank)
    seed_everything()

    try: 
        train(local_rank, global_rank, world_size)
    finally:
        cleanup()


if __name__ == '__main__':
    main()