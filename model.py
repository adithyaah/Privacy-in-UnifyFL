from collections import OrderedDict
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from datasets.utils.logging import disable_progress_bar
import flwr
from flwr.client import NumPyClient
from flwr.common import Metrics, Context
from flwr_datasets import FederatedDataset

import psutil
import os

# Device setup
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training on {DEVICE}")
print(f"Flower {flwr.__version__} / PyTorch {torch.__version__}")
disable_progress_bar()

# Constants
NUM_CLIENTS = 10
BATCH_SIZE = 32

# Load and prepare datasets
def load_datasets(partition_id: int):
    fds = FederatedDataset(dataset="cifar10", partitioners={"train": NUM_CLIENTS})
    partition = fds.load_partition(partition_id)

    # Divide into train and test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)

    # Image transforms
    pytorch_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    def apply_transforms(batch):
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch

    # Apply transforms
    partition_train_test = partition_train_test.with_transform(apply_transforms)

    trainloader = DataLoader(partition_train_test["train"], batch_size=BATCH_SIZE, shuffle=True)
    valloader = DataLoader(partition_train_test["test"], batch_size=BATCH_SIZE)
    testset = fds.load_split("test").with_transform(apply_transforms)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE)

    return trainloader, valloader, testloader

# Define the model
class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Train loop
def train(net, trainloader, epochs: int, verbose=False,dp=False, C = 1.0, SIGMA = 1.0):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())
    net.train()

    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        for batch in trainloader:
            images, labels = batch["img"].to(DEVICE), batch["label"].to(DEVICE)
            optimizer.zero_grad()
            outputs = net(images)
            if(not dp) :
                loss = criterion(outputs, labels)
                loss.backward() # just computes the loss           
                optimizer.step() #w = w - learning rate * loss function;
            else:
                # DP branch: per-sample losses, per-sample grads, clip, sum, add noise
                losses = F.cross_entropy(outputs, labels, reduction="none")  # (batch,)
                batch_size = images.size(0)

                # Prepare accumulators for gradients (same order as net.parameters())
                accum_grads = [torch.zeros_like(p, device=p.device) for p in net.parameters()]

                # Compute per-sample grads
                for i in range(batch_size):
                    # retain_graph True for all but last sample to free graph ASAP
                    retain = True if i < (batch_size - 1) else False
                    grads = torch.autograd.grad(losses[i], net.parameters(), retain_graph=retain, create_graph=False)

                    # compute L2 norm of grads (detach to avoid linking graphs)
                    total_norm_sq = torch.tensor(0.0, device=images.device)
                    for g in grads:
                        total_norm_sq = total_norm_sq + (g.detach().pow(2).sum())
                    total_norm = torch.sqrt(total_norm_sq)

                    # clip factor as a TENSOR on the correct device, clamped to <= 1.0
                    clip_factor = torch.clamp((C / (total_norm + 1e-6)), max=1.0).to(total_norm.device)  #Calculating clip factor here to use for clipping on HOW MUCH a single parameter can contribute to the loss function
                

                    # accumulate clipped grads (use detached grads)
                    for acc, g in zip(accum_grads, grads):
                        acc.add_(g.detach() * clip_factor)

                # Add Gaussian noise per-parameter and set p.grad
                for p, acc in zip(net.parameters(), accum_grads):
                    # noise std = sigma * C
                    noise = torch.randn_like(acc) * (SIGMA * C)
                    p.grad = (acc + noise) / float(batch_size)

                # optimizer step with the noisy averaged gradients
                optimizer.step()
                loss = losses.mean()
                # --- End DP-SGD block ---
            process = psutil.Process(os.getpid())
            memory_in_MB = process.memory_info().rss / 1024**2
            print(f"CPU memory used: {memory_in_MB:.2f} MB")
            print(f"CPU memory after batch: {process.memory_info().rss / 1024**2:.2f} MB")


              # scalar for logging
            epoch_loss += loss.item()
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()

        epoch_loss /= len(trainloader)
        epoch_acc = correct / total
        if verbose:
            print(f"Epoch {epoch+1}: Train Loss={epoch_loss:.4f}, Accuracy={epoch_acc:.4f}")



# Test loop
def test(net, testloader):
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()

    with torch.no_grad():
        for batch in testloader:
            images, labels = batch["img"].to(DEVICE), batch["label"].to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    loss /= len(testloader)
    accuracy = correct / total
    return loss, accuracy

# Federated Learning Utils
def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

# Visualize sample data
def visualize_sample(trainloader):
    batch = next(iter(trainloader))
    images, labels = batch["img"], batch["label"]

    # Reshape and convert images to a NumPy array
    images = images.permute(0, 2, 3, 1).numpy()
    images = images / 2 + 0.5  # Denormalize

    fig, axs = plt.subplots(4, 8, figsize=(12, 6))
    for i, ax in enumerate(axs.flat):
        ax.imshow(images[i])
        ax.set_title(trainloader.dataset.features["label"].int2str([labels[i]])[0])
        ax.axis("off")

    fig.tight_layout()
    plt.show()

# Run training
if __name__ == "__main__":
    trainloader, valloader, testloader = load_datasets(partition_id=0)
    net = Net().to(DEVICE)

    visualize_sample(trainloader)

    for epoch in range(5):
        train(net, trainloader, 1)
        loss, accuracy = test(net, valloader)
        print(f"Epoch {epoch+1}: Validation Loss={loss:.4f}, Accuracy={accuracy:.4f}")

    loss, accuracy = test(net, testloader)
    print(f"\nFinal Test Set Performance:\n\tLoss={loss:.4f}\n\tAccuracy={accuracy:.4f}")
