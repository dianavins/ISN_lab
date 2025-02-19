# imports
import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import numpy as np
import itertools

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
torch.set_default_dtype(torch.float64)

# Binarze transform converts continuous values to binary values based on a threshold.
class Binarize:
    def __init__(self, threshold=0.5):
        self.threshold = threshold
    
    def __call__(self, x):
        return (x > self.threshold).to(dtype=torch.float64)
    
transform = transforms.Compose([
            transforms.ToTensor(),
            Binarize(threshold=0.5),
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

# Split training data into train and validation sets
train_size = 50000
val_size = 10000
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

# dataloader arguments
batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define SpikingNNwork
class ANN(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Initialize layers
        self.fc1 = nn.Linear(784, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 10)
        
        # Xavier initialization
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        
    def forward(self, x):
        # Apply sigmoid activation after each layer
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = self.fc3(x)  # No activation on the final layer (raw logits)
        return x

# Load the network onto CUDA if available
net = ANN().to(device)
print("Network initialized")

# loss and optimizer
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=5e-4, betas=(0.9, 0.999))

# pass data into the network, sum the spikes over time
# and compare the neuron with the highest number of spikes
# with the target

def print_batch_accuracy(data, targets, train=False):
    """
    Calculate and print the accuracy for a single minibatch.
    
    Args:
        data (Tensor): Input data batch
        targets (Tensor): Target labels
        train (bool): Flag indicating if this is training data (default: False)
    
    Returns:
        None
    """
    output = net(data.view(batch_size, -1))
    _, idx = output.max(1)
    acc = np.mean((targets == idx).detach().cpu().numpy())

    if train:
        print(f"Train set accuracy for a single minibatch: {acc*100:.2f}%")
    else:
        print(f"Test set accuracy for a single minibatch: {acc*100:.2f}%")

def train_printer():
    """
    Print training progress information including loss and accuracy metrics.
    
    Prints current epoch, iteration, train/test loss, and batch accuracy for both
    training and test sets.
    
    Returns:
        None
    """
    print(f"Epoch {epoch}, Iteration {iter_counter}")
    print(f"Train Set Loss: {loss_hist[counter]:.2f}")
    print(f"Test Set Loss: {test_loss_hist[counter]:.2f}")
    print_batch_accuracy(data, targets, train=True)
    print_batch_accuracy(test_data, test_targets, train=False)
    print("\n")
    

# Training Loop
num_epochs = 10
loss_hist = []
test_loss_hist = []
counter = 0
num_steps = 25

# Outer training loop
for epoch in range(num_epochs):
    iter_counter = 0
    train_batch = iter(train_loader)

    # Minibatch training loop
    for data, targets in train_batch:
        data = data.view(-1, 784).to(device)
        targets = targets.to(device)

        # forward pass
        net.train()
        output = net(data)

        # calculate loss
        loss_val = loss(output, targets)

        # Gradient calculation + weight update
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        # Store loss history for future plotting
        loss_hist.append(loss_val.item())

        # Test set
        with torch.no_grad():
            net.eval()
            test_data, test_targets = next(iter(test_loader))
            test_data = test_data.to(device)
            test_targets = test_targets.to(device)

            # Test set forward pass
            test_output = net(test_data.view(batch_size, -1))

            # Test set loss
            test_loss = loss(test_output, test_targets)
            test_loss_hist.append(test_loss.item())

            # Print train/test loss/accuracy
            if counter % 100 == 0:
                train_printer()
            counter += 1
            iter_counter +=1
            
# Plot Loss
fig = plt.figure(facecolor="w", figsize=(10, 5))
plt.plot(loss_hist)
plt.plot(test_loss_hist)
plt.title("Loss Curves")
plt.legend(["Train Loss", "Test Loss"])
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.show()


# Test Evaluation
total = 0
correct = 0

# drop_last switched to False to keep all samples
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

with torch.no_grad():
    net.eval()
    for data, targets in test_loader:
        data = data.to(device)
        targets = targets.to(device)

        # forward pass
        output = net(data.view(data.size(0), -1))  # Only one output, no need to unpack

        # calculate total accuracy
        _, predicted = output.max(1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
    
# final test accuracy and loss
test_acc = 100 * correct / total
loss = loss_hist[-1]
print(f"Final Test Accuracy: {test_acc:.2f}%")
print(f"Final Test Loss: {loss:.2f}")