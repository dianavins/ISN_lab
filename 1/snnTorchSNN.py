# https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_5.html

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

dtype = torch.float64
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
torch.set_default_dtype(torch.float64)

# Define a transform
class Binarize:
    def __init__(self, threshold=0.5):
        self.threshold = threshold
    
    def __call__(self, x):
        return (x > self.threshold).to(dtype=dtype)
    
transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            Binarize(threshold=0.5),
            transforms.Normalize((0,), (1,))])

mnist_train = datasets.MNIST('./tmp_mnist', train=True, download=True, transform=transform)
mnist_test = datasets.MNIST('./tmp_mnist', train=False, download=True, transform=transform)

# Split training data into train and validation sets
train_size = 50000
val_size = 10000
train_dataset, val_dataset = random_split(mnist_train, [train_size, val_size])

# dataloader arguments
batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=False)


# Temporal Dynamics
num_steps = 25
beta = 0.95

# Define Network
class Net(nn.Module):
    def __init__(self):
        super().__init__()

        # Convert all parameters to float64
        self.to(dtype=dtype)
        
        # Initialize layers
        self.fc1 = nn.Linear(784, 2048)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(2048, 1024)
        self.lif2 = snn.Leaky(beta=beta)
        self.fc3 = nn.Linear(1024, 10)
        self.lif3 = snn.Leaky(beta=beta)
        
        # Xavier initialization
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        
        

    def forward(self, x):

        x = x.to(dtype=dtype)


        # Initialize hidden states at t=0
        mem1 = self.lif1.init_leaky().to(dtype=dtype)
        mem2 = self.lif2.init_leaky().to(dtype=dtype)
        mem3 = self.lif3.init_leaky().to(dtype=dtype)

        # Record the final layer
        spk3_rec = []
        mem3_rec = []

        for step in range(num_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            spk1 = spk1.to(dtype=dtype)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2 = spk2.to(dtype=dtype)
            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif1(cur3, mem3)
            spk3 = spk3.to(dtype=dtype)
            spk3_rec.append(spk3)
            mem3_rec.append(mem3)

        return torch.stack(spk3_rec, dim=0), torch.stack(mem3_rec, dim=0)

# Load the network onto CUDA if available
net = Net().to(device)

# loss and optimizer
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=5e-4, betas=(0.9, 0.999))
for param_group in optimizer.param_groups:
    param_group['params'] = [p.to(dtype=dtype) for p in param_group['params']]

# pass data into the network, sum the spikes over time
# and compare the neuron with the highest number of spikes
# with the target

def print_batch_accuracy(data, targets, train=False):
    output, _ = net(data.view(batch_size, -1))
    _, idx = output.sum(dim=0).max(1)
    acc = np.mean((targets == idx).detach().cpu().numpy())

    if train:
        print(f"Train set accuracy for a single minibatch: {acc*100:.2f}%")
    else:
        print(f"Test set accuracy for a single minibatch: {acc*100:.2f}%")

def train_printer():
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
        spk_rec, mem_rec = net(data)

        # initialize the loss & sum over time
        loss_val = torch.zeros((1), device=device)
        for step in range(num_steps):
            loss_val += loss(mem_rec[step], targets)

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
            test_spk, test_mem = net(test_data.view(batch_size, -1))

            # Test set loss
            test_loss = torch.zeros((1), device=device)
            for step in range(num_steps):
                test_loss += loss(test_mem[step], test_targets)
            test_loss_hist.append(test_loss.item())

            # Print train/test loss/accuracy
            if counter % 50 == 0:
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
test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True, drop_last=False)

with torch.no_grad():
  net.eval()
  for data, targets in test_loader:
    data = data.to(device)
    targets = targets.to(device)

    # forward pass
    test_spk, _ = net(data.view(data.size(0), -1))

    # calculate total accuracy
    _, predicted = test_spk.sum(dim=0).max(1)
    total += targets.size(0)
    correct += (predicted == targets).sum().item()
    
# final test accuracy and loss
test_acc = 100 * correct / total
loss = loss_hist[-1]
print(f"Final Test Accuracy: {test_acc:.2f}%")
print(f"Final Test Loss: {loss:.2f}")