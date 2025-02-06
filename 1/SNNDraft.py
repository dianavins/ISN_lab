import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import snntorch as snn
from torchvision import datasets, transforms
import numpy as np
from tqdm import tqdm

# Set random seed for reproducibility
torch.manual_seed(42)

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set default tensor type to float64
torch.set_default_dtype(torch.float64)

# Define binarization transform: converts continuous values to binary values based on a threshold.
class Binarize(object):
    def __init__(self, threshold=0.5):
        self.threshold = threshold
    
    def __call__(self, x):
        return (x > self.threshold).to(dtype=torch.float64)  # Convert to double

# Data loading and preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    Binarize(threshold=0.5)
])

# Load MNIST dataset
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Split training data into train and validation sets
train_size = 50000
val_size = 10000
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

# Create data loaders
batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define the SNN model
class SpikingNN(nn.Module):
    def __init__(self, beta=0.95):
        super().__init__()
        
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
        
        # Convert all parameters to float64
        self.double()
        
    def forward(self, x):
        # Ensure input is double precision
        x = x.double()
        
        # Initialize hidden states with double precision
        mem1 = self.lif1.init_leaky().double()
        mem2 = self.lif2.init_leaky().double()
        mem3 = self.lif3.init_leaky().double()
        
        # Forward pass
        cur1 = self.fc1(x)
        spk1, mem1 = self.lif1(cur1, mem1)
        spk1 = spk1.double()  # Ensure double precision
        
        cur2 = self.fc2(spk1)
        spk2, mem2 = self.lif2(cur2, mem2)
        spk2 = spk2.double()  # Ensure double precision
        
        cur3 = self.fc3(spk2)
        spk3, mem3 = self.lif3(cur3, mem3)
        
        return torch.softmax(spk3, dim=1)

# Initialize model
model = SpikingNN().to(device)
model.to(torch.float64)

# Define loss function and optimizer
loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training function
def train_epoch(model, loader, optimizer, loss):
    """
    Train the model for one epoch.
    
    Args:
        model (nn.Module): Neural network model to train
        loader (DataLoader): DataLoader containing the training data
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters
        loss (callable): Loss function
        
    Returns:
        tuple: (average_loss, accuracy)
            - average_loss (float): Average loss per batch for the epoch
            - accuracy (float): Training accuracy as a percentage
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for inputs, targets in tqdm(loader, desc="Training"):
        inputs = inputs.view(-1, 784).to(device).double()
        targets = targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        batch_loss = loss(outputs, targets)
        
        batch_loss.backward()
        optimizer.step()
        
        total_loss += batch_loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    return total_loss / len(loader), 100. * correct / total

# Validation function
def validate(model, loader, loss):
    """
    Validate the model on a validation set.
    
    Args:
        model (nn.Module): Neural network model to validate
        loader (DataLoader): DataLoader containing the validation data
        loss (callable): Loss function
        
    Returns:
        tuple: (average_loss, accuracy)
            - average_loss (float): Average loss per batch for validation set
            - accuracy (float): Validation accuracy as a percentage
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc="Validating"):
            inputs = inputs.view(-1, 784).to(device).double()
            targets = targets.to(device)
            
            outputs = model(inputs)
            batch_loss = loss(outputs, targets)
            
            total_loss += batch_loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return total_loss / len(loader), 100. * correct / total

# Training loop
num_epochs = 10
best_val_acc = 0

for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, loss)
    val_loss, val_acc = validate(model, val_loader, loss)
    
    print(f'Epoch: {epoch+1}/{num_epochs}')
    print(f'Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.2f}%')
    print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%')
    print('-' * 60)
    
# Testing function
def test(model, loader, loss):
    """
    Evaluate the model on a test set.
    
    Args:
        model (nn.Module): Neural network model to test
        loader (DataLoader): DataLoader containing the test data
        loss (callable): Loss function
        
    Returns:
        tuple: (average_loss, accuracy)
            - average_loss (float): Average loss per batch for test set
            - accuracy (float): Test accuracy as a percentage
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc="Testing"):
            inputs = inputs.view(-1, 784).to(device).double()
            targets = targets.to(device)
            
            outputs = model(inputs)
            batch_loss = loss(outputs, targets)
            
            total_loss += batch_loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return total_loss / len(loader), 100. * correct / total

# Testing the model after training
model.load_state_dict(torch.load('best_snn_model.pth'))
test_loss, test_acc = test(model, test_loader, loss)

print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%')