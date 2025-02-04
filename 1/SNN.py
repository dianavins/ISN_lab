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
print(f"Using device: {device}")

# Set default tensor type to float64
torch.set_default_tensor_type(torch.DoubleTensor)

# Define binarization transform
class Binarize(object):
    def __init__(self, threshold=0.5):
        self.threshold = threshold
    
    def __call__(self, x):
        return (x > self.threshold).double()  # Convert to double

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
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size, beta=0.95):
        super().__init__()
        
        # Convert all parameters to float64 (64-bit precision)
        self.fc1 = nn.Linear(input_size, hidden1_size)
        self.lif1 = snn.Leaky(beta=beta)
        
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.lif2 = snn.Leaky(beta=beta)
        
        self.fc3 = nn.Linear(hidden2_size, output_size)
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
        
        return torch.softmax(mem3, dim=1)

# Initialize model
print("Initializing model...")
model = SpikingNN(
    input_size=784,  # 28x28 pixels
    hidden1_size=2048,
    hidden2_size=1024,
    output_size=10
).to(device)

# Ensure model is in double precision
model = model.double()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training function
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for inputs, targets in tqdm(loader, desc="Training"):
        inputs = inputs.view(-1, 784).to(device).double()
        targets = targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    return total_loss / len(loader), 100. * correct / total

# Validation function
def validate(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc="Validating"):
            inputs = inputs.view(-1, 784).to(device).double()
            targets = targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return total_loss / len(loader), 100. * correct / total

# Training loop
print("Starting training...")
num_epochs = 10
best_val_acc = 0

for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
    val_loss, val_acc = validate(model, val_loader, criterion)
    
    print(f'Epoch: {epoch+1}/{num_epochs}')
    print(f'Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.2f}%')
    print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%')
    print('-' * 60)
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_snn_model.pth')

print(f'Best Validation Accuracy: {best_val_acc:.2f}%')