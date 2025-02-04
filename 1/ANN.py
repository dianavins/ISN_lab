import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data Preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: (x > 0.5).float().to(torch.float64))  # Binarize and convert to double
])

# Load datasets
dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_dataset, val_dataset = random_split(dataset, [50000, 10000])
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Data loaders
batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Model Definition
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 2048) # input layer: 784, hidden layer: 2048
        self.sigmoid1 = nn.Sigmoid() # activation function
        self.fc2 = nn.Linear(2048, 1024) # hidden layer: 2048, output layer: 1024
        self.sigmoid2 = nn.Sigmoid() # activation function
        self.fc3 = nn.Linear(1024, 10) # output layer: 1024, output classes: 10

    def forward(self, x): 
        x = x.view(-1, 784) # flatten the input tensor
        x = self.sigmoid1(self.fc1(x)) # pass through the first layer and activation function
        x = self.sigmoid2(self.fc2(x)) # pass through the second layer and activation function
        x = self.fc3(x) # pass through the output layer
        return x 

model = Net().double().to(device) # double precision

# Xavier Initialization with Sigmoid gain
def init_weights(m): 
    if isinstance(m, nn.Linear): 
        gain = nn.init.calculate_gain('sigmoid') if m is not model.fc3 else 1.0 
        nn.init.xavier_normal_(m.weight, gain=gain)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.01)

model.apply(init_weights)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
epochs = 10
for epoch in range(epochs):
    model.train()
    train_loss, train_correct, train_total = 0.0, 0, 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    train_loss /= train_total
    train_acc = train_correct / train_total

    # Validation
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_loss /= val_total
    val_acc = val_correct / val_total

    print(f'Epoch {epoch+1}/{epochs}:')
    print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}')
    print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}\n')

# Test Evaluation
model.eval()
test_correct, test_total = 0, 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

test_acc = test_correct / test_total
print(f'Test Accuracy: {test_acc:.4f}')