# from google.colab import drive
# drive.mount('/content/drive')

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import itertools

train_transform = transforms.Compose([
    transforms.Resize((28, 28)),  # Resizes to 28x28
    transforms.ToTensor(),  # Converts to tensor
])

test_transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])



batch_size = 128
#Load MNIST dataset and apply transformations
full_train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=train_transform) #60k images
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=test_transform) #10k images



#split the training dataset into training and validation datasets
train_size = int(0.83 * len(full_train_dataset)) #50k for training
val_size = len(full_train_dataset) - train_size #10k for validation
train_dataset, _ = torch.utils.data.random_split(full_train_dataset, [train_size, val_size])

#create a separate validation dataset with the test transforms
val_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=test_transform)

#subset only the remaining 10% of the data for validation
_, val_dataset = torch.utils.data.random_split(val_dataset, [train_size, val_size])


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

labels_map = {
    0: "0",
    1: "1",
    2: "2",
    3: "3",
    4: "4",
    5: "5",
    6: "6",
    7: "7",
    8: "8",
    9: "9",
}

#Display a batch of images
def show_images(train_dataset, cols, rows):
  figure = plt.figure(figsize=(8, 8))

  for i in range(1, cols * rows + 1):
      sample_idx = torch.randint(len(train_dataset), size=(1,)).item()
      img, label = train_dataset[sample_idx]
      figure.add_subplot(rows, cols, i)
      plt.title(labels_map[label])
      plt.axis("off")
      plt.imshow(img.squeeze(), cmap="gray")

  plt.show()

show_images(train_dataset, 3, 3)

x = torch.rand(5, 10)
print(x)
x.max(dim=1, keepdim=True)[0]

x = torch.rand(5, 4, 4, 256)
print(x.size())
inputs = torch.cat([x[:, 0, layer1col, 0:64] for layer1col in range(4)], dim=1)
print(inputs.size())

#PIC Conv Layer
class PICConvLayer(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
    super(PICConvLayer, self).__init__()

    #initialize weights with a normal distribution
    self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False) # define convolutional kernels

    self.pool = nn.AvgPool2d(kernel_size=2, stride=2) #sum pooling(sum of all kernels = avg * 4)

    self.bias = nn.Parameter(torch.zeros(1, 6, 1, 1)) #a bias term for each feature map(6 feature maps total)

    self.sigmoid = nn.Sigmoid() # apply sigmoid acvtivation


  def forward(self, x):
    x = self.conv(x) # convolutional layer
    x = self.pool(x) # pooling layer
    x = x + self.bias # add bias
    x = self.sigmoid(x) # apply sigmoid activation
    return x




#EIC Layer 1 Core
class Core(nn.Module):
  '''
  Represents a core in the EIC Layer 1
  '''
  def __init__(self, in_features, out_features):
    super(Core, self).__init__()
    self.fc = nn.Linear(in_features, out_features, bias=False) # fully connected layer
    self.bias = 0.0 # bias term
    self.sigmoid = nn.Sigmoid() # apply sigmoid activation

  def forward(self, x):
    x = self.fc(x) # pass through fully connected layer
    self.bias = -0.5 * x.max(dim=1, keepdim=True)[0] # bias term = max of every batch (each * -0.5)
    x = x + self.bias # add bias
    x = self.sigmoid(x) # apply sigmoid activation
    return x



#EIC Layer 1 with 15 cores(4x4 grid minus one core at the lower-right corner)
class EICLayer1(nn.Module):
  '''
  Layer 1 has a 4 rows of 4 cores (3 cores in 4th layer). Each core takes 256 inputs and produces 256 outputs.
  Layer 1 takes 1184 inputs (8 0's appended to PIC activations) which are split into 256 chunks for each core,
  repeated ~3.2 times to span all 15 cores.
  '''
  def __init__(self):
    super(EICLayer1, self).__init__()

    # initialize the 15 cores
    self.cores = nn.ModuleList([Core(in_features=256, out_features=256) for i in range(15)])

  def forward(self, x):
    #split the 1184 inputs into chunks of 256 for each core
    #store the activations in a 4x4 grid
    batch_size = x.size(0)
    activations_grid = torch.empty(batch_size, 4, 4, 256) #for every batch, 4x4 grid of cores, each with 256 activations
    input_idx = 0
    core_idx = 0

    # loop over the cores, 256 inputs at a time
    for row in range(4):
      for col in range(4):
        if row != 3 and col != 3: # core (3,3) is empty # shouldn't it be and instead of or?
          if 1184 - input_idx >= 256: # not the edge case (end of a set of PIC activations)
            chunk = x[:, input_idx:input_idx+256] # define the input chunk
            input_idx = input_idx + 256 # increment the input index

          else: # need to loop back around to the beginning of the PIC activations
            remainder = 256 - (1184 - input_idx) # get the remainder
            chunk = torch.cat((x[:, input_idx:], x[:, :remainder]), dim=1) # concatenate the two parts
            input_idx = remainder # set the input index to the remainder

          activations_grid[:, row, col, :] = self.cores[core_idx](chunk) # add the activation to the grid (activations computed for entire batch size at once)
          core_idx = core_idx + 1 # increment the core index. Row 1 = cores 0-3, Row 2 = cores 4-7, Row 3 = cores 8-11, Row 4 = cores 12-14

    return activations_grid # return the 4x4 grid of activations

#EIC Layer 2 with 16 cores(on chip, these cores occupy rows 0 through 3, cols 4 through 7)
class EICLayer2(nn.Module):
  '''
  Layer 2 has a 4 rows of 4 cores. Each core takes 256 inputs and produces 256 outputs.
  The first core of Layer 2 takes the first 64 activations of each of the 4 cores in the corresponding row in Layer 1.
  The second core of layer 2 takes the second 64 activations of each of the 4 cores in the corresponding row in Layer 1, and so forth.
  '''
  def __init__(self):
    super(EICLayer2, self).__init__()

    # create the 16 cores, 12 with 256 inputs and 4 with 192 inputs (due to absence of a (3,3) core in Layer 1)
    self.cores = nn.ModuleList(
        [Core(in_features=256, out_features=256) for i in range(12)] +
        [Core(in_features=192, out_features=256) for i in range(4)]
    )

  def forward(self, x):
    # input from EICLayer1 of size [batch_size, 4, 4, 256] (4x4 grid of activations from 15 cores, location(3,3) empty)
    batch_size = x.size(0)
    activations_grid = torch.empty(batch_size, 4, 4, 256) # initialize output tensor for every batch, 4x4 grid of cores, each with 256 activations
    input_idx = 0
    core_idx = 0


    # loop over the cores, 256(or 192) inputs at a time
    # first three rows take 4 sets of 64 inputs(256 total) of the 4 Layer1 cores on its row
    # last row takes 3 sets of 64 inputs(192 total) of the 3 Layer1 cores on its row
    for row in range(4):
      start_idx = 0
      
      # defining number of cores per row depending on the row
      num_layer1cols = 4
      if row == 3:
        num_layer1cols = 3

      # loop over the cores in the row, specifying 256 inputs at a time 
      for layer2col in range(4):
        end_idx = start_idx + 64 # end index of the chunk
        chunk = torch.cat([x[:, row, layer1col, start_idx:end_idx] for layer1col in range(num_layer1cols)], dim=1) # concatenate the 64 inputs from each of the 4 Layer1 cores on the row

        activations_grid[:, row, layer2col, :] = self.cores[core_idx](chunk) # add the activation to the grid(activations computed for entire batch size at once)
        core_idx = core_idx + 1
        start_idx = start_idx + 64

    return activations_grid

#EIC Layer 3 with 16 cores(on chip, these cores occupy rows 4 through 7, cols 4 through 7)
class EICLayer3(nn.Module):
  '''
  Layer 3 does the same as Layer 2, but taking 64 activations each of 4 cores columnwise rather than rowwise.
  A core in Layer 3 takes the first 64 activations of each of the 4 Layer2 cores on its column.
  Each core in Layer 3 has 64 outputs.
  '''
  def __init__(self):
    super(EICLayer3, self).__init__()

    # create the 16 cores
    self.cores = nn.ModuleList([Core(in_features=256, out_features=64) for i in range(16)])

  def forward(self, x):
    # input from EICLayer2 of size [batch_size, 4, 4, 256] (4x4 grid of activations from 16 cores)
    batch_size = x.size(0)
    activations_grid = torch.empty(batch_size, 4, 4, 64) # initialize output tensor for every batch, 4x4 grid of cores, each with 64 activations
    input_idx = 0
    core_idx = 0

    # each core recieves 64 outputs of the 4 Layer2 cores on its column
    # loop over the cores, 256 inputs at a time
    for col in range(4):
      start_idx = 0

      for layer3row in range(4):
        end_idx = start_idx + 64 # define boundaries of the chunk
        chunk = torch.cat([x[:, layer2row, col, start_idx:end_idx] for layer2row in range(4)], dim=1) # concatenate the 64 inputs from each of the 4 Layer2 cores on the column

        activations_grid[:, layer3row, col, :] = self.cores[core_idx](chunk) # add the activation to the grid(activations computed for entire batch size at once)
        core_idx = core_idx + 1 # increment the core index
        start_idx = start_idx + 64 # move to the next set of 64 inputs

    return activations_grid # return the 4x4 grid of activations

# EIC Layer 4 with 4 cores(on chip, these cores occupy rows 4 through 7, col 3)
class EICLayer4(nn.Module):
  '''
  Layer 4 has one column (column 3) of 4 cores. Each core takes 64 inputs and produces 64 outputs.
  Each core in Layer 4 takes the 64 activations of the 4 Layer3 cores on its row (256 total inputs).
  Each core in Layer 4 has 64 outputs.
  '''
  def __init__(self):
    super(EICLayer4, self).__init__()

    # create the 4 cores
    self.cores = nn.ModuleList([Core(in_features=256, out_features=64) for i in range(4)])

  def forward(self, x):
    # input from EICLayer2 of size [batch_size, 4, 4, 64] (4x4 grid of activations from 16 cores)
    batch_size = x.size(0)
    activations_grid = torch.empty(batch_size, 4, 1, 64) # initialize output tensor for every batch, 4x1 grid of cores, each with 64 activations
    input_idx = 0
    core_idx = 0

    # each core recieves all 64 outputs of the 4 Layer3 cores on its row
    # loop over the cores, 256 inputs at a time
    for row in range(4):
      chunk = torch.cat([x[:, row, layer3col, :] for layer3col in range(4)], dim=1) # concatenate the 64 inputs from each of the 4 Layer3 cores on the row

      activations_grid[:, row, 0, :] = self.cores[core_idx](chunk) # add the activation to the grid(activations computed for entire batch size at once)
      core_idx = core_idx + 1 # increment the core index

    return activations_grid # return the 4x1 grid of activations


# overall architecture
class Part8_cd(nn.Module):
  '''
  PIC activations -> PIC Conv Layer -> EIC Layer 1 -> EIC Layer 2 -> EIC Layer 3 -> EIC Layer 4 -> flatten -> Output Layer -> Softmax
  '''
  def __init__(self):
    super(Part8_cd, self).__init__()
    self.piclayer1 = PICConvLayer(in_channels=1, out_channels=6, kernel_size=3, stride=1, padding=1)
    self.flatten = nn.Flatten()
    self.eiclayer1 = EICLayer1()
    self.eiclayer2 = EICLayer2()
    self.eiclayer3 = EICLayer3()
    self.eiclayer4 = EICLayer4()
    self.output = nn.Linear(256, 10) #the output layer on the chip is is the core at location (3,3)
    self.softmax = nn.Softmax(dim=1)

  def forward(self, x):
    x = self.piclayer1(x)
    x = self.flatten(x)
    #pad the inputs to 1184 activations(divisble by 16)
    batch_size = x.size(0)
    x = torch.cat((x, torch.zeros(batch_size, 8)), dim=1)
    x = self.eiclayer1(x)
    x = self.eiclayer2(x)
    x = self.eiclayer3(x)
    x = self.eiclayer4(x)
    x = self.flatten(x)
    x = self.output(x)
    x = self.softmax(x)
    return x

model = Part8_cd()
print(model)
# in_channels: MNIST is grayscale, so each pixel has one value
# out_channels: equal to the number of kernels, which is 6
# kernel_size: size of each kernel
# stride: how much a kernel shifts across the image each step
# padding: number of layers of 0s added around the image. To preserve the image size(zero padding), this number is equal to (kernel size(3) - 1)/2 = 1

# test output shape
for images, labels in train_loader:
  output = model(images)
  print(f"Output shape: {output.shape}")
  break  #run only one batch

# looks good, 64 images(1 batch) and there are 10 classes

# using Adam optimizer with a learning rate of 0.001
def create_adam_optimizer(model, learning_rate=0.001):
  return torch.optim.Adam(model.parameters(), lr=learning_rate)

# Attempting to use Projected Gradient Descent to solve negative weights problem(rather than clamping during forward method)
def train_model_with_pgd_qat(model, train_loader, val_loader, optimizer, scheduler, epochs=5, batch_size=64, loss_fn=nn.CrossEntropyLoss()):
  
  # Setup device
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model.to(device)
  print(device)
  best_val_accuracy = 0

  early_stop_threshold = 10 # stop training if validation loss does not improve for 10 epochs
  best_val_loss = 150.0 # initialize with a high value
  epochs_without_improvement = 0 # counter for epochs without improvement

  #lists to store loss and accuracy values per epoch
  train_losses = []
  val_losses = []
  train_accuracies = []
  val_accuracies = []

  for epoch in range(epochs):
    
    # training phase
    model.train() # Set model to training mode
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for i, data in enumerate(train_loader, 0):

      # get the inputs; data is a list of [inputs, labels]
      inputs, labels = data

      inputs = inputs.to(device)
      labels = labels.to(device)

      # zero the parameter gradients
      optimizer.zero_grad()

      # update output, loss function optimzer
      outputs = model(inputs)
      loss = loss_fn(outputs, labels)
      loss.backward()
      optimizer.step()

      # apply Projected Gradient Descent (PGD) to ensure nonnegative weights before the optimizer steps.
      #  with torch.no_grad():  #disable gradient tracking
      #   for param in model.parameters():
      #     #project all weights to nonnegative values
      #     param.data = torch.clamp(param.data, min=0.00001)


      #     # #quantize all weights
      #     # #exclude conv and pooling layers from 8-bit quantization
      #     # if not isinstance(param, nn.Conv2d) and not isinstance(param, nn.MaxPool2d):
      #     #   #param.data = weight_quantizer(param.data)
      #     #   param.data = torch.round(param.data * 256) / 256


      # any_negative = False
      # #check if weights are nonnegative
      # for name, param in model.named_parameters():
      #     if torch.any(param.data < 0):
      #         any_negative = True
      # if any_negative:
      #   print("Negative weight values found")


      # get model's prediction
      _, predicted = torch.max(outputs, 1)

      # print statistics
      running_loss += loss.item()
      if i % 2000 == 1999:    # print every 2000 mini-batches
        print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
        running_loss = 0.0

      correct_train += (predicted == labels).sum().item() # sum up correct predictions for each batch
      total_train += labels.size(0) # sum up total number of samples

    # calculate training loss and accuracy
    train_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct_train / total_train

    # append the values to the lists
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)


    # validation phase
    model.eval() # Set model to evaluation mode
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad(): # disable gradient tracking for computational efficiency
      for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device) # move to device for  computational efficiency

        # forward pass
        outputs = model(inputs) # a singleforward pass
        loss = loss_fn(outputs, labels)

        # statistics
        val_loss += loss.item() # sum up batch loss
        _, predicted = torch.max(outputs, 1) # get the predicted class
        val_correct += (predicted == labels).sum().item() # sum up correct predictions
        val_total += labels.size(0) # sum up total number of samples

    val_loss /= len(val_loader) # average loss
    val_accuracy = 100 * val_correct / val_total # average accuracy

    val_losses.append(val_loss) 
    val_accuracies.append(val_accuracy) 

    # check for early stopping based on validation loss
    if val_losses[-1] < best_val_loss: # if there is an improvement in loss
      best_val_loss = val_losses[-1] # update the best validation loss
      epochs_without_improvement = 0 # reset the counter
    else:
      epochs_without_improvement += 1 

    # If the validation loss starts increasing or is plateauing, stop training
    if epochs_without_improvement > early_stop_threshold:
      print(f"Early stopping at epoch {epoch}")
      break



    # Print statistics after each epoch
    print(f"Epoch {epoch+1}/{epochs}, Learning Rate: {optimizer.param_groups[0]['lr']},")
    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%,")
    print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

    if (val_accuracy >= best_val_accuracy): # if there is an improvement in accuracy
      # save model
      torch.save(model.state_dict(), './pic_qat_step5_9.pth')
      best_val_accuracy = val_accuracy

    # step the scheduler at the end of each epoch (PART 5), for dynamic learning rates
    # scheduler.step()
    # scheduler.step(val_loss)


  print("Training complete")



  #plot losses and accuracies
  plt.figure(figsize=(10, 5))

  plt.subplot(1, 2, 1)
  plt.plot(range(len(train_losses)), train_losses, label='Training Loss') # plot training loss
  plt.plot(range(len(val_losses)), val_losses, label='Validation Loss') # plot validation loss
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.title('Training and Validation Loss')
  plt.legend()

  plt.subplot(1, 2, 2)
  plt.plot(range(len(train_accuracies)), train_accuracies, label='Training Accuracy') # plot training accuracy
  plt.plot(range(len(val_accuracies)), val_accuracies, label='Validation Accuracy') # plot validation accuracy
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.title('Training and Validation Accuracy')
  plt.legend()

  plt.tight_layout()
  plt.show()

  min_val_loss_epoch = val_losses.index(min(val_losses)) + 1 # get the epoch with the lowest validation loss
  print(f"Epoch with the lowest validation loss: {min_val_loss_epoch}")

def evaluate_model(model, test_loader):
    '''
    Evaluate the model for accuracy on the test set
    '''
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0

    for data in test_loader: # Loop through all test batches
        inputs, labels = data

        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)  # Get the predicted class
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    accuracy = 100 * correct / total
    return accuracy

def train_test_model_qat(model, learning_rate=0.01, num_epochs=10):
  '''
  train and test the model with quantization aware training
  '''
  loss_fn = nn.CrossEntropyLoss()
  print(learning_rate)
  optimizer = create_adam_optimizer(model, learning_rate)
  #optimzer = create_sgd_optimizer(model, learning_rate)

  #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 100, gamma = 1) #essentially no scheduler
  #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2, 5, 7], gamma=0.1)
  #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1) #learning rate annealing (STEP 5), standard step decreases the lr by a scale gamma every number of epochs
  #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0.0001) #learning rate annealing(STEP 5), cosine gradually decreases the learning rate more and more, good for sensitive weights
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, min_lr=0.00001) #learning rate annealing (STEP 5), automatically reduces lr when val loss plateaus

  train_model_with_pgd_qat(model, train_loader, val_loader, optimizer, scheduler, loss_fn=loss_fn, epochs=num_epochs) # train the model using PGD


  model.eval()
  #model_int8 = torch.ao.quantization.convert(model, inplace=True)
  #print(model_int8)
  #model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
  print("Testing accuracy: " + str(evaluate_model(model, test_loader)))
  return model

epochs = 100
learning_rate = 0.001
train_test_model_qat(model, learning_rate, num_epochs=epochs)

PATH = "/content/pic_qat_step5_9.pth"
#best_model = torch.load(PATH)
#best_model.eval()
#evaluate_model(best_model, test_loader)

best_model = Part8_cd()  # Make sure this matches the saved model's architecture

best_model.load_state_dict(torch.load(PATH, weights_only=False))

best_model.eval()

evaluate_model(best_model, test_loader)

torch.save()
