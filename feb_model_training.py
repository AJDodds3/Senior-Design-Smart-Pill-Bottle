import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np


class EnhancedCNN(nn.Module):
    def __init__(self):
        super(EnhancedCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)  
        self.fc1 = nn.Linear(128 * 8 * 8, 128)  
        self.fc2 = nn.Linear(128, 2) 

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * 8 * 8)  
        x = self.dropout(torch.relu(self.fc1(x)))  
        x = self.fc2(x)
        return x

# Transformations with Data Augmentation
train_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomRotation(15),           # Random rotation
    transforms.ColorJitter(brightness=0.2),  # Brightness adjustment
    transforms.RandomHorizontalFlip(),       # Horizontal flip
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


train_data = datasets.ImageFolder(root="Datasets/Train", transform=train_transform)
test_data = datasets.ImageFolder(root="Datasets/Test", transform=test_transform)
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
test_loader = DataLoader(test_data, batch_size=16, shuffle=False)

# Initialize model, loss function, and optimizer
model = EnhancedCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Reduced learning rate


best_accuracy = 0
epochs_no_improve = 0
patience = 70  # Stop if no improvement after 70 epochs

epochs = 120  # Maximum number of epochs
for epoch in range(epochs):
    model.train()  #iterate through the training data over and over with the CNN learning algorithm
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    # Print training loss per epoch
    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}")
    
    # Evaluate on the test set after each epoch
    model.eval()  
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    
    # Check for early stopping
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        epochs_no_improve = 0
        torch.save(model.state_dict(), "best_model_cpu.pth", _use_new_zipfile_serialization=False)
        #torch.save(model.state_dict(), "best_model.pth") # Save the best model
    else:
        epochs_no_improve += 1

    if epochs_no_improve == patience:
        print("Early stopping triggered!")
        break

print(f"Training completed with best accuracy: {best_accuracy:.2f}%")
