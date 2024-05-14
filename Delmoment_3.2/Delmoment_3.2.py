import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time

# Constants
batch_size = 32  # Adjusted batch size for better GPU utilization
epochs = 10

# Modify ResNet-18 model for FashionMNIST
class ModifiedResNet(nn.Module):
    def __init__(self, output_size):
        super(ModifiedResNet, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Use ResNet-18 for lighter computation
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # Modify the input layer to accept grayscale images
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Modify the output layer to match the number of classes in FashionMNIST
        self.model.fc = nn.Linear(self.model.fc.in_features, output_size)
        self.to(self.device)

    def forward(self, x):
        return self.model(x)

# Model instantiation
model = ModifiedResNet(output_size=10)

# Optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
criteria = torch.nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# Data transformations
train_transform = transforms.Compose([
    transforms.Resize(32),  # Resize to 32x32
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

test_transform = transforms.Compose([
    transforms.Resize(32),  # Resize to 32x32
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# FashionMNIST datasets
train_set = datasets.FashionMNIST(root='./data', train=True, download=True, transform=train_transform)
test_set = datasets.FashionMNIST(root='./data', train=False, download=True, transform=test_transform)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)

def plot_progress(train_losses, test_losses, train_accuracies, test_accuracies):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title('Training and Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.title('Training and Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.show()

def evaluate(model, dataloader, criteria):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for input, target in dataloader:
            input, target = input.to(model.device), target.to(model.device)
            output = model(input)
            loss = criteria(output, target)
            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

def train(train_loader, model, criteria, optimizer, epochs, scheduler):
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []

    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for input, target in train_loader:
            input, target = input.to(model.device), target.to(model.device)
            optimizer.zero_grad()
            output = model(input)
            loss = criteria(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

        scheduler.step()

        train_loss = total_loss / len(train_loader)
        train_accuracy = 100 * correct / total

        test_loss, test_accuracy = evaluate(model, test_loader, criteria)

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)

        end_time = time.time()
        duration = end_time - start_time
        print(f"\nEpoch {epoch+1}/{epochs}\nTrain Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%\nTest Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.2f}%\nDuration: {duration:.2f} seconds")

    plot_progress(train_losses, test_losses, train_accuracies, test_accuracies)

# Start the training process
train(train_loader, model, criteria, optimizer, epochs, scheduler)