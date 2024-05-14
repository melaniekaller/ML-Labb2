import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time

# Constants
batch_size = 32
epochs = 10

# Define the improved Perceptron model
class Perceptron(nn.Module):
    def __init__(self, input_size, output_size):
        super(Perceptron, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.layers = nn.Sequential(
            nn.Conv2d(input_size, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.4),
            
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, output_size)
        )
        self.to(self.device)

    def forward(self, x):
        return self.layers(x)

# Model instantiation
model = Perceptron(input_size=1, output_size=10)

# Optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
criteria = torch.nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Enhanced data transformations
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(28, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# FashionMNIST datasets
train_set = datasets.FashionMNIST(root='./data', train=True, download=True, transform=train_transform)
test_set = datasets.FashionMNIST(root='./data', train=False, download=True, transform=test_transform)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

def plot_progress(train_losses, test_losses, train_accuracies, test_accuracies):
    plt.figure(figsize=(12, 6))

    for i, (metrics, title, ylabel) in enumerate([
        ((train_losses, test_losses), 'Training and Test Loss', 'Loss'),
        ((train_accuracies, test_accuracies), 'Training and Test Accuracy', 'Accuracy (%)')
    ]):
        plt.subplot(1, 2, i+1)
        plt.plot(metrics[0], label='Train')
        plt.plot(metrics[1], label='Test')
        plt.title(title)
        plt.xlabel('Epochs')
        plt.ylabel(ylabel)
        plt.legend()
        plt.ylim(0, 100 if ylabel == 'Accuracy (%)' else None)

    plt.tight_layout()
    plt.show()

# Function to evaluate the model
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

# Function to train the model
def train(train_loader, model, criteria, optimizer, epochs, scheduler):
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []

    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        print(f"\nEpoch {epoch+1}")

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

        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, '
              f'\nTest Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.2f}%')

        end_time = time.time()
        duration = end_time - start_time
        print(f"Completed in {duration:.2f} seconds")
    
    plot_progress(train_losses, test_losses, train_accuracies, test_accuracies)
        
# Train the model
train(train_loader, model, criteria, optimizer, epochs, scheduler)