import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class Perceptron(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Perceptron, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = Perceptron(input_size=28*28, hidden_size=512, output_size=10)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")
    for i in range(num_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("CUDA is not available.")

def train(dataloader, model, criteria, optimizer, epochs):
    num_batches = len(dataloader)
    for epoch in range(epochs):
        model.train()
        train_loss, correct, total = 0, 0, 0
        print(f"\nEpoch {epoch+1}")

        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criteria(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)

        train_loss /= total
        accuracy = 100 * correct / total
        print(f"Train loss: {train_loss:.4f}, Train accuracy: {accuracy:.2f}%")
        evaluate(model, criteria, dataloader)

def evaluate(model, criteria, dataloader):
    num_batches = len(dataloader)
    test_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criteria(outputs, targets)

            test_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)

    test_loss /= total
    accuracy = 100 * correct / total
    print(f"Test loss: {test_loss:.4f}, Test accuracy: {accuracy:.2f}%\n-------------------------------")

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criteria = torch.nn.CrossEntropyLoss()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

train(train_loader, model, criteria, optimizer, epochs=5)