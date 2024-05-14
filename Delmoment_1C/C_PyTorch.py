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

def train(dataloader, model, criteria, optimizer, epochs):
    for epoch in range(epochs):
        model.train()
        train_loss, correct, total = 0, 0, 0
        num_batches = len(dataloader)

        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criteria(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)

        average_train_loss = train_loss / num_batches
        accuracy = 100 * correct / total
        print(f"\nEpoch {epoch + 1}\nTrain Loss: {average_train_loss:.4f}, Train Accuracy: {accuracy:.2f}%")
        evaluate(model, criteria, dataloader)

def evaluate(model, criteria, dataloader):
    model.eval()
    test_loss, correct, total = 0, 0, 0
    num_batches = len(dataloader)

    with torch.no_grad():
        for inputs, targets in dataloader:
            outputs = model(inputs)
            loss = criteria(outputs, targets)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)

        average_test_loss = test_loss / num_batches
        accuracy = 100 * correct / total
        print(f"Test Loss: {average_test_loss:.4f}, Test Accuracy: {accuracy:.2f}%\n---------------------------------------")

optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
criteria = torch.nn.CrossEntropyLoss()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # Adjusted normalization
])

train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

train(train_loader, model, criteria, optimizer, epochs=5)
