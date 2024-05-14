import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import datetime, time
import os
import json
import matplotlib.pyplot as plt

writer = SummaryWriter()

with open('config.json') as config_file:
    config = json.load(config_file)

input_size = config['input_size']
hidden_size = config['hidden_size']
output_size = config['output_size']
batch_size = config['batch_size']
learning_rate = config['learning_rate']
epochs = config['epochs']
model_path = config['model_path']

class Perceptron(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Perceptron, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.conv1 = nn.Conv2d(input_size, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.25)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(0.25)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 7 * 7, hidden_size)
        self.dropout3 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_size, output_size)

        self.to(self.device)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout3(x)
        x = self.fc2(x)
        return x    

model = Perceptron(input_size=1, hidden_size=512, output_size=10)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
criteria = torch.nn.CrossEntropyLoss()

transform = transforms.Compose([ 
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))
])

train_set = datasets.MNIST(root='./data', train=True,  download=True, transform=transform)
test_set  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False)

train_transform = transforms.Compose([
    transforms.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.8, 1.2)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(saturation=0.5),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def save_checkpoint(state, filename="checkpoint.pth.tar", directory="checkpoints"):
    """Spara checkpoint till disk."""
    filepath = os.path.join(directory, filename)
    torch.save(state, filepath)
    print(f"Checkpoint saved to {filepath}\n-------------------------------")

def get_checkpoint_filename(epoch_number):
    now = datetime.datetime.now()  # Nuvarande datum och tid
    date_time = now.strftime("%Y%m%d_%H%M%S")  # Formatera som sträng
    filename = f"checkpoint_epoch_{epoch_number}_{date_time}.pth.tar"
    return filename

def load_checkpoint(model, optimizer, filename='checkpoint.pth.tar'):
    """Ladda checkpoint från disk."""
    if os.path.isfile(filename):
        print(f"Loading checkpoint '{filename}'")
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"Loaded checkpoint '{filename}' (epoch {checkpoint['epoch']})")
    else:
        print(f"No checkpoint found at '{filename}'")

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

def plot_examples(loader, model, correct=True, device='cpu'):
    model.eval()
    count = 0
    axes = plt.subplots(2, 3, figsize=(10, 5))
    axes = axes.flatten()

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            for img, label, pred in zip(images, labels, predicted):
                if (pred == label) is correct:
                    ax = axes[count]
                    img = img.cpu().numpy().squeeze()
                    ax.imshow(img, cmap='Greens' if correct else 'Reds', interpolation='none')
                    ax.set_title(f'True: {label} Pred: {pred}')
                    ax.axis('off')
                    count += 1
                    if count == 6:
                        plt.tight_layout()
                        plt.show()
                        return

def train(dataloader, model, criteria, optimizer, epochs):
    best_loss = float('inf')
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    for epoch in range(epochs):
        start_time = time.time()  # Starttid vid början av epochen
        model.train()
        print(f"\nEpoch {epoch+1}")

        train_loss = 0
        correct = 0
        total = 0

        for input, target in dataloader:
            input, target = input.to(model.device), target.to(model.device)
            optimizer.zero_grad()
            pred = model(input)
            loss = criteria(pred, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(pred.data, 1)
            correct += (predicted == target).sum().item()
            total += target.size(0)
        
        current_train_loss = train_loss / len(train_loader)
        current_train_accuracy = 100 * correct / total
        train_losses.append(current_train_loss)
        train_accuracies.append(current_train_accuracy)

        test_loss, test_accuracy = evaluate(model, test_loader, criteria)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        print(f'Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accuracies[-1]:.2f}%, '
              f'Test Loss: {test_losses[-1]:.4f}, Test Acc: {test_accuracies[-1]:.2f}%')

        end_time = time.time()  # Sluttid vid slutet av epochen
        duration = end_time - start_time  # Tiden det tar att genomföra epochen
        print(f"Completed in {duration:.2f} seconds")

        # Logga data till TensorBoard
        writer.add_scalar('Epoch training time', duration, epoch)
        writer.add_scalar('Training Loss', train_losses[-1], epoch)
        writer.add_scalar('Training Accuracy', train_accuracies[-1], epoch)
        writer.add_scalar('Test Loss', test_losses[-1], epoch)
        writer.add_scalar('Test Accuracy', test_accuracies[-1], epoch)

        # Spara en checkpoint vid varje epok eller när ett nytt bästa resultat uppnås
        if current_train_loss < best_loss:
            best_loss = current_train_loss
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_loss': best_loss,
                'optimizer' : optimizer.state_dict(),
            }, filename=get_checkpoint_filename(epoch+1))
        evaluate(model, dataloader, criteria)

    plot_progress(train_losses, test_losses, train_accuracies, test_accuracies)

def evaluate(model, dataloader, criteria):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for input, target in dataloader:
            input, target = input.to(model.device), target.to(model.device)
            pred = model(input)
            loss = criteria(pred, target)
            test_loss += loss.item()
            _, predicted = torch.max(pred.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    total_loss = test_loss / len(dataloader)
    accuracy = 100 * correct / total
    return total_loss, accuracy

train(train_loader, model, criteria, optimizer, epochs=epochs)