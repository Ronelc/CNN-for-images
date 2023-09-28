import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

# Set device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Define the CNN architecture
class CNN(nn.Module):
    def __init__(self, out_channels):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=out_channels,
                               kernel_size=3,
                               stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels, kernel_size=3,
                               stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels, kernel_size=3,
                               stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(in_features=out_channels * 4 * 4, out_features=10)

    def forward(self, x, out_channels, is_nonlinear=True):
        x = self.conv1(x)
        if is_nonlinear: x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        if is_nonlinear: x = self.relu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        if is_nonlinear: x = self.relu3(x)
        x = self.pool3(x)
        x = x.view(-1, out_channels * 4 * 4)
        x = self.fc1(x)
        return x


def create_data_set(batch_size):
    # Load the CIFAR10 dataset
    train_dataset = datasets.CIFAR10(root='data/', train=True,
                                     transform=transforms.ToTensor(),
                                     download=True)
    test_dataset = datasets.CIFAR10(root='data/', train=False,
                                    transform=transforms.ToTensor(),
                                    download=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                              shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size,
                             shuffle=False)
    return train_loader, test_loader


def train_model(out_channels, train_loader, num_epochs, model, criterion,
                optimizer, is_nonlinear):
    # Train the model
    train_loss = 0.0
    total = 0
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images, out_channels, is_nonlinear)
            loss = criterion(outputs, labels)
            train_loss += loss.item() * labels.size(0)
            total += labels.size(0)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print loss every 100 iterations
            if (i + 1) % 100 == 0:
                print(
                    f'Epoch [{epoch + 1}/{num_epochs}], Iteration [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')
    avg_train_loss = train_loss / total
    return (avg_train_loss)


def test_model(model, test_loader, out_channels, criterion, is_nonlinear):
    # Test the model
    test_loss = 0.0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images, out_channels, is_nonlinear)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * labels.size(0)
            total += labels.size(0)

    avg_test_loss = test_loss / total
    print('Average loss on the test set: %.3f' % avg_test_loss)
    return avg_test_loss


def plot_loss(loss_arr, out_channels, title):
    plt.plot(out_channels, loss_arr)
    plt.ylabel("Loss")
    plt.xlabel("num of channels")
    plt.title(f"{title} loss Curve ")
    plt.show()


def run_model(class_, out_channels, is_nonlinear=True):
    test_loss_array, train_loss_array = [], []

    for num_of_channels in out_channels:
        # Initialize the model, loss function, and optimizer
        model = class_(num_of_channels).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        train_loss = train_model(num_of_channels, train_loader, num_epochs,
                                 model, criterion, optimizer, is_nonlinear)
        test_loss = test_model(model, test_loader, num_of_channels, criterion,
                               is_nonlinear)
        test_loss_array.append(test_loss)
        train_loss_array.append(train_loss)

    plot_loss(train_loss_array, out_channels, "Train")
    plot_loss(test_loss_array, out_channels, "Test")
    print(
        f"lowest Loss received with {out_channels[np.argmin(test_loss_array)]} channels, minimal Loss: {min(test_loss_array)}")
    return np.argmin(test_loss_array)


class CNN_Q4(nn.Module):
    def __init__(self, out_channels):
        super(CNN_Q4, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=out_channels,
                               kernel_size=3)
        self.fc = nn.Linear(in_features=out_channels * 30 * 30,
                            out_features=10)

    def forward(self, x, out_channels, is_nonlinear=True):
        x = self.conv1(x)
        x = F.relu(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    # Set hyperparameters
    batch_size = 128
    learning_rate = 0.001
    num_epochs = 10

    train_loader, test_loader = create_data_set(batch_size)
    out_channels = [2, 4, 8, 16, 32, 64, 128, 512]

    # Q1
    argmin = run_model(CNN, out_channels, is_nonlinear=True)

    # Q2
    run_model(CNN, [128, 512, 1024, 2048], is_nonlinear=False)

    # Q3
    run_model(CNN_Q4, [512])
