# Imports
import torch
import torch.nn as nn  # all the Nueral network modules, loss functions
import torch.optim as optim  # all optimizations, Stochastic gradient decent, adam
import torch.nn.functional as F  # all the function doesn't have parameter, activation fun relu, tanh
from torch.utils.data import DataLoader  # easier dataset management, mini batches
import torchvision.datasets as datasets  # dataset
import torchvision.transforms as transforms  # transformation that can perform on our dataset


# Create simple CNN
class CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=8,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1)
        )
        self.pool = nn.MaxPool2d(
            kernel_size=(2, 2),
            stride=(2, 2)
        )
        self.conv2 = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1)
        )
        self.fc1 = nn.Linear(
            16*7*7,
            num_classes
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)

        return x


# model = NN(784, 10)
# x = torch.rand(64, 784)
# print(model(x).shape)
# model = CNN()
# x = torch.randn(64, 1, 28, 28)
# print(x.shape)
# print(model(x)[1])
def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> saving checkpoint")
    torch.save(state, filename)
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
in_channels = 1
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 5

# Load Data
train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# Initialize network
model = CNN(in_channels=in_channels, num_classes=num_classes).to(device)

# Loss and optimizer
c_iteration = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train Network
for epoch in range(num_epochs):
    losses = []
    if epoch == 2:
        checkpoint = {
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        save_checkpoint(checkpoint)


    for batch_idx, (data, targets) in enumerate(train_loader):

        # Get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)

        # forward
        scores = model(data)
        loss = c_iteration(scores, targets)
        losses.append(loss)

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()
    total_loss = 0
    count = 0
    for i in losses:
        total_loss += i.item()
        count += 1
    print(f'Loss at epoch 0 was {total_loss/count}')

# Check accuracy on training & test to see how good our model
def check_accuracy(loader, model):
    if loader.dataset.train:
        print("Checking accuracy on training data")
    else:
        print("Checking accuracy on test data")

    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f'Got {num_correct}/{num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')

    model.train()

check_accuracy(train_loader, model)
check_accuracy(test_loader, model)