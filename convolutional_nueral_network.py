# Imports
import torch
import torch.nn as nn  # all the Nueral network modules, loss functions
import torch.optim as optim  # all optimizations, Stochastic gradient decent, adam
from torch.utils.data import DataLoader  # easier dataset management, mini batches
import torchvision.datasets as datasets  # dataset
import torchvision.transforms as transforms  # transformation that can perform on our dataset
import pprint
import pickle


# Create simple CNN
class CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(16*7*7, num_classes)
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


# model = CNN()
# x = torch.randn(64, 1, 28, 28)
# print(x.shape)
# print(model(x)[1])

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
in_channels = 1
num_classes = 10
learning_rate = 0.001
batch_size = 100
num_epochs = 50

# Load Data
train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# Initialize network
Classifier = CNN(in_channels=in_channels, num_classes=num_classes).to(device)

# Loss and optimizer
c_iteration = nn.CrossEntropyLoss()
optimizer = optim.Adam(Classifier.parameters(), lr=learning_rate)

def check_accuracy(loader, model):
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

    return float(num_correct)/float(num_samples)*100

# Train Network
for epoch in range(num_epochs):
    loss_array = []
    for batch_idx, (data, targets) in enumerate(train_loader):

        # Get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)

        # forward
        scores = Classifier(data)
        loss = c_iteration(scores, targets)
        loss_array.append(loss)
        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()
    accuracy = check_accuracy(train_loader, Classifier)
    print(
        f"Epoch [{epoch}/{num_epochs}] Loss C: {torch.mean(torch.FloatTensor(loss_array)): .4f} Accuracy C:{accuracy: .4f}%"
    )

pp = pprint.PrettyPrinter(indent=4)
print('Classifier Save')
pp.pprint(Classifier.state_dict())

with open("C.pkl", "wb") as fp:
    pickle.dump(Classifier.state_dict(), fp)

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

check_accuracy(train_loader, Classifier)
check_accuracy(test_loader, Classifier)