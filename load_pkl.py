import torch
import torch.nn as nn  # all the Nueral network modules, loss functions
import torch.optim as optim  # all optimizations, Stochastic gradient decent, adam
from torch.utils.data import DataLoader  # easier dataset management, mini batches
import torchvision.datasets as datasets  # dataset
import torchvision.transforms as transforms  # transformation that can perform on our dataset
from torchvision.utils import save_image
import pprint
import pickle

# Hyperparameters etc.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Discriminator(nn.Module):
    def __init__(self, img_dim):
        super().__init__()
        self.fc1 = nn.Linear(img_dim, 1024)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features // 2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features // 2)
        self.fc4 = nn.Linear(self.fc3.out_features, 1)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.leaky_relu(self.fc1(x))
        x = self.dropout(x)
        x = self.leaky_relu(self.fc2(x))
        x = self.dropout(x)
        x = self.leaky_relu(self.fc3(x))
        x = self.dropout(x)
        return torch.sigmoid(self.fc4(x))


class Generator(nn.Module):
    def __init__(self, z_dim, img_dim):
        super().__init__()
        self.fc1 = nn.Linear(z_dim, 256)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features * 2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features * 2)
        self.fc4 = nn.Linear(self.fc3.out_features, img_dim)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.leaky_relu(self.fc1(x))
        x = self.leaky_relu(self.fc2(x))
        x = self.leaky_relu(self.fc3(x))
        return torch.tanh(self.fc4(x))

z_dim = 100
img_dm = 28 * 28 * 1
batch_size = 100
new_gen = Generator(z_dim, img_dm).to(device)
pp = pprint.PrettyPrinter(indent=4)

with open("400/G.pkl", "rb") as fp:
    new_gen.load_state_dict(pickle.load(fp))

print('Generator Load')
pp.pprint(new_gen.state_dict())

with torch.no_grad():

    for i in range(10):
        temp_noise = torch.randn((batch_size, z_dim)).to(device)
        fake = new_gen(temp_noise).reshape(-1, 1, 28, 28)
        for j in range(batch_size):
            save_image(fake[i], f"image/fake_103/{i}{j}.png")