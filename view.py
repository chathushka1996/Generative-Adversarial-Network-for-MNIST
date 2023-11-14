import torch

x = torch.randn(1, 5, 7)
x = x.view(-1)
print(x.size())

x = torch.randn(2, 4)
x = x.view(-1, 8)
print(x.size())

x = torch.randn(2, 4)
x = x.view(-1)
print(x.size())

x = torch.randn(2, 4, 3)
x = x.view(-1)
print(x.size())

# prerequisites
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image


class Generator(nn.Module):
    def __init__(self, z_dim, in_channels, out_channels):
        super().__init__()
        self.conv_trans1 = nn.ConvTranspose2d(z_dim, out_channels*16, 4, 1, 0)
        self.conv_trans2 = nn.ConvTranspose2d(self.conv_trans1.out_channels, int(self.conv_trans1.out_channels/2), 4, 1, 0, bias=False)
        self.conv_trans3 = nn.ConvTranspose2d(self.conv_trans2.out_channels, int(self.conv_trans1.out_channels/2), 4, 1, 0, bias=False)
        self.conv_trans4 = nn.ConvTranspose2d(self.conv_trans3.out_channels, int(self.conv_trans1.out_channels/2), 4, 1, 0, bias=False)
        self.conv_trans5 = nn.ConvTranspose2d(self.conv_trans4.out_channels, in_channels, 4, 1, 1)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.tan_h = nn.Tanh()

    def forward(self, x):
        x = self.leaky_relu(self.conv_trans1(x))
        x = self.leaky_relu(self.conv_trans2(x))
        x = self.leaky_relu(self.conv_trans3(x))
        x = self.leaky_relu(self.conv_trans4(x))
        x = self.leaky_relu(self.conv_trans5(x))
        return torch.tanh(x)

model = Generator(100, 1, 8)
x = torch.randn(100, 100, 1, 1)
print(x.shape)
print(model(x)[1])