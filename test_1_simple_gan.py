import torch
import torch.nn as nn  # all the Nueral network modules, loss functions
import torch.optim as optim  # all optimizations, Stochastic gradient decent, adam
import torchvision.utils
from torch.utils.data import DataLoader  # easier dataset management, mini batches
import torchvision.datasets as datasets  # dataset
import torchvision.transforms as transforms  # transformation that can perform on our dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image


class Discriminator(nn.Module):
    def __init__(self, img_dim=16*7*7):
        super(Discriminator, self).__init__()
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
            28*28*16,
            1
        )
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()
        self.leakRelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.leakRelu(x)
        # x = self.pool(x)
        x = self.conv2(x)
        x = self.leakRelu(x)
        # x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        x = self.sig(x)
        return x


class Generator(nn.Module):
    def __init__(self, z_dim, img_dim):
        super(Generator, self).__init__()
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
            16*10*10,
            28 * 28 * 1
        )
        self.tan = nn.Tanh()
        self.leakRelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.leakRelu(x)
        x = self.conv2(x)
        x = self.leakRelu(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        x = self.tan(x)
        return x


# Hyperparameters etc.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lr = 3e-4
z_dim = 16*10*10
img_dm = 28 * 28 * 1
batch_size = 32
num_epochs = 50
gen_dim = 10

disc = Discriminator(img_dm).to(device)
gen = Generator(z_dim, img_dm).to(device)
fixed_noise = torch.randn((batch_size, z_dim)).to(device)

transforms = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

dataset = datasets.MNIST(root="dataset/", transform=transforms, download=True)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
opt_disc = optim.Adam(disc.parameters(), lr=lr)
opt_gen = optim.Adam(gen.parameters(), lr=lr)
criterion = nn.BCELoss()

writer_fake = SummaryWriter(f'runs/GAN_MNIST/fake')
writer_real = SummaryWriter(f'runs/GAN_MNIST/real')

step = 0

for epoch in range(num_epochs):
    for batch_idx, (real, _) in enumerate(loader):
        real = real.to(device)
        batch_size = real.shape[0]

        ### Train Discriminator: max log(D(real))) + log(1 - D(G(z)))
        noise = torch.randn(batch_size, 1, gen_dim, gen_dim).to(device)
        fake = gen(noise)
        disc_real = disc(real)
        lossD_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(real)
        lossD_fake = criterion(disc_fake, torch.ones_like(disc_fake))
        lossD = (lossD_real + lossD_fake) / 2
        disc.zero_grad()
        lossD.backward(retain_graph=True)
        opt_disc.step()

        ### Train Generator min log(1 - D(G(Z))) <--> max log(D(G(z)))
        output = disc(fake).view(-1)
        lossG = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        lossG.backward()
        opt_gen.step()

        if batch_idx == 0:
            print(
                f"Epoch [{epoch}/{num_epochs}] Loss D: {lossD: .4f}, Loss G: {lossG: .4f}"
            )

            with torch.no_grad():
                print(fixed_noise.shape)
                fake = gen(fixed_noise).reshape(-1, 1, 28, 28)
                # print(fake[i].shape)  # torch.Size([64,3,28,28])
                # img1 = images[0]  # torch.Size([3,28,28]
                # img1 = img1.numpy() # TypeError: tensor or list of tensors expected, got <class 'numpy.ndarray'>
                for i in range(32):
                    save_image(fake[i], f"image/fake/gray_img_{step}_{i}.jpg")
                data = real.reshape(-1, 1, 28, 28)
                for i in range(32):
                    save_image(data[i], f"image/real/gray_img_{step}_{i}.jpg")
                img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                img_grid_real = torchvision.utils.make_grid(data, normalize=True)

                writer_fake.add_image(
                    "MNIST Fake Images", img_grid_fake, global_step=step
                )

                writer_real.add_image(
                    "MNIST Fake Images", img_grid_real, global_step=step
                )

                step += 1
