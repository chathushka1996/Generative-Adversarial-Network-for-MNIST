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
    def __init__(self, img_dim):
        super().__init__()
        self.fc1 = nn.Linear(img_dim, 128)
        self.leakyRelu = nn.LeakyReLU(0.1)
        self.fc2 = nn.Linear(128, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.leakyRelu(x)
        x = self.fc2(x)
        x = self.sig(x)
        return x


class Generator(nn.Module):
    def __init__(self, z_dim, img_dim):
        super().__init__()
        self.fc1 = nn.Linear(z_dim, 256)
        self.leakyRelu = nn.LeakyReLU(0.1)
        self.fc2 = nn.Linear(256, img_dim)
        self.tan = nn.Tanh()

    def forward(self, x):
        x = self.fc1(x)
        x = self.leakyRelu(x)
        x = self.fc2(x)
        x = self.tan(x)
        return x


# Hyperparameters etc.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lr = 3e-4
z_dim = 100
img_dm = 28 * 28 * 1
batch_size = 100
num_epochs = 50

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
        real = real.view(-1, 784).to(device)
        batch_size = real.shape[0]

        ### Train Discriminator: max log(D(real))) + log(1 - D(G(z)))
        noise = torch.randn(batch_size, z_dim).to(device)
        fake = gen(noise)
        disc_real = disc(real).view(-1)
        lossD_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake).view(-1)
        lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
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
                for i in range(batch_size):
                    save_image(fake[i], f"image/fake/gray_img_{step}_{i}.jpg")
                data = real.reshape(-1, 1, 28, 28)
                for i in range(batch_size):
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
