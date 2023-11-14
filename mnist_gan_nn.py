#-----------------------------------------------------------------------------------------------------
import torch
import torch.nn as nn  # all the Nueral network modules, loss functions
import torch.optim as optim  # all optimizations, Stochastic gradient decent, adam
from torch.utils.data import DataLoader  # easier dataset management, mini batches
import torchvision.datasets as datasets  # dataset
import torchvision.transforms as transforms  # transformation that can perform on our dataset
from torchvision.utils import save_image
import pprint
import pickle


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


# Hyperparameters etc.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lr = 0.0002
z_dim = 100
img_dm = 28 * 28 * 1
batch_size = 100
num_epochs = 50

disc = Discriminator(img_dm).to(device)
gen = Generator(z_dim, img_dm).to(device)

transforms = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

dataset = datasets.MNIST(root="dataset/", transform=transforms, download=True)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
opt_disc = optim.Adam(disc.parameters(), lr=lr)
opt_gen = optim.Adam(gen.parameters(), lr=lr)
criterion = nn.BCELoss()


for epoch in range(num_epochs):
    D_loss_array = []
    G_loss_array = []
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
        D_loss_array.append((lossD))
        disc.zero_grad()
        lossD.backward(retain_graph=True)
        opt_disc.step()

        ### Train Generator min log(1 - D(G(Z))) <--> max log(D(G(z)))
        output = disc(fake).view(-1)
        lossG = criterion(output, torch.ones_like(output))
        G_loss_array.append(lossG)
        gen.zero_grad()
        lossG.backward()
        opt_gen.step()

    # print('[%d/%d]: loss_d: %.3f, loss_g: %.3f' % (
    #     (epoch), n_epoch, torch.mean(torch.FloatTensor(D_losses)), torch.mean(torch.FloatTensor(G_losses))))
    print(
        f"Epoch [{epoch}/{num_epochs}] Loss D: {torch.mean(torch.FloatTensor(D_loss_array)): .4f}, Loss G: {torch.mean(torch.FloatTensor(G_loss_array)): .4f}"
    )


with torch.no_grad():

    for i in range(10):
        temp_noise = torch.randn((batch_size, z_dim)).to(device)
        fake = gen(temp_noise).reshape(-1, 1, 28, 28)
        for j in range(batch_size):
            save_image(fake[i], f"image/fake_chamath/1/{i}{j}.png")

            # Convert tensor to list
            tensor_list = fake[i][0].tolist()

            # Open a text file in write mode
            with open(f'image/fake_chamath/1/{i}{j}.txt', 'w') as file:
                # Write the entire list as a single line in the text file
                file.write(str(tensor_list))

            # Close the text file
            file.close()



pp = pprint.PrettyPrinter(indent=4)
# print('Discriminator Save')
# pp.pprint(disc.state_dict())

# Save model
with open("D.pkl", "wb") as fp:
    pickle.dump(disc.state_dict(), fp)

new_disc = Discriminator(img_dm)
with open("D.pkl", "rb") as fp:
    new_disc.load_state_dict(pickle.load(fp))
# print('Discriminator Load')
# pp.pprint(new_disc.state_dict())


# print('Generator Save')
# pp.pprint(gen.state_dict())

# Save model
with open("G.pkl", "wb") as fp:
    pickle.dump(gen.state_dict(), fp)

new_gen = Generator(z_dim, img_dm)
with open("G.pkl", "rb") as fp:
    new_gen.load_state_dict(pickle.load(fp))

# print('Generator Load')
# pp.pprint(new_gen.state_dict())
