import torch
import torch.nn as nn  # all the Nueral network modules, loss functions
import torch.optim as optim  # all optimizations, Stochastic gradient decent, adam
from torch.utils.data import DataLoader  # easier dataset management, mini batches
import torchvision.datasets as datasets  # dataset
import torchvision.transforms.functional as F
import torchvision.transforms as transforms  # transformation that can perform on our dataset
from torchvision.utils import save_image, make_grid
from torch.autograd import Variable
import pprint
import pickle
from PIL import Image

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


with torch.no_grad():
    test_z = Variable(torch.randn(batch_size, z_dim).to(device))
    generated = new_gen(test_z)
    print(generated.size())
    # Reshape the tensor to [100, 28, 28]
    reshaped_tensor = generated.view(100, 28, 28)
    print(reshaped_tensor.size())

    # Create a new blank image for the grid
    grid_image = Image.new('L', (280, 280), color=255)

    # Iterate over the images and paste them into the grid image
    for i, image in enumerate(reshaped_tensor):
        image = F.to_pil_image(image)
        x = (i % 10) * 28
        y = (i // 10) * 28
        grid_image.paste(image, (x, y))

    # Save the grid image
    grid_image.save('image_grid.png')

    fake = new_gen(test_z).reshape(-1, 1, 28, 28)
    for i in range(batch_size):
        text = f'{i}'
        if(i < 10):
            text = f'0{i}'
        save_image(fake[i], f"400/fake_chathushka/4/0{text}.png")

        # Convert tensor to list
        latent_list = test_z[i].tolist()

        # Open a text file in write mode
        with open(f'400/fake_chathushka/4/0{text}.txt', 'w') as file:
            # Write the entire list as a single line in the text file
            file.write(str(latent_list))

        # Close the text file
        file.close()
