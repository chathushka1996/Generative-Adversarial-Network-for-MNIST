# =================================================================================== #
#                             Initializing Tensor                                     #
# =================================================================================== #

import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
my_tensor = torch.tensor(
    [[1, 2, 3], [1, 2, 4]],
    dtype=torch.float32,
    device=device,
    requires_grad=True
)
print(my_tensor)
print(my_tensor.dtype)
print(my_tensor.device)
print(my_tensor.shape)
print(my_tensor.requires_grad)

# Other common initialization methods
x = torch.empty(size=(3, 3))
x = torch.empty(size=(3, 3)).normal_(mean=0, std=1)
x = torch.empty(size=(3, 3)).uniform_(0, 1)
print(x)
y = torch.zeros((3, 3))  # rand, ones, eye
y = torch.diag(torch.ones(3))
print(y)
z = torch.arange(start=0, end=5, step=1)  # linspace
print(z)

# How to initialize and convert tensors to other types (int, float, double)
tensor = torch.asarray(4)

print(tensor.bool())
print(tensor.short())
print(tensor.long())
print(tensor.half())
print(tensor.float())

# Array to Tensor conversion and vice-versa
import numpy as np
np_array = np.zeros((5, 5))
tensor = torch.from_numpy(np_array)
np_array_back = tensor.numpy()