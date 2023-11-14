# =================================================================================== #
#                             Tensor Indexing                                         #
# =================================================================================== #

import torch

batch_size = 10
features = 25
x = torch.rand((batch_size, features))
print(x)
print(x[0]) # x[0, :]
print(x[0].shape)

print(x[:, 0])

print(x[2, 0:10]) # 0:10 --> [0, 1, 2, ... 9]

x[0, 0] = 100




# Fancy indexing
x = torch.arange(10)
print(x)
indices = [2, 5, 8]

print(x[indices])


x = torch.rand((3, 5))
print(x)
rows = torch.tensor([1, 0])
print(rows)
cols = torch.tensor([4, 0])
print(cols)
print(x[rows, cols])

# More advanced indexing

x = torch.arange(10)
print(x[(x<2) | (x>8)])
print(x[(x<2) & (x>8)])
print(x[x.remainder(2) == 0])


# Useful operations
print(torch.where(x > 5, x, x ** 2))
print(torch.tensor([0, 0, 1, 1, 2, 3]).unique())
print(x.ndimension())
print(x.numel())