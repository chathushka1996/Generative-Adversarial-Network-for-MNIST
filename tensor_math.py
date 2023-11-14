# =================================================================================== #
#                             Tensor Math & Comparison Operations                     #
# =================================================================================== #

import torch

x = torch.tensor([1, 2, 4])
y = torch.tensor([9, 8, 6])

# Addition
z1 = torch.empty(3)
z2 = torch.add(x, y, out=z1)
z3 = x + y
torch.add(x, y, out=z1)
print(z1, z2, z3)

# Subtraction
z4 = y - x
print(z4)

# Division
z5 = y / x
print(z5)

# inplace operations
t = torch.zeros(3)
t.add_(x)
t += x
print(t)

# Exponentiation

z = x.pow(2)
z = x ** 2

# Simple comparison
z = x > 0
print(z)

# Matrix Multiplication
x1 = torch.rand((3, 5))
x2 = torch.rand((5, 3))
x3 = torch.mm(x1, x2)
x3 = x2.mm(x1)

print(x3)

# Matrix exponentiation
matrix_exp = torch.rand(5, 5)
print(matrix_exp.matrix_power(3))

# element wise mult
z = x * y
print(z)

# element wise mult
z = torch.dot(x, y)
print(z)

# Batch Matrix Multiplication

batch = 32
n = 10
m = 20
p = 30
tensor_1 = torch.rand((batch, n, m))
tensor_2 = torch.rand((batch, m, p))

out = torch.bmm(tensor_1, tensor_2) # (batch, n, p)
# print(out)


# Example of Broadcasting
x1 = torch.rand((5, 5))
x2 = torch.rand((1, 5))
print(x1, x2)
z = x1 - x2
z = x1 ** x2
print(z)

# Other useful tensor operations
sum_x = torch.sum(x, dim=0)
values, indices = torch.max(x, dim=0)
values, indices = torch.min(x, dim=0)
abs_x = torch.abs(x)
z = torch.argmax(x, dim=0)
z = torch.argmin(x, dim=0)
mean_x = torch.mean(x.float(), dim=0)
z = torch.eq(x, y)
print(z)

sorted_y = torch.sort(y, dim=0, descending=False)
print(sorted_y)


z = torch.clamp(x, min=0)




x = torch.tensor([1,0,1,1,0,1], dtype=torch.bool)
print(x)

z = torch.any(x)
print(z)
z = torch.all(x)
print(z)