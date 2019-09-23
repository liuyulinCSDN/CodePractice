# -*- coding: utf-8 -*-

import torch

x = torch.randn(4,4)
y = x.view(16)
z = x.view(-1, 8)
m = x.view(-1,2)
print(x.size(0), y.size(),z.size())

print(x)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Assuming that we are on a CUDA machine, this should print a CUDA device:

print(device)