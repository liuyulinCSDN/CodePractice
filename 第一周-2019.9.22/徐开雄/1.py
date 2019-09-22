import torch
import numpy as np


np_data = np.arange(6).reshape((2,3))
torch_data = torch.from_numpy(np_data)
tensor2array = torch_data.numpy()

#abs
data = [-1,-1,1,-8,-9]
tensor = torch.FloatTensor(data)

print(
    '\nnumpy:\n',np_data,
    '\ntorch:\n',torch_data,
    '\ntensor2array:\n',tensor2array,
    '\n\nabs:',
    '\nnumpy:\n',np.abs(data),
    '\naverage:\n',np.mean(data),
    '\ntorch:\n',torch.abs(tensor)
)

