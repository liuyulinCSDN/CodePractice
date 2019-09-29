import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.autograd import Variable


tensor = torch.FloatTensor([[1,2],[3,4]])
variable = Variable(tensor,requires_grad=True)


t_out = torch.mean(tensor*tensor)
v_out = torch.mean(variable*variable)

v_out.backward()
print(variable.grad)

np_data = np.arange(6).reshape([2,3])
torch_data = torch.from_numpy(np_data)
tensor2array = torch_data.numpy()

print(np_data)
print(torch_data)
print(tensor2array)

#abs
data = [[-1,-2],[3,-2]]
print(data)
tensor = torch.FloatTensor(data)
print(tensor)
print(
    'abs',
    '\nnumpy:',np.matmul(data,data),
    '\ntorch',torch.mm(tensor,tensor)
)