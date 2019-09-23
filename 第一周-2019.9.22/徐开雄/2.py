import torch
import numpy as np

data = [[1,2],[3,4]]
tensor = torch.FloatTensor(data) #32位浮点
data = np.array(data)

print(
    '\nnumpy:\n',data.dot(data),
    '\ntorch:\n',tensor.mm(tensor)
)