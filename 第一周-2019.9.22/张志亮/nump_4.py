import torch
import numpy as np

#abs
data=[-1,-2,1,2]
tensor=torch.FloatTensor(data)  #转换为32bit浮点数
print(
    '\nabs',
    '\nnumpy',np.abs(data),
    '\nnumpy', np.mean(data),
     '\ntorch',torch.mean(tensor)
)