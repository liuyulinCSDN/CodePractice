import torch
import numpy as np

data=[[1,2],[3,4]]
data1=[1,2,3,4]
tensor=torch.FloatTensor(data)  #转换为32bit浮点数
tensor1=torch.FloatTensor(data1)  #转换为32bit浮点数
data=np.array(data)
data1=np.array(data1)

print(
    '\nnumpy:',np.matmul(data,data),
    '\nnumpy:',data.dot(data),
    '\ntorch:', torch.mm(tensor, tensor),
    '\ntorch:', tensor1.dot(tensor1)
)