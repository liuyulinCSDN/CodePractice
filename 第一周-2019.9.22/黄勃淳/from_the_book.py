# from __future__ import print_function
# import torch as t
# x = t.Tensor(5, 3)
# print(x)
# x = t.rand(5, 3)
# print(x.size())
# print('hang', x.size()[0], 'lie', x.size()[1])
#
#
# ############add
#
# y = t.rand(5, 3)
# ######1
# print(x + y)
# #####2
# print(t.add(x, y))
# #######3
# result = t.Tensor(5, 3)
# t.add(x, y, out=result)  #####区别在于有没有预留的result而已
# print(result)
# print('org y', y)
# y.add(x)
# print('first', y)
# y.add_(x)
# print('second', y)  ########函数名字加下划线的都会改变Tensor的值
#
# print(x[1, :])
#
# a = t.ones(5)
# b = a.numpy()
# print(a, b)
import numpy as np
import torch as t
# a = np.ones(5)
# b = t.from_numpy(a)
# print(a, b)
# b.add_(1)
# print(a, b)
# x = t.rand(5, 3)
# y = x.add_(x)
# if t.cuda.is_available():
#     x = x.cuda()
#     y = y.cuda()
#     print(x+y)

from torch.autograd import Variable
x = Variable(t.ones(2, 2))
print(x)
y = x.sum()
print(y)
y.grad_fn
y.backward()
y = (x[0][0], x[1][0], x[0][1], x[1][1])
print(x.grad)



