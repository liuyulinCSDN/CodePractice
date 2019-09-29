import torch as t
import torch.nn.functional as F
import matplotlib.pyplot as plt
# 假数据
n_data = t.ones(100, 2)         # 数据的基本形态
x0 = t.normal(2*n_data, 1)      # 类型0 x data (tensor), shape=(100, 2)
y0 = t.zeros(100)               # 类型0 y data (tensor), shape=(100, )
x1 = t.normal(-2*n_data, 1)     # 类型1 x data (tensor), shape=(100, 1)
y1 = t.ones(100)                # 类型1 y data (tensor), shape=(100, )

# 注意 x, y 数据的数据形式是一定要像下面一样 (torch.cat 是在合并数据)
x = t.cat((x0, x1), 0).type(t.FloatTensor)  # FloatTensor = 32-bit floating
y = t.cat((y0, y1), ).type(t.LongTensor)    # LongTensor = 64-bit integer

plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy(), s=100, lw=0, cmap='RdYlGn')
plt.show()

# 画图
# plt.scatter(x.data.numpy(), y.data.numpy())
# plt.show()
# class Net(t.nn.Module):
#     def __init__(self, n_future, n_hidden, n_output):
#         super(Net, self).__init__()
#         self.hidden = t.nn.Linear(n_future, n_hidden)   #linear定义输入输出
#         self.output = t.nn.Linear(n_hidden, n_output)
#
#     def forward(self, x):
#         x = F.relu(self.hidden(x))
#         x = self.output(x)
#         return x
# net = Net(n_future=2,n_hidden=10,n_output=2)

net2 = t.nn.Sequential(
    t.nn.Linear(2, 10),
    t.nn.ReLU(),
    t.nn.Linear(10, 2),
)
print(net2)

optimizer = t.optim.SGD(net2.parameters(), lr=0.2)   #优化器函数定义的时候，先放要优化的网络的参数，后加入网络的学习率
loss_func = t.nn.CrossEntropyLoss()


for t in range(100000):
    out = net2(x)     # 喂给 net 训练数据 x, 输出分析值
    loss = loss_func(out, y)     # 计算两者的误差
    optimizer.zero_grad()   # 清空上一步的残余更新参数值
    loss.backward()         # 误差反向传播, 计算参数更新值
    optimizer.step()
    if t % 500 == 0:
        print('the loss eq to', loss)
########训练的一般方法##########
####1.让网络输出
####2.计算损失
####3.重置梯度
####4.反向传播，依据步长
###############################


