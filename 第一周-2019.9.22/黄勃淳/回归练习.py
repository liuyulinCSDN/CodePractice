import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
y = x.pow(2) + 0.2*torch.rand(x.size())

x, y = Variable(x), Variable(y)
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):   #搭建层的信息
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x#传递
net = Net(n_feature=1, n_hidden=10, n_output=1)
# print(net)


optimizer = torch.optim.SGD(net.parameters(), lr=0.2)
loss_func = torch.nn.MSELoss()   #设置损失函数计算方式为均方误差

for t in range(100000):
    prediction = net(x)
    loss = loss_func(prediction, y)   #最好是真值在后

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if t % 10000 == 0:
        # plot and show learning process
        print('the loss eq ', loss, 'and the Y is ', prediction)
