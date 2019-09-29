import torch
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt

x = torch.unsqueeze(torch.linspace(-1,1,100),dim=1)   #unsqueeze是将一维数据转换为二维数据（Torch只处理二维数据 ）
y = x.pow(2) + 0.2*torch.rand(x.size())  #x.pow(2)代表x^2
#plt.scatter(x.data.numpy(),y.data.numpy())
#plt.show()

class Net(torch.nn.Module):
    def __init__(self, n_features, n_hidden, n_output):   #搭建这些层需要的信息
        super(Net, self).__init__()   #继承Net到torch.nn.Module这个模块
        self.hidden = torch.nn.Linear(n_features, n_hidden)   #表示输入与隐藏层
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x
net = Net(1, 10, 1)   #分别是输入，隐藏，输出
print(net)


optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
loss_func = torch.nn.MSELoss()


plt.ion()
plt.show()

for t in range(100):
    prediction = net(x)

    loss = loss_func(prediction, y)

    optimizer.zero_grad()   #所有梯度先降为0
    loss.backward()   #用于计算每个节点的梯度
    optimizer.step()   #利用学习率优化每一步的梯度
    if t % 5 == 0:
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'loss =%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)
    plt.ioff()
    plt.show()









