import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

n_data = torch.ones(100, 2)
x0 = torch.normal(2 * n_data, 1)   #class x data (tensor),shape=(100,2)
y0 = torch.zeros(100)   #x0的标签
x1 = torch.normal(-2 * n_data, 1)
y1 = torch.ones(100)   #x1的标签,class y data (tensor),shape=(100,1)
x = torch.cat((x0, x1), 0).type(torch.FloatTensor)   #FloatTensor = 32bit floating
y = torch.cat((y0, y1), ).type(torch.LongTensor)   #LongTensor = 64bit integer

x, y = Variable(x), Variable(y)


# plt.scatter(x.data.numpy(),y.data.numpy())
# plt.show()

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = torch.relu(self.hidden(x))
        x = self.predict(x)
        return x


net = Net(n_feature=2, n_hidden=10, n_output=2)
print(net)

optimizer = torch.optim.SGD(net.parameters(), lr=0.02)
loss_func = torch.nn.CrossEntropyLoss()

plt.ion()
plt.show()

for t in range(100):
    out = net(x)

    loss = loss_func(out, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if t % 2 == 0:
        plt.cla()
        prediction = torch.max(out, 1)[1]
        pred_y = prediction.data.numpy()
        target_y = y.data.numpy()
        plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
        accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)
        plt.text(1.5, -4, 'Accuracy =%.2f' % accuracy, fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)
plt.ioff()
#torch.save(net1, 'net.pkl')#保存整个网络entire net
#torch.save(net1.state_dict(), 'net_params.pkl') #只保留计算图里面的参数parameters

#def restore_net():#提取整个网络数据
    net2 = torch.load('net.pkl')

#def restore_params():#提取整个网络参数
    net3 = torch.nn.Sequential(
        torch.nn.Linear(1, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1)
    )
    net3.load_state_dict(torch.load('net_params.pkl'))