import torch
import torch.utils.data as Data
import torch.nn.functional as F
import matplotlib.pyplot as plt

#我们做了几个超参数
LR=0.01
BATCH_SIZE=32
EPOCH=12

x = torch.unsqueeze(torch.linspace(-1, 1, 1000), dim=1)
y = x.pow(2) + 0.1*torch.normal(torch.zeros(*x.size()))

# plt.scatter(x.numpy(), y.numpy())
# plt.show()           #打印出的是回归的数据点，我们就用这一批的数据来训练

torch_dataset = Data.TensorDataset(x, y)
loader=Data.DataLoader(dataset=torch_dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=2)

class Net(torch.nn.Module):                #建立神经网络的框架
    def __init__(self):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(1, 20)
        self.predict = torch.nn.Linear(20, 1)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x

#我们用默认的建立神经网络的框架来建立四个神经网络，并且四个神经网络用不同的优化器来优化
if __name__ == '__main__':
   net_SGD= Net()
   net_Momentum= Net()
   net_RMSprop = Net()
   net_Adam= Net()
   nets = [net_SGD, net_Momentum, net_RMSprop, net_Adam]  #将四个神经网络全部放在一个数组当中，之后用for循环将他们提取出来，一个个用不同的优化器进行优化

#建立四个不同的优化器来优化四个架构一致的神经网络
   opt_SGD=torch.optim.SGD(net_SGD.parameters(), lr=LR)
   opt_Momentum=torch.optim.SGD(net_Momentum.parameters(), lr=LR, momentum=0.8)   #可以将SGD看成momentum=0的Momentum
   opt_RMSprop =torch.optim.RMSprop(net_RMSprop.parameters(), lr=LR, alpha=0.9)
   opt_Adam= torch.optim.Adam(net_Adam.parameters(), lr=LR, betas=(0.9, 0.99))
   optimizers=[opt_SGD, opt_Momentum, opt_RMSprop, opt_Adam]

   loss_func=torch.nn.MSELoss() #回归的误差计算公式
   losses_his=[[],[],[],[]]     #为了把不同的优化器优化的神经网络的误差记录下来，我们创造了一个数组

   for epoch in range(EPOCH):
      print('Epoch: ', epoch)
      for step,(b_x,b_y) in enumerate(loader):
         for net,opt,l_his in zip(nets,optimizers,losses_his):
             output = net(b_x)
             loss = loss_func(output, b_y)
             opt.zero_grad()
             loss.backward()
             opt.step()
             l_his.append(loss.data.numpy())   #将误差放入记录之中

   labels = ['SGD', 'Momentum', 'RMSprop', 'Adam']
   for i, l_his in enumerate(losses_his):
       plt.plot(l_his, label=labels[i])
   plt.legend(loc='best')
   plt.xlabel('Steps')
   plt.ylabel('Loss')
   plt.ylim((0, 0.2))
   plt.show()




