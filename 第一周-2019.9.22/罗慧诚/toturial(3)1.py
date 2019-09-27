import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

x=torch.unsqueeze(torch.linspace(-1,1,100),dim=1)
y=x.pow(2)+0.2*torch.rand(x.size())

x, y=Variable(x),Variable(y) #因为神经网络只能输入Variable形式的

#plt.scatter(x.data.numpy(),y.data.numpy()) //scatter为打印散点图
#plt.show()   //这个为散点图显示

class Net(torch.nn.Module):      #继承从module的模块,括号里为继承
    def __init__(self,n_feature,n_hidden,n_output):          #定义搭建神经网络这些层所需要的信息
        super(Net,self).__init__()
        self.hidden=torch.nn.Linear(n_feature,n_hidden)    #隐藏层有多少数据个数输入，输出多少神经元
        self.predict=torch.nn.Linear(n_hidden,n_output)            #预测层有多少神经元从隐藏层接受，


    def forward(self,x):           #神经网络前项传递的一个过程，就是我们前项信息放在forward上一个个组合起来，流程图就是forward这个过程所做的事情
        x=F.relu(self.hidden(x))             #神经网络搭建过程
                                             #用激励函数激活一下信息。在激活函数之前，我们要使用hidden层来加工一下我们输入的信息，并且输出N个神经元的个数，输出之后，我们再用激活函数将其激活
                                             #之前我们有讲过激活函数的用途，即嵌套出隐藏层的信息
        x=self.predict(x)
        return x                   #为什么我们预测时不用激励函数呢？ 因为在大多数回归问题中，我们预测的值分布可以从负无穷到正无穷，如果用了激励函数，那么当x<0时，y=0.就相当于预测值中有一段被截断了

net =Net(1,10,1)          #输入值只有一个，因为我的x值只包含了一个点的一个值信息，而我的隐藏层有10个神经元，我的输出值（y）只有一个
print(net)

plt.ion()            #首先要设置plt为一个实时打印的过程
plt.show()


optimizer=torch.optim.SGD(net.parameters(),lr=0.5)       #用优化器来优化神经网络参数,lr为学习效率，一般学的越快，lr就越高，但是学的越快，反而有时候更不好，因为学的越快，很多东西都漏掉了
loss_func=torch.nn.MSELoss()                 #loss_func为神经网络中计算误差的手段，MESLoss为均方差来处理回归问题就足以应对

for t in range(100):                     #我们训练100步试试看
     prediction=net(x)                   #算出我们的预测值

     loss=loss_func(prediction,y)        #计算预测值和真实值的误差
     optimizer.zero_grad()               #把所有神经网络（net）中的梯度设为0
     loss.backward()                     #给每一个神经网络中的节点（我觉得应该是神经元）计算出他的梯度grad
     optimizer.step()                    #以学习效率0.5来优化我这种梯度（即用optimizer来优化神经网络的梯度）
     if t % 5==0:                          #每五步我们就打印一次图上的信息
         plt.cla()
         plt.scatter(x.data.numpy(), y.data.numpy())
         plt.plot(x.data.numpy(),prediction.data.numpy(),'r-',lw=5)  #神经网络现在学习到什么程度了
         plt.text(0.5,0,'Loss=%.4f' % loss.data.numpy(),fontdict={'size':20,'color':'red'}) #打印出学习网络它开始学习时的误差是多少
         plt.pause(0.1)

plt.ioff()
plt.show()









