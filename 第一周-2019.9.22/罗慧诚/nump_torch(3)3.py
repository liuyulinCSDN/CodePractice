import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

n_data=torch.ones(100,2)     #这是我们数据的一个基数
x0=torch.normal(2*n_data,1)    #这群要分类数据的其中一堆数据横坐标和纵坐标都包含在x0中.因为x=2的坐标轴上左右分布，故这堆数据的标签为0
y0=torch.zeros(100)          #这堆数据的标签它类型为0
x1=torch.normal(-2*n_data,1)   #这群要分类数据的另一堆数据横坐标和纵坐标都包含在x1中.因为x=-2的坐标轴上左右分布，故这堆数据的标签为1
y1=torch.ones(100)             #x1这堆数据的标签都为1
x=torch.cat((x0,x1),0).type(torch.FloatTensor)   #将x都合并在一起，当做数据，x认定为64位的浮点数
y=torch.cat((y0,y1),).type(torch.LongTensor)     #将y都合并在一起，当作标签，y认定为64位的长整型

x, y=Variable(x),Variable(y) #因为神经网络只能输入Variable形式的

#plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy(), s=100, lw=0, cmap='RdYlGn')
#plt.show()

#method1
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

net1 =Net(2,10,2)          #输入输出为两个特征，因为x0,x1代表x和y两个坐标轴上的值，有10个神经元
#[0,1]  输出为[0,1],即默认分类为1，即1在哪个位置，就认为分类为哪个
#[1,0]  分类为0
#[0,0,1] 三分类例子，分在第三类
#[1,0,0]分类为第一类

#method2
net2=torch.nn.Sequential(      #即在括号之中一层层的累神经层
     torch.nn.Linear(2,10),
     torch.nn.ReLU(),          #激励函数也可以当成神经层,是当成层的一个类，所以它会有名字。而func.relu（）是当成一个功能，故它没有名字。而两种方法搭建的效果是一样的
     torch.nn.Linear(10,2),

)

print(net1)
print(net2)



# plt.ion()            #首先要设置plt为一个实时打印的过程
# plt.show()
#
#
# optimizer=torch.optim.SGD(net.parameters(),lr=0.02)       #用优化器来优化神经网络参数,lr为学习效率，一般学的越快，lr就越高，但是学的越快，反而有时候更不好，因为学的越快，很多东西都漏掉了
# loss_func=torch.nn.CrossEntropyLoss()                 #针对于分类问题的熵损失，熵损失计算出的值是softmax(概率），即用该损失函数训练的神经网络输出的计算结果为概率
#                                                       #如果有一个三分类问题，输出结果为[0.1,0.2,0.7],则认为这个数据点被分为第一个类的概率为10%，第二个类的概率为20%，第三个类的概率为70%
#                                                       #交叉熵就是计算分类问题输出结果的标签误差 标签即为[0,1,1]
#
# for t in range(100):                     #我们训练100步试试看
#      out=net(x)                   #这不算真正的概率，这边的值可能为[-2,-12,20],我们用softmax来转换成概率，即F.softmax(out) 转换为[0.1,0.2,0.7] 概率才是真正的prediction
#
#      loss=loss_func(out,y)        #计算预测值和真实值的误差
#      optimizer.zero_grad()               #把所有神经网络（net）中神经元的梯度设为0
#      loss.backward()                     #用误差反向传递，告诉每一个神经网络中的节点（我觉得应该是神经元）要有多少grad
#      optimizer.step()                    #以各个节点上的grad施加到神经网络参数上去，这就是整个优化的过程
#      if t % 2==0:                          #每五步我们就打印一次图上的信息
#          plt.cla()
#          prediction = torch.max(F.softmax(out), 1)[1]  #问你选择哪一类时，就输出类的概率的最大值，而最大值的概率的位置就是索引为1的地方，prediction就代表你的索引在的那个位置是哪个位置
#          pred_y = prediction.data.numpy()
#          target_y = y.data.numpy()
#          plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
#          accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)
#          plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color': 'red'})
#          plt.pause(0.1)
#
# plt.ioff()
# plt.show()