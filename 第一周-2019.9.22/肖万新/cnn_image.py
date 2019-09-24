import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
# import argparse
import torch.nn.functional as F
# 定义是否使用GPU
device = torch.device("cpu")
# 定义网络结构
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Sequential(     #input_size=(1*28*28)
            # 输入通道1，输出通道6（有6个卷积核），卷积核尺寸5*5，步长1，padding为2
            nn.Conv2d(1, 6, 5, 1, 2), #padding=2保证输入输出尺寸相同
            nn.ReLU(),      #input_size=(6*28*28)
            # 池化层
            nn.MaxPool2d(kernel_size=2, stride=2),#output_size=(6*14*14)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),      #input_size=(16*10*10)
            nn.MaxPool2d(2, 2)  #output_size=(16*5*5)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU()
        )
        # 输出即为各个数字的概率
        self.fc3 = nn.Linear(84, 10)
        

    # 定义前向传播过程，输入为x
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # nn.Linear()的输入输出都是维度为一的值，所以要把多维度的tensor展平成一维
        # 此时x是64*5*5*16的张量，需要将它展开成64行，每行代表一张图片，每一行经过全连接层后
        # 就得到这张图片对应各个数字的概率
        # view可以改变tensor的尺寸
        x = x.view(x.size(0), -1)   
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x


# 超参数设置
EPOCH = 4   #遍历数据集次数
BATCH_SIZE = 64      #批处理尺寸(batch_size)
LR = 0.001        #学习率

# 定义数据预处理方式，转为tensor类型
transform = transforms.ToTensor()

# 用torchvision下载训练数据集
trainset = tv.datasets.MNIST(
    root='./data/',
    train=True,
    download=True,
    transform=transform)

# 定义训练批处理数据
trainloader = torch.utils.data.DataLoader(
    trainset,                 #输入的数据类型
    batch_size=BATCH_SIZE,    #定义每次喂给神经网络多少行数据
    shuffle=True,             #每次迭代都将数据打乱
    )

# 用torchvision下载测试数据集
testset = tv.datasets.MNIST(
    root='./data/',
    train=False,
    download=True,
    transform=transform)

# 定义测试批处理数据
testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    )

# 定义损失函数loss function 和优化方式（采用SGD）
net = LeNet().to(device)           # 将所有最开始读取数据时的tensor变量copy一份到device所指定的CPU上去，之后的运算都在CPU上进行。
criterion = nn.functional.nll_loss  # 交叉熵损失函数，通常用于多分类问题上
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9)    #优化方法用SGD
#
# 训练
if __name__ == "__main__":
    net.train()
    for epoch in range(EPOCH):
        sum_loss = 0.0
        # 数据读取
        # 将trainloader组合为一个索引序列，同时列出数据和数据下标，其中i是元素的索引，data是元素
        for i, data in enumerate(trainloader):
            # data是一个含有两个张量的列表
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # 梯度清零，不清零的话，再次循环式时会加上现在的值，导致梯度越来越大
            optimizer.zero_grad()    

            # forward + backward
            outputs = net(inputs)      # 输入图片，得出结果

            # 输出的outputs和原来导入的labels作为loss函数的输入就可以得到损失
            loss = criterion(outputs, labels)
            loss.backward()
            #只有用了optimizer.step()，模型才会更新
            optimizer.step()


            # 每训练100个batch打印一次平均loss
            sum_loss += loss.item()
            if i % 100 == 99:
                print('[%d, %d] loss: %.03f'
                      % (epoch + 1, i + 1, sum_loss / 100))
                sum_loss = 0.0
        # 每跑完一次epoch测试一下准确率
        with torch.no_grad():
            correct = 0
            total = 0
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                # 取得分最高的那个类
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()
            print('第%d个epoch的识别准确率为：%d%%' % (epoch + 1, (100 * correct / total)))
    #输出参数
    for param_tensor in net.state_dict():
        print(param_tensor, "\t", net.state_dict()[param_tensor].size())

    #保存参数
    torch.save(net.state_dict(), 'params.pkl')
    #保存整个模型
    torch.save(net,'model.pth')

"""
总结：
1.定义网络结构
2.设置超参数：遍历数据集次数，批处理尺寸，学习率
3.定义损失函数
4.在main函数中写训练代码，一般为一个for循环

"""
