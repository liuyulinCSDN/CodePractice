import torch
import matplotlib.pyplot as plt

x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
y = x.pow(2) + 0.2*torch.rand(x.size())

def save():
    net1 = torch.nn.Sequential(
        torch.nn.Linear(1, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1)
    )
    optimizer = torch.optim.SGD(net1.parameters(), lr=0.5)
    loss_func = torch.nn.MSELoss()

    for t in range(100):
        prediction = net1(x)
        loss = loss_func(prediction, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # net1出图方式
    plt.figure(1, figsize=(10, 3))
    plt.subplot(131)
    plt.title('Net1')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)

    # 2 ways to save the net
    torch.save(net1,'net.pkl')  # 如何保存整个神经网络net1（用快速搭建法搭建），其中神经网络的名字被设为net.pkl
    torch.save(net1.state_dict(),'net_params.pkl')   #不保留整个神经网络(的计算图)，但保留神经网络（图）里的每个参数(比如神经元的个数，比如输出的个数),state_dictionary指神经网络现在的状态，状态包括比如神经元的个数，比如输出的个数


def restore_net():
    net2=torch.load('net.pkl')   #load为提取
    prediction = net2(x)

    # net2出图方式
    plt.subplot(132)
    plt.title('Net2')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)


def restore_params():
    # restore only the parameters in net1 to net3
    net3 = torch.nn.Sequential(
        torch.nn.Linear(1, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1)
    )    #Net3要提取参数，也要建立一套和net1一样的神经网络，但肯定参数和net1不一样

    net3.load_state_dict(torch.load('net_params.pkl'))  #提取net1中所有参数到net3中，'net_params.pkl'保存的是net1中的所有参数
    prediction = net3(x)

    # net3出图方式
    plt.subplot(133)
    plt.title('Net3')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
    plt.show()    #这个show()是直接打印出三个图像


save()           #保存神经网络数据

restore_net()    #提取神经网络数据方法1

restore_params()  #提取神经网络数据方法2