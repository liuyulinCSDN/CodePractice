import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.utils.data as Data
#各类损失函数
# tensor=torch.FloatTensor([[1,2],[3,4]])
# variable=Variable(tensor,True)
# print(tensor)
# print(variable)
# v_out=torch.mean(variable*variable)
# print(v_out)
# v_out.backward()
# print(v_out)
# print(variable.grad)
# x=torch.linspace(5,-5,200)
# #print(x)
# x=Variable(x)
# x_np=x.data.numpy()
#print(x_np)
# y_relu=F.relu(x).data.numpy()
# # print(y_relu)
# y_sigmoid=F.sigmoid(x).data.numpy()
# y_tanh=F.tanh(x).data.numpy()
# y_softplus=F.softplus(x).data.numpy()
# plt.figure("exaple",(8,6))
# plt.subplot(221)
# plt.plot(x_np,y_relu,c='red',label='relu')
# plt.ylim((-1,5))
# plt.legend('best')
# plt.subplot(222)
# plt.plot(x_np,y_sigmoid,c='red',label='sigmoid')
# plt.ylim((-0.2,1.2))
# plt.legend(loc='best')
# plt.subplot(223)
# plt.plot(x_np,y_softplus,c='red',label='softplus')
# plt.ylim((-0.2,6))
# plt.legend('best')
# plt.subplot(224)
# plt.plot(x_np,y_tanh,c='red',label='tanh')
# plt.ylim(-1.2,1.2)
# plt.legend('best')
# plt.show()
# print(torch.linspace(-1,1,100))
    # print(torch.linspace(-1,1,100))
#回归数据神经网络训练
# x=torch.unsqueeze(torch.linspace(-1,1,100),dim=1)
# y=x.pow(2)+0.2*torch.rand(x.size())
# x=Variable(x)
# y=Variable(y)
# class Net(torch.nn.Module):
#     def __init__(self,n_feature,n_hidden,n_output):
#         super(Net, self).__init__()
#         self.hidden=torch.nn.Linear(n_feature,n_hidden)
#         self.predict=torch.nn.Linear(n_hidden,n_output)
#     def foward(self,x):
#         x=F.relu(self.hidden(x))
#         x=self.predict(x)
#         return x
# net=Net(n_feature=1,n_hidden=10,n_output=1)
# print(net)
# optimizer=torch.optim.SGD(net.parameters(),lr=0.5)
# loss_func=torch.nn.MSELoss()
# for i in range(100):
#     plt.ion()
#     plt.show()
#     prediction=net.foward(x)
#     loss=loss_func(prediction,y)
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#     if i%5==0:
#         plt.cla()
#         plt.scatter(x.data.numpy(),y.data.numpy())
#         plt.plot(x.data.numpy(),prediction.data.numpy(),'r-',lw=5)
#         plt.text(0.5,0.5,'loss=%.4f'%(loss.data.numpy()),fontdict={'color':'red','size':20})
#         plt.pause(0.1)
#分类数据神经网络训练
# n_data=torch.ones(100,2)
# x0=torch.normal(2*n_data,1)
# y0=torch.zeros(100)
# x1=torch.normal(-2*n_data,1)
# y1=torch.ones(100)
# x=torch.cat((x0, x1), 0).type(torch.FloatTensor)
# y=torch.cat((y0, y1), ).type(torch.LongTensor)
# print(y.shape)
# plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy(), s=100, lw=0, cmap='RdYlGn')
# plt.show()
# class Net(torch.nn.Module):
#     def __init__(self,n_feature,n_hidden,n_output):
#         super(Net, self).__init__()
#         self.hidden=torch.nn.Linear(n_feature,n_hidden)
#         self.output=torch.nn.Linear(n_hidden,n_output)
#     def foward(self,x):
#         x=F.relu(self.hidden(x))
#         x=self.output(x)
#         return x
# net=Net(n_feature=2,n_hidden=10,n_output=2)
# optimizer=torch.optim.SGD(net.parameters(),lr=0.002)
# loss_func=torch.nn.CrossEntropyLoss()
# for i in range(100):
#     plt.ion()
#     plt.show()
#     out=net.foward(x)
#     loss=loss_func(out,y)
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#     if i%1==0:
#         plt.cla()
#         prediction=torch.max(F.softmax(out),1)[1]
#         pred_y = prediction.data.numpy()
#         target_y = y.data.numpy()
#         plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
#         accuracy = sum(pred_y == target_y)/200.
#         plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color':  'red'})
#         plt.pause(0.01)
#
# plt.ioff()
# plt.show()
# 快速建立
# test=torch.nn.Sequential(torch.nn.Linear(1,10),torch.nn.ReLU(),torch.nn.Linear(10,1),)
#存储
# x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
# y = x.pow(2) + 0.2*torch.rand(x.size())
# def save():
#     net=torch.nn.Sequential(torch.nn.Linear(1,10),
#                             torch.nn.ReLU(),
#                             torch.nn.Linear(10,1))
#     optimizer=torch.optim.SGD(net.parameters(),lr=0.5)
#     loss1=torch.nn.MSELoss()
#     for i in range(100):
#         prediction=net(x)
#         loss2=loss1(prediction,y)
#         optimizer.zero_grad()
#         loss2.backward()
#         optimizer.step()
#     torch.save(net,'net.pkl')
#     torch.save(net.state_dict(), 'net_params.pkl')
#     plt.figure(1,figsize=(20,3))
#     plt.subplot(131)
#     plt.title("net1")
#     plt.scatter(x.data.numpy(),y.data.numpy())
#     plt.plot(x.data.numpy(),prediction.data.numpy(),'r-',lw=5)
#     plt.show()
# def restore_net():
#     net2=torch.load('net.pkl')
#     prediction=net2(x)
#     plt.subplot(132)
#     plt.title("net2")
#     plt.scatter(x.data.numpy(),y.data.numpy())
#     plt.plot(x.data.numpy(),prediction.data.numpy(),'r-',lw=5)
#     plt.show()
# def restore_params():
#     net3 = torch.nn.Sequential(
#         torch.nn.Linear(1, 10),
#         torch.nn.ReLU(),
#         torch.nn.Linear(10, 1)
#     )
#     net3.load_state_dict(torch.load('net_params.pkl'))
#     prediction = net3(x)
#     prediction=net3(x)
#     plt.subplot(133)
#     plt.title("net3")
#     plt.scatter(x.data.numpy(),y.data.numpy())
#     plt.plot(x.data.numpy(),prediction.data.numpy(),'r-',lw=5)
#     plt.show()
# save()
# restore_net()
# restore_params()
#批处理
# BATCH_SIZE=5
# x=torch.linspace(1,10,10)
# y=torch.linspace(10,1,10)
# torch_dataset=Data.TensorDataset(x,y)
# loader=Data.DataLoader(dataset=torch_dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=2)
# for epoch in range(3):
#     if __name__ == '__main__':
#         for step,(batch_x,batch_y) in enumerate(loader):
#             net=torch.nn.Sequential(torch.nn.Linear(1,10),
#                                         torch.nn.ReLU(),
#                                         torch.nn.Linear(10,1))
#             optimizer=torch.optim.SGD(net.parameters(),lr=0.2)
#             loss_func=torch.nn.MSELoss()
#         print('Epoch: ', epoch, '| Step: ', step, '| batch x: ',
#               batch_x.numpy(), '| batch y: ', batch_y.numpy())
#
#优化器
# LR = 0.01
# BATCH_SIZE = 32
# EPOCH = 12
# x = torch.unsqueeze(torch.linspace(-1, 1, 1000), dim=1)
# y = x.pow(2) + 0.1*torch.normal(torch.zeros(*x.size()))
# plt.scatter(x.data.numpy(),y.data.numpy())
# plt.show()
# torch_dataset=Data.TensorDataset(x,y)
# torch_load=Data.DataLoader(dataset=torch_dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=2)
# class Net(torch.nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.hidden=torch.nn.Linear(1,20)
#         self.predict=torch.nn.Linear(20,1)
#     def forward(self,x):
#         x=F.relu(self.hidden(x))
#         x=self.predict(x)
#         return x
# net_SGD         = Net()
# net_Momentum    = Net()
# net_RMSprop     = Net()
# net_Adam        = Net()
# nets = [net_SGD, net_Momentum, net_RMSprop, net_Adam]
# opt_SGD=torch.optim.SGD(net_SGD.parameters(),lr=LR)
# opt_Momentum=torch.optim.SGD(net_Momentum.parameters(),lr=LR,momentum=0.8)
# opt_RMSprop=torch.optim.RMSprop(net_RMSprop.parameters(),lr=LR,alpha=0.9)
# opt_Adam=torch.optim.Adam(net_Adam.parameters(),lr=LR,betas=(0.9,0.99))
# optimizers = [opt_SGD, opt_Momentum, opt_RMSprop, opt_Adam]
# loss_func=torch.nn.MSELoss()
# losses=[[],[],[],[]]
# for epoch in range(EPOCH):
#     print('EPOCH: ',epoch)
#     if __name__ == '__main__':#进行
#         for step,(b_x,b_y) in enumerate(torch_load):
#             # print(b_x,b_y)
#             for net,opt,l_his in zip(nets,optimizers,losses):
#                 output=net(b_x)
#                 loss=loss_func(output,b_y)
#                 opt.zero_grad()
#                 loss.backward()
#                 opt.step()
#                 l_his.append(loss.data.numpy())
#                 print(l_his)













