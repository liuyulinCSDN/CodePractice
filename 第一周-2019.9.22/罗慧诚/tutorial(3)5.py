import torch
import torch.utils.data as Data  #用于数据库的样本取量

torch.manual_seed(1)

BATCH_SIZE=5

x = torch.linspace(1, 10, 10)       # 1,2,3,4,5,6,7,8,9,10
y = torch.linspace(10, 1, 10)       # 10,9,8,7,6,5,4,3,2,1

torch_dataset=Data.TensorDataset(x,y)   #定义一个dataset数据库.拿数据进行训练的就是data-tensor,拿数据进行算误差的就是target-tensor
loader=Data.DataLoader(
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,                 #训练数据时要不要随机打乱数据，再进行抽样，false就是不打乱数据进行训练
    num_workers=2,                #每一次提取数据时是用线程还是进程来提取的

)         #loader就是让我们训练的数据变成一小批一小批的
def show_batch():
    for epoch in range(3):              #拿整体数据(即10个数据)训练3次
        for step,(batch_x,batch_y) in enumerate(loader):    #每一次训练数据，我们把10个数据数据拆成两个数据进行训练，训练三次。每次训练时，loader可以决定要不要打乱数据进行训练。enumerate就是说每次loader提取时都给loader增加一个索引。比如第一个loader提取时设为loader==1,第二次loader==2.诸如此类
        #training
            print('Epoch: ', epoch, '| Step: ', step, '| batch x: ',
                 batch_x.numpy(), '| batch y: ', batch_y.numpy())

if __name__ == '__main__':
    show_batch()