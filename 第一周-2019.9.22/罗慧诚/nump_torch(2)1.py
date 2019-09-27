import torch
from torch.autograd import Variable

tensor=torch.FloatTensor([[1,2],(3,4)])
Variable=Variable(tensor, requires_grad=True)

print(tensor)
print(Variable)

t_out=torch.mean(tensor*tensor)  #x^2
v_out=torch.mean(Variable*Variable)

print(t_out)
print(v_out)

v_out.backward()
#v_out=1/4*sum(var*var)
#d(v_out)/d(var)=1/4*2*Variable=Variable/2
print(Variable.grad)
print(Variable.data)
print(Variable.data.numpy())

