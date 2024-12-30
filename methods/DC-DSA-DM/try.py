import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Parameters and DataLoaders
# y(data_size) = X(data_size,2)xW(2,1) + b(标量)
# input_size = 2:代表设计矩阵 X 有2个特征
# output_size = 1:代表标签y 只有1个数据
input_size = 2
output_size = 1
batch_size = 200
data_size = 200

# 优先使用GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Dummy DataSet
# 制作一个随机数数据集。你只需要实现getitem函数
# @paras:    length:数据集的长度(数据点的个数)
#            n_features:数据的维度
class RandomDataset(Dataset):
    def __init__(self, length,n_features):
        self.len = length
        self.weight = torch.tensor([[5.2],[9.6]])
        self.bias = torch.tensor(3.4)
        self.data = torch.randn(length, n_features)
        self.targets = torch.matmul(self.data,self.weight) + self.bias
    def __getitem__(self, index):
        return self.data[index],self.targets[index]
    def __len__(self):
        return self.len

rand_loader = DataLoader(dataset=RandomDataset(data_size, 2),
                         batch_size=batch_size, shuffle=True)

# `DataParallel`可以用在任何模型上。
# 模型中的print语句将打印输入tensor和输出tensor的size.
# 注意batch rank0会打印什么

class Model(nn.Module):
    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, input):
        output = self.fc(input)
        print("In Model: input size", input.size(),"output size", output.size())
        return output

# 生成一个model实例:检测是否有多个GPU

model = Model(input_size, output_size)
if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
  model = nn.DataParallel(model)

model.to(device)

# 3 loss 

# 4 optimizer
optimizer = optim.Adam(model.parameters(),lr=0.5)

def Train():
    for epoch in range(200):
        for (data,labels) in rand_loader:
            # forward
            input = data.to(device)
            targets = labels.to(device)
            print(input.device, targets.device, model.device_ids)
            output = model(input)
            #loss = sum((output-targets)*(output-targets))/batch_size
            loss = F.mse_loss(output,targets)
            #if epoch%(20) == 0:
            print("Outside: input size", input.size(),"output_size", output.size(),'loss:{}'.format(loss.item()))
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

Train()
print(list(model.parameters()))