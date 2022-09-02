import torch
from torch import nn
import numpy as np


# 搭建网络结构
class NetWork(nn.Module):

    def __init__(self):
        super(NetWork, self).__init__()
        self.conv1 = nn.Conv2d(16,64,kernel_size=(3,3),stride=(1,1),padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.fc1 = nn.Linear(64*4*4,4)

        self.conv2 = nn.Conv2d(16,64,kernel_size=(3,3),stride=(1,1),padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc2 = nn.Linear(64*4*4,4)

    def forward(self,inputs,training = None):
        inputs = inputs.reshape([-1,16,4,4])
        x1 = self.bn1(self.conv1(inputs))
        x1 = x1.reshape([-1,64*4*4])
        x1 = self.fc1(x1)
        x2 = self.bn2(self.conv2(inputs))
        x2 = x2.reshape([-1, 64 * 4 * 4])
        x2 = self.fc2(x2)
        x = x1+x2
        print(x1)
        print(x2)
        print(x)
        out = x - torch.transpose(x1.max(1)[0],0,1)
        print(x1.max(1)[0])
        return out

    def init_weight(self):
        pass

# 搭建运行体
class Agent:
    def __init__(self):
        pass
    def choose_action(self,state):
        pass
    def store_memory(self):
        pass
    def learn(self):
        pass
    def train(self):
        pass
    def test(self):
        pass
    def predict(self,state):
        pass

if __name__ == '__main__':
    net = NetWork()
    print(net(torch.randn([2, 16, 4, 4])))