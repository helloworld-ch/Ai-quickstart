from lib import ws,max_len,test_batch_size
from torch import nn
from torch.nn import functional as F
import torch
from torch.optim import Adam
from datasets import get_data_loader
import os
import numpy as np
from tqdm import tqdm

list_len = 100
labels_len = 2
learning_rate = 1e-3

class ImdbModel(nn.Module):
    def __init__(self):
        super(ImdbModel, self).__init__()
        self.embedding = nn.Embedding(len(ws),list_len)
        self.fc1 = nn.Linear(max_len*list_len,labels_len)

    def forward(self,inputs):
        x = self.embedding(inputs)
        x = torch.reshape(x,[-1,max_len*list_len])
        x = self.fc1(x)

        return F.log_softmax(x,dim = -1)

network = ImdbModel()
optimizer = Adam(network.parameters(),lr = learning_rate)
# 加载模型
if os.path.exists("./models/ImdbNet.pkl"):
    network.load_state_dict(torch.load("./models/ImdbNet.pkl"))
    optimizer.load_state_dict(torch.load("./models/ImdbNetOptimizer.pkl"))
    print("加载成功")

def train(echop):
    for i in range(echop):
        # 准备数据集
        dataloader = get_data_loader()

        for step,(x,y) in enumerate(tqdm(dataloader)):
            optimizer.zero_grad()
            predict = network(x)
            loss = F.nll_loss(predict,y)
            loss.backward()
            optimizer.step()

            # if step% 100 == 0:
            #     print(loss.item())

            if step%100 == 0 or step == len(dataloader)-1:
                torch.save(network.state_dict(),"./models/ImdbNet.pkl")
                torch.save(optimizer.state_dict(),"./models/ImdbNetOptimizer.pkl")

def test():
    # 准备数据集
    dataloader = get_data_loader(False,test_batch_size)
    loss_list = []
    acc_list = []
    network.eval()
    for step,(x,y) in enumerate(tqdm(dataloader)):
        with torch.no_grad():
            predict = network(x)
            loss = F.nll_loss(predict,y)
            loss_list.append(loss)
            correct = torch.argmax(predict,dim=1)
            acc = y.eq(correct).float().mean()
            acc_list.append(acc)
    loss = np.mean(loss_list)
    acc = np.mean(acc_list)
    print("loss : ", loss, "acc : ", acc)


if __name__ == '__main__':
    # for i in range(10):
    #     train(i)

    test()