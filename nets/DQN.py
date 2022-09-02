import torch
from tqdm import tqdm
import numpy as np
from torch import nn
import DQN玩游戏.config as config
from collections import deque
import os
from DQN玩游戏.gameApi import game

#搭建DQN网络模型
class DQNet(nn.Module):

    def __init__(self):
        super(DQNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=config.images_frames,out_channels=32,kernel_size=(8,8),stride=(4,4),padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.max_pool2x2_1 = nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))

        self.conv2 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=(4,4),stride=(2,2),padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.max_pool2x2_2 = nn.MaxPool2d(kernel_size=(2,2),stride=(2,2),padding=1)

        self.conv3 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=(3,3),stride=(1,1),padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.max_pool2x2_3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=1)

        self.fc1 = nn.Linear(256,128)
        self.fc2 = nn.Linear(128,config.actions_size)

        self.__init_weights()
    def forward(self,inputs,training=True):
        # 添加一个batch，修正shape[4,80,80] => [batch,chanels,h,w]
        inputs = np.reshape(inputs,[-1,4,80,80])

        x = self.conv1(inputs)
        x = self.bn1(x)
        x = torch.relu(self.max_pool2x2_1(x))

        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(self.max_pool2x2_2(x))

        x = self.conv3(x)
        x = self.bn3(x)
        x = torch.relu(self.max_pool2x2_3(x))

        x = x.reshape([-1,256]) # 拉平
        x = torch.relu(self.fc1(x))
        out = self.fc2(x)
        return out
    def __init_weights(self):
        nn.init.trunc_normal_(self.conv1.weight,std=0.01)
        nn.init.trunc_normal_(self.conv2.weight,std=0.01)
        nn.init.trunc_normal_(self.conv3.weight, std=0.01)
        nn.init.trunc_normal_(self.bn1.weight, std=0.01)
        nn.init.trunc_normal_(self.bn2.weight, std=0.01)
        nn.init.trunc_normal_(self.bn3.weight, std=0.01)

        nn.init.constant_(self.conv1.bias,0.01)
        nn.init.constant_(self.conv2.bias,0.01)
        nn.init.constant_(self.conv3.bias, 0.01)
        nn.init.constant_(self.bn1.bias, 0.01)
        nn.init.constant_(self.bn2.bias, 0.01)
        nn.init.constant_(self.bn3.bias, 0.01)

        nn.init.trunc_normal_(self.fc1.weight,std=0.01)
        nn.init.trunc_normal_(self.fc1.weight, std=0.01)
        nn.init.constant_(self.fc1.bias, 0.01)
        nn.init.constant_(self.fc1.bias, 0.01)

# 搭建dqnAgent
class DQNAgent():

    def __init__(self,game,network:nn.Module,eval_network:nn.Module,config,**kwargs):
        self.game = game
        self.config = config
        self.game_memories = deque()

        self.network = network
        self.eval_net = eval_network
        self.loss_fc = nn.MSELoss()
        self.optimizer = torch.optim.Adam(eval_network.parameters(),lr=self.config.learning_rate)

    def store_transition(self,memory_ceil):
        '''
        管理记忆库
        :param memory_ceil: 记忆细胞 （s,a,r,next_s）
        :return:
        '''
        if len(self.game_memories)>self.config.memory_capacity:
            self.game_memories.popleft()
        self.game_memories.append(memory_ceil)

    # 根据网络或者探索策略选择action
    def choose_action(self,state):
        if state is None:
            return np.random.randint(0,config.actions_size)
        # 前面加一个batch维度
        state = torch.unsqueeze(torch.FloatTensor(state),0)
        if np.random.uniform()<self.config.epsilon:
            actions_value = self.network.forward(state)
            action = torch.argmax(actions_value,dim=1).numpy()[0]
        else:
            action = np.random.randint(0,config.actions_size)
        return action

    def learn(self,step):
        # 目标网络更新
        if step % self.config.target_replace_iter == 0:
            self.network.load_state_dict(self.eval_net.state_dict())

        # 取样记忆库
        samples = np.random.choice(np.arange(len(self.game_memories)), self.config.batch_size)
        # 分别记录s,r,a,next_s
        s_list = []
        r_list = []
        a_list = []
        next_s_list = []
        for i in samples:
            s_list.append(self.game_memories[i][0])
            a_list.append(self.game_memories[i][1])
            r_list.append(self.game_memories[i][2])
            next_s_list.append(self.game_memories[i][3])
        s_list = torch.FloatTensor(np.array(s_list))
        a_list = torch.LongTensor(np.array(a_list).reshape([-1,1]))
        r_list = torch.FloatTensor(np.array(r_list).reshape([-1,1]))
        next_s_list = torch.FloatTensor(np.array(next_s_list))

        # 让评论网络来做出估计
        q_value = self.eval_net(s_list).gather(1,a_list)

        q_next = self.network(next_s_list).detach()
        q_target = r_list + self.config.gamma*q_next.max(1)[0].reshape([self.config.batch_size,1])

        self.optimizer.zero_grad()
        loss = self.loss_fc(q_value, q_target)# 输入32个评估值和32个目标值，使用均方损失函数
        loss.backward()  # 误差反向传播, 计算参数更新值
        self.optimizer.step()

    # 训练
    def train(self):
        if not os.path.exists(self.config.save_dir):
            os.mkdir(self.config.save_dir)
        # if self.config.model_path is not None:
        #     network.load_state_dict(network.state_dict(),torch.load(self.config.save_dir))

        for step in tqdm(range(self.config.echop)):
            print("--------第",step+1,"轮-----------")
            Reward = 0
            s = self.game.reset()
            while(True):
                # 反馈结果
                # self.game.show_image()

                # 选择动作
                action = self.choose_action(s)
                # 得到下一帧状态
                next_s,r,done,info = self.game.nextFrame(action)
                # 存储样本
                self.store_transition([s,action,r,next_s])
                Reward+=r
                # 状态更新
                s = next_s

                if len(self.game_memories)>self.config.memory_capacity:
                    # 容量足够后可以开始训练
                    self.learn(step)

                if done == 0:
                    # 游戏结束
                    print("Reward :",Reward)
                    break
    # 测试
    def test(self):
        self.train()
        pass

if __name__ == '__main__':
    network = DQNAgent(game.Game(),DQNet(),DQNet(),config)
    network.test()

