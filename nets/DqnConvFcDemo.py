import numpy as np
import torch
from torch import nn
import os
from collections import deque
import DQN玩游戏.config_2048 as config
from DQN玩游戏.gameApi.game2048 import Game

"""
搭建网络模型，这里直接采用4层线性层搭建
1.日本选手的7层卷积网络
【4，4，16】 = 》【4，4，256】
【4，4，256】 = 》【4，4，256】
【4，4，256】 = 》【4，4，256】
【4，4，256】 = 》【4，4，256】
【4，4，256】 = 》【4，4，256】
flatten => Linear
softmax(actions = 4)

"""


class NetWork(nn.Module):

    def __init__(self):
        super(NetWork, self).__init__()

        self.conv1 = nn.Conv2d(16, 64, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.fc1 = nn.Linear(64*4*4, 128)
        self.out = nn.Linear(128, config.actions_size)

    def forward(self, inputs, training=None):
        x = self.bn1(self.conv1(inputs))
        x = x.reshape([-1,64*4*4])
        x = torch.relu(self.fc1(x))
        out = self.out(x)
        return out

    def __init_weight(self):
        nn.init.trunc_normal_(self.conv1.weight,std=0.1)
        nn.init.trunc_normal_(self.fc1.weight, std=0.1)
        nn.init.trunc_normal_(self.out.weight, std=0.1)

        nn.init.trunc_normal_(self.conv1.bias,0.1)
        nn.init.trunc_normal_(self.fc1.bias, 0.1)
        nn.init.trunc_normal_(self.out.bias, 0.1)


class DQNAgent():

    def __init__(self, game, config, **kwargs):
        self.game = game
        self.config = config
        self.game_memory = deque()

        self.network = NetWork()
        self.eval_network = NetWork()
        self.loss_fc = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.eval_network.parameters(), lr=self.config.learning_rate)

    def choose_action(self, state=None, gamma=config.gamma):
        if state is None:
            action = np.random.randint(0, self.config.actions_size)
        else:
            # 前面加一个batch维度
            state = torch.unsqueeze(torch.FloatTensor(state), 0)
            if np.random.uniform() < gamma:
                action = self.eval_network.forward(state)
                action = torch.argmax(action, dim=1).numpy()[0]
            else:
                action = np.random.randint(0, self.config.actions_size)
        return action

    def store_memory(self, memory_ceil):
        if len(self.game_memory) > self.config.max_memory_size:
            self.game_memory.popleft()
        # memory_ceil[0] = np.log2(memory_ceil[0] + 1) / 16
        # memory_ceil[3] = np.log2(memory_ceil[3] + 1) / 16
        self.game_memory.append(memory_ceil)

    def learn(self, step):
        # 重置目标网络
        if step % self.config.target_update_iter == 0:
            self.network.load_state_dict(self.eval_network.state_dict())

        # 从记忆库中取batch 进行训练
        samples_indexs = np.random.choice(np.arange(len(self.game_memory)), self.config.batch_size)
        s_list = []
        a_list = []
        r_list = []
        next_s_list = []

        for i in samples_indexs:
            s_list.append(self.game_memory[i][0])
            a_list.append(self.game_memory[i][1])
            r_list.append(self.game_memory[i][2])
            next_s_list.append(self.game_memory[i][3])
        s_list = torch.FloatTensor(np.array(s_list))
        a_list = torch.LongTensor(np.array(a_list).reshape([-1, 1]))
        # r_list = torch.FloatTensor(np.log2(np.array(r_list).reshape([-1,1])+1))
        r_list = torch.FloatTensor(np.array(r_list).reshape([-1, 1]))
        next_s_list = torch.FloatTensor(np.array(next_s_list))

        q_value = self.eval_network(s_list).gather(1, a_list)

        q_next = self.network(next_s_list).detach()
        q_target = r_list + self.config.gamma * q_next.max(1)[0].reshape([self.config.batch_size, 1])

        self.optimizer.zero_grad()
        loss = self.loss_fc(q_value, q_target)
        loss.backward()  # 误差反向传播, 计算参数更新值
        self.optimizer.step()

    def train(self):
        if os.path.exists(self.config.save_dir + "/" + self.config.model_path):
            self.network.load_state_dict(self.network.state_dict(),
                                         torch.load(self.config.save_dir + "/" + self.config.model_path))
            self.eval_network.load_state_dict(self.eval_network.state_dict(),
                                              torch.load(self.config.save_dir + "/" + self.config.model_path))
            print("加载成功")

        for step in range(self.config.echop):
            print("--------第", step + 1, "轮-----------")
            Reward = 0
            s = self.game.reset()
            while (True):
                # 反馈结果
                # self.game.show()
                # print(len(self.game_memory))
                # print(Reward)

                # 选择动作
                action = self.choose_action(s)
                # 得到下一帧状态
                next_s, r, done, info = self.game.nextFrame(action)
                # 存储样本
                self.store_memory([s, action, r, next_s])
                Reward += r
                # 状态更新
                s = next_s

                if len(self.game_memory) > self.config.max_memory_size:
                    # 容量足够后可以开始训练
                    self.learn(step)

                if done == 0:
                    # 游戏结束
                    print()
                    self.game.show()
                    print()
                    print("Reward :", Reward)
                    break
        self.save_model()

    def save_model(self):
        if not os.path.exists(self.config.save_dir):
            os.mkdir(self.config.save_dir)
        torch.save(self.eval_network.state_dict(), self.config.save_dir + "/" + self.config.model_path)
        print("保存成功")

    def predict(self):
        if os.path.exists(self.config.save_dir + "/" + self.config.model_path):
            self.eval_network.load_state_dict(self.eval_network.state_dict(),
                                              torch.load(self.config.save_dir + "/" + self.config.model_path))
            print("加载成功")
        s = self.game.reset()
        Reward = 0
        while (True):
            a = self.choose_action(gamma=1)
            next_s, r, done, info = self.game.nextFrame(a)
            self.store_memory([s, a, r, next_s])
            s = next_s
            Reward += r
            if done == 0:
                # 游戏结束
                print()
                self.game.show()
                print()
                print("Reward :", Reward)
                break


if __name__ == '__main__':
    agent = DQNAgent(Game(), config)
    # agent.train()
    agent.predict()
    # net = NetWork()
    # net(torch.zeros([1,16,4,4]))


