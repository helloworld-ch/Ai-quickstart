import pygame
import numpy as np
import DQN玩游戏.config as config
import random

class Game():
    def __init__(self):
        pass

    def reset(self):
        s = np.random.uniform(0, 255, [80, 80])
        state_begin = np.stack([s, s, s, s])
        return state_begin

    def nextFrame(self,action):
        """
        根据动作得到下一状态
        :param action: 动作
        :return: 下一个状态，奖励，是否结束，信息
        done :1表示还在继续
        done :0表示游戏结束
        """
        r = random.uniform(1,4)
        next_s = np.random.uniform(0,255,[80,80])
        next_s = np.stack([next_s, next_s, next_s, next_s])
        done = random.randint(1,1000)
        if done == 3:
            done = 0
        else:
            done = 1#
        info = "状态更新"
        return next_s,r,done,info

    def show_image(self):
        print("t游戏时刻状态图")

if __name__ == '__main__':
    game = Game()
    print(len(game.nextFrame(None)))
