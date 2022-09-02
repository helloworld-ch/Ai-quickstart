import numpy as np
import random

class Game():

    def __init__(self):
        self.actions_label = {0:"向上",1:"向下",2:"向左",3:"向右"}
        self.score = 0
        self.bgs = {0: "#FFFFF0", 2: "#eee4da", 4: "#ede0c8", 8: "#f2b179", 16: "#f59563", 32: "#f67c5f", 64: "#f65e3b",
               128: "#edcf72", 256: "#edcc61",
               512: "#edc850", 1024: "#edc53f", 2048: "#edc22e", 4096: "#1d0220"}  # 颜色的16进制值
        self.mp = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]  # 当做二维数组来记录4*4方格的变化
        self.vis = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]  # 用来标记4*4方格都有哪几个位置已经有值
        self.newmp = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]  # 作为中介数组来使用
        self.vc = [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]  # 判断是否每一个位置都有数据
        self.random_num()

    def reset(self):
        self.flag = False
        self.score = 0
        self.bgs = {0: "#FFFFF0", 2: "#eee4da", 4: "#ede0c8", 8: "#f2b179", 16: "#f59563", 32: "#f67c5f", 64: "#f65e3b",
                    128: "#edcf72", 256: "#edcc61",
                    512: "#edc850", 1024: "#edc53f", 2048: "#edc22e", 4096: "#1d0220"}  # 颜色的16进制值
        self.mp = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]  # 当做二维数组来记录4*4方格的变化
        self.vis = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]  # 用来标记4*4方格都有哪几个位置已经有值
        self.newmp = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]  # 作为中介数组来使用
        self.vc = [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]  # 判断是否每一个位置都有数据
        self.random_num()
        temp_mp = np.copy(np.log2(np.array(self.mp)+1) / 16)
        # return np.array(temp_mp) # 线性
        return np.stack([temp_mp, temp_mp, temp_mp, temp_mp,
                         temp_mp, temp_mp, temp_mp, temp_mp,
                         temp_mp, temp_mp, temp_mp, temp_mp,
                         temp_mp, temp_mp, temp_mp, temp_mp]) # 卷积

    def show(self):
        print(np.array(self.mp))

    def init(self):
        self.vis = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        self.newmp = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    def init_mp(self):
        self.mp = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        
    def random_num(self):
        x1 = random.randint(0, 3)  # 整型数据范围0-3
        y1 = random.randint(0, 3)
        if self.vis[x1][y1] == 0:
            self.mp[x1][y1] = random.choice([2, 4, 2, 2])  # 随机在2和4之间取值
            self.vis[x1][y1] = 1

    def gameover(self):
        if self.vis == self.vc and self.panduan() == False:
            return 0
        else:
            return 1

    def panduan (self):
        movex=[-1,1,0,0]
        movey=[0,0,-1,1]
        for i in range(4):  #判断每一位元素的周围
            for j in range(4):
                for l in range(4):
                    newx=int(i+movex[l])
                    newy=int(j+movey[l])
                    if (newx<0 or newx>3)or(newy<0 or newy>3):
                        continue
                    else:
                        if self.mp[i][j]==self.mp[newx][newy]:
                            return True
        return False

    def put_up(self):
        self.init()  #初始化，newmp，vis
        for i in range(4):  #向上合并，去除为0的格子
            l=0
            for j in range(4):
                if self.mp[j][i]==0:
                    continue
                else:
                    self.newmp[l][i]=self.mp[j][i]
                    l+=1
        for i in range(4):  #从第二个开始只和它前一个数比较，如果相等则加上去并让这个位置等于0
            for j in range(1,4):
                if self.newmp[j][i]==0:
                    break
                else:
                    if self.newmp[j][i]==self.newmp[j-1][i]:
                        self.flag =True
                        self.score+=self.newmp[j][i]+self.newmp[j-1][i]
                        self.newmp[j-1][i]=self.newmp[j][i]+self.newmp[j-1][i]
                        self.newmp[j][i]=0
        if self.newmp==self.mp:  #如果向上合并后，和和相邻位置加后和之前未修改的一样则说明该方向无法操作直接跳出
            return
        self.init_mp()  #初始化mp把加后的值再次向上合并然后传给mp
        for i in range(4):
            l=0
            for j in range(4):
                if self.newmp[j][i]==0:
                    continue
                else:
                    self.mp[l][i]=self.newmp[j][i]
                    self.vis[l][i]=1
                    l+=1
        self.random_num()  #合并以后再空位随机产生一个2或
        return
    #向下
    def put_down(self):
        self.init()
        for i in range(4):
            l=3
            j=3
            while j>=0:
                if self.mp[j][i]==0:
                    j-=1
                    continue
                else:
                    self.newmp[l][i]=self.mp[j][i]
                    l-=1
                    j-=1
        for i in range(4):
            j=2
            while j>=0:
                if self.newmp[j][i]==0:
                    break
                else:
                    if self.newmp[j][i]==self.newmp[j+1][i]:
                        self.flag = True
                        self.score += self.newmp[j][i] + self.newmp[j + 1][i]
                        self.newmp[j+1][i]=self.newmp[j][i]+self.newmp[j+1][i]
                        self.newmp[j][i]=0
                j-=1
        if self.newmp==self.mp:
            return
        self.init_mp()
        for i in range(4):
            l=3
            j=3
            while j>=0:
                if self.newmp[j][i]==0:
                    j-=1
                    continue
                else:
                    self.mp[l][i]=self.newmp[j][i]
                    self.vis[l][i]=1
                    l-=1
                j-=1
        self.random_num()
        return
    #向左
    def put_left(self):
        self.init()
        for i in range(4):
            l=0
            for j in range(4):
                if self.mp[i][j]==0:
                    continue
                else:
                    self.newmp[i][l]=self.mp[i][j]
                    l+=1
        for i in range(4):
            for j in range(1,4):
                if self.newmp[i][j]==0:
                    break
                else:
                    if self.newmp[i][j]==self.newmp[i][j-1]:
                        self.flag = True
                        self.score += self.newmp[i][j] + self.newmp[i][j-1]
                        self.newmp[i][j-1]=self.newmp[i][j]+self.newmp[i][j-1]
                        self.newmp[i][j]=0
        if self.newmp==self.mp:
            return
        self.init_mp()
        for i in range(4):
            l=0
            for j in range(4):
                if self.newmp[i][j]==0:
                    continue
                else:
                    self.mp[i][l]=self.newmp[i][j]
                    self.vis[i][l]=1
                    l+=1
        self.random_num()
        return
    #向右
    def put_right(self):
        self.init()
        for i in range(4):
            l=3
            j=3
            while j>=0:
                if self.mp[i][j]==0:
                    j-=1
                    continue
                else:
                    self.newmp[i][l]=self.mp[i][j]
                    l-=1
                    j-=1
        for i in range(4):
            j=2
            while j>=0:
                if self.newmp[i][j]==0:
                    break
                else:
                    if self.newmp[i][j]==self.newmp[i][j+1]:
                        self.flag = True
                        self.score += self.newmp[i][j] + self.newmp[i][j + 1]
                        self.newmp[i][j+1]=self.newmp[i][j]+self.newmp[i][j+1]
                        self.newmp[i][j]=0
                j-=1
        if self.newmp==self.mp:
            return
        self.init_mp()
        for i in range(4):
            l=3
            j=3
            while j>=0:
                if self.newmp[i][j]==0:
                    j-=1
                    continue
                else:
                    self.mp[i][l]=self.newmp[i][j]
                    self.vis[i][l]=1
                    l-=1
                j-=1
        self.random_num()
        return

    def playgame(self):
        self.show()
        while(self.gameover()!= 0):
            action = int(input())
            print(self.actions_label[action])
            score_tem = self.score
            if(action == 0):
                self.put_up()
            elif(action == 1):
                self.put_down()
            elif(action == 2):
                self.put_left()
            else:
                self.put_right()
            self.show()
            print(self.score)
            print(self.score-score_tem)


    def nextFrame(self,action):
        score_tem = self.score
        info = self.actions_label[action]
        if action == 0:
            self.put_up()
        elif action == 1:
            self.put_down()
        elif action == 2:
            self.put_left()
        else:
            self.put_right()
        temp_mp = np.copy(np.log2(np.array(self.mp)+1) / 16)
        # s = np.array(temp_mp) # 线性
        s = np.stack([temp_mp, temp_mp, temp_mp, temp_mp,
                         temp_mp, temp_mp, temp_mp, temp_mp,
                         temp_mp, temp_mp, temp_mp, temp_mp,
                         temp_mp, temp_mp, temp_mp, temp_mp])  # 卷积
        r = self.score-score_tem
        done = self.gameover()
        # if done == 0:
        #     r = -100
        if self.flag:
            r = r
        else:
            r = 0
        self.flag = False
        return s,r,done,info

if __name__ == '__main__':
    game = Game()
    game.playgame()