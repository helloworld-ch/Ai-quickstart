"""
配置中心
"""

batch_size = 32
images_frames = 4 # 喂养数据通道个数 ；；固定
actions_size = 4 # 动作个数
learning_rate = 0.01 # 学习率
epsilon = 0.9 # greedy policy preb
gamma = 0.9 # 折扣率
target_replace_iter = 100 # 目标网络更新频率
memory_capacity = 2000 # 记忆池大小
save_dir = '/saved_models'
model_path = None
echop = 3
