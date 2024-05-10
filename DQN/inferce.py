import random
import torch
import collections
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import gymnasium as gym
import torch.nn.functional as F
import matplotlib.pyplot as plt
from main import Net

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
env = gym.make("CartPole-v0",render_mode='human')
n_states = env.observation_space.shape[0]  # 4
n_actions = env.action_space.n  # 2
# 实例化DQN
q_net = Net(n_states, 128, n_actions)

q_net.load_state_dict(torch.load('./q_net.pt'))

for i in range(10):
    state = env.reset()[0]  # len=4
    # 记录每个回合的回报
    episode_return = 0
    done = False
    while True:
        state = torch.Tensor(state[np.newaxis, :])
        actions_value = q_net(state)
        action = actions_value.argmax().item()  # int
        # 更新环境
        next_state, reward, done, truncated, _ = env.step(action)
        # if truncated:
        #     done = True
        # 更新当前状态
        state = next_state
        # 更新回合回报
        episode_return += reward
        # 结束
        if done: 
            print(f'reward:{episode_return}')
            break