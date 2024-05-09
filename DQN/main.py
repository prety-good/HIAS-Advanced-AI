import random
import torch
import collections
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import gymnasium as gym
import torch.nn.functional as F
import matplotlib.pyplot as plt
# --------------------------------------- #
# 经验回放池
# --------------------------------------- #
 
class ReplayBuffer():
    def __init__(self, capacity):
        # 创建一个先进先出的队列，最大长度为capacity，保证经验池的样本量不变
        self.buffer = collections.deque(maxlen=capacity)
    # 将数据以元组形式添加进经验池
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    # 随机采样batch_size行数据
    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)  # list, len=32
        # *transitions代表取出列表中的值，即32项
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done
    # 目前队列长度
    def size(self):
        return len(self.buffer)
    
# -------------------------------------- #
# 构造深度学习网络，输入状态s，得到各个动作的reward
# -------------------------------------- #
 
class Net(nn.Module):
    # 构造只有一个隐含层的网络
    def __init__(self, n_states, n_hidden, n_actions):
        super(Net, self).__init__()
        # [b,n_states]-->[b,n_hidden]
        self.fc1 = nn.Linear(n_states, n_hidden)
        # [b,n_hidden]-->[b,n_actions]
        self.fc2 = nn.Linear(n_hidden, n_actions)
    # 前传
    def forward(self, x):  # [b,n_states]
        x = self.fc1(x)
        x =  F.relu(x)
        x = self.fc2(x)
        return x
 
# -------------------------------------- #
# 构造深度强化学习模型
# -------------------------------------- #
 
class DQN:
    #（1）初始化
    def __init__(self, n_states, n_hidden, n_actions,
                 learning_rate, gamma, epsilon,
                 target_update, device):
        # 属性分配
        self.n_states = n_states  # 状态的特征数
        self.n_hidden = n_hidden  # 隐含层个数
        self.n_actions = n_actions  # 动作数
        self.learning_rate = learning_rate  # 训练时的学习率
        self.gamma = gamma  # 折扣因子，对下一状态的回报的缩放
        self.epsilon = epsilon  # 贪婪策略，有1-epsilon的概率探索
        self.target_update = target_update  # 目标网络的参数的更新频率
        self.device = device  # 在GPU计算
        # 计数器，记录迭代次数
        self.count = 0
 
        # 构建2个神经网络，相同的结构，不同的参数
        # 实例化训练网络  [b,4]-->[b,2]  输出动作对应的奖励
        self.q_net = Net(self.n_states, self.n_hidden, self.n_actions)
        # 实例化目标网络
        self.target_q_net = Net(self.n_states, self.n_hidden, self.n_actions)
 
        # 优化器，更新训练网络的参数
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.learning_rate)
 
    #（3）网络训练
    def update(self, transition_dict):  # 传入经验池中的batch个样本
        #state, action, reward, next_state, done
        # 获取当前时刻的状态 array_shape=[b,4]
        states = torch.tensor(transition_dict['states'], dtype=torch.float)
        # 获取当前时刻采取的动作 tuple_shape=[b]，维度扩充 [b,1]
        actions = torch.tensor(transition_dict['actions']).view(-1,1)
        # 当前状态下采取动作后得到的奖励 tuple=[b]，维度扩充 [b,1]
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1,1)
        # 下一时刻的状态 array_shape=[b,4]
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float)
        # 是否到达目标 tuple_shape=[b]，维度变换[b,1]
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1,1)
 
        # 输入当前状态，得到采取各运动得到的奖励 [b,4]==>[b,2]==>[b,1]
        # 根据actions索引在训练网络的输出的第1维度上获取对应索引的q值（state_value）
        q_values = self.q_net(states).gather(1, actions)  # [b,1]
        # 下一时刻的状态[b,4]-->目标网络输出下一时刻对应的动作q值[b,2]-->
        # 选出下个状态采取的动作中最大的q值[b]-->维度调整[b,1]
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1,1)
        # 目标网络输出的当前状态的q(state_value)：即时奖励+折扣因子*下个时刻的最大回报
        q_targets = rewards + self.gamma * max_next_q_values * (1-dones)
 
        # 目标网络和训练网络之间的均方误差损失
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))

        self.optimizer.zero_grad()
        dqn_loss.backward()
        self.optimizer.step()
 
        # 在一段时间后更新目标网络的参数
        if self.count % self.target_update == 0:
            # 将目标网络的参数替换成训练网络的参数
            self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.count += 1


    def take_action(self, state):
        # 维度扩充，给行增加一个维度，并转换为张量shape=[1,4]
        state = torch.Tensor(state[np.newaxis, :])
        # 如果小于该值就取最大的值对应的索引
        if np.random.random() < self.epsilon:  # 0-1
            # 前向传播获取该状态对应的动作的reward
            actions_value = self.q_net(state)
            # 获取reward最大值对应的动作索引
            action = actions_value.argmax().item()  # int
        # 如果大于该值就随机探索
        else:
            # 随机选择一个动作
            action = np.random.randint(self.n_actions)
        return action

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
    
    capacity = 500  # 经验池容量
    lr = 2e-3  
    gamma = 0.9  # 折扣因子
    epsilon = 0.9  # 贪心系数
    target_update = 200  # 目标网络的参数的更新频率
    batch_size = 32
    n_hidden = 64  # 隐含层神经元个数
    min_size = 200  # 经验池超过200后再训练
    return_list = []  # 记录每个回合的回报
    
    # 加载环境
    env = gym.make("CartPole-v1", render_mode="human")
    n_states = env.observation_space.shape[0]  # 4
    n_actions = env.action_space.n  # 2
    
    # 实例化经验池
    replay_buffer = ReplayBuffer(capacity)
    # 实例化DQN
    agent = DQN(n_states=n_states,
                n_hidden=n_hidden,
                n_actions=n_actions,
                learning_rate=lr,
                gamma=gamma,
                epsilon=epsilon,
                target_update=target_update,
                device=device,
            )
    
    # 训练模型
    for i in tqdm(range(500), total = 500):  # 100回合
        # 每个回合开始前重置环境
        state = env.reset()[0]  # len=4
        # 记录每个回合的回报
        episode_return = 0
        done = False
        
        while True:
            # 获取当前状态下需要采取的动作
            action = agent.take_action(state)
            # 更新环境
            next_state, reward, done, _, _ = env.step(action)
            # 添加经验池
            replay_buffer.add(state, action, reward, next_state, done)
            # 更新当前状态
            state = next_state
            # 更新回合回报
            episode_return += reward

            # 当经验池超过一定数量后，训练网络
            if replay_buffer.size() > min_size:
                # 从经验池中随机抽样作为训练集
                s, a, r, ns, d = replay_buffer.sample(batch_size)
                # 构造训练集
                transition_dict = {
                    'states': s,
                    'actions': a,
                    'next_states': ns,
                    'rewards': r,
                    'dones': d,
                }
                # 网络更新
                agent.update(transition_dict)
            # 结束
            if done: break
        
        # 记录每个回合的回报
        return_list.append(episode_return)
    
    # 绘图
    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('DQN Returns')
    plt.savefig('./results.png')
    plt.show()

                         