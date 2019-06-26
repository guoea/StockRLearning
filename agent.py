#!/usr/bin/python
#coding=utf8
"""
# Author: shikanon
# Created Time : 2019-05-26 15:53:11

# File Name: agent.py
# Description:

"""
import math
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from StockRLearning.env import StockTradingEnv
from StockRLearning.estimator import Estimator



class Agent:
    def __init__(self, mode="train"):
        if mode == "train":
            self.env = StockTradingEnv()
        elif mode == "test":
            self.env = StockTradingEnv("./train_data/test/", MAX_STEP=100000)
        else:
            self.env = StockTradingEnv("./features/", MAX_STEP=100000)
        self.env.reset()
        self.transactions = {
            "date": [], 
            "capital":[], 
            "reward":[], 
            "action": [], 
            "code":[],
            }
    
    def get_action(self, features):
        '''and feature_dict["MA5_breakout"] > feature_dict["MA60_breakout"]
        '''
        features_name = ['turnover', 'p_change',
       'CCI_14', 'RSI_SMA_6', 'RSI_EWMA_6', 'MACD_12_26_9', 'MACD_TREND_3',
       'KDJ_K', 'KDJ_D', 'KDJ_J', 'VR_26', 'VR_MA_6', 'ZJTJ_9',
       'ZJTJ_uprolling_9', 'LON_10', 'LON_MA10', 'ATR', 'ATR_cummean',
       'number', "MA5_ratio", "MA10_ratio", "MA20_ratio", "MA30_ratio",
        "MA40_ratio", "MA60_ratio", "MA90_ratio", 'action']
        feature_dict = dict(zip(features_name, features))
        if feature_dict["MA5_ratio"] > feature_dict["MA60_ratio"]:
            return 1 #买入
        elif feature_dict["MA5_ratio"] < feature_dict["MA60_ratio"]:
            return 0 #卖出
        else: #其他得情况保持不变，返回上一次得结果
            return feature_dict["action"]

    def choice_action(self):
        for i in range(len(self.env.stock)):
            if i == 0: # 第一天不操作
                action = 0
            else:
                action = self.get_action(obs)
            obs, reward, done, info = self.env.step(action)
            self.transactions["date"].append(info["date"])
            self.transactions["capital"].append(info["totalReward"])
            self.transactions["reward"].append(info["afterReward"])
            self.transactions["action"].append(info["action"])
            self.transactions["code"].append(info["code"])
            if done:
                break


class BaseModel:
    def __init__(self, env):
        self.features_name = list(env.features.columns) + ["action"]
        print(self.features_name)

    def get_action(self, features):
        feature_dict = dict(zip(self.features_name, features))
        if feature_dict["KDJ_J"] < 0:
            return 1 #买入
        elif feature_dict["KDJ_J"] > 100:
            return 0 #卖出
        else: #其他得情况保持不变，返回上一次得结果
            return feature_dict["action"]

            
class Net(nn.Module):
    def __init__(self, input_size, output_size):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(input_size, 16)
        self.linear2 = nn.Linear(16, 16)
        self.linear3 = nn.Linear(16, output_size)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        return x




class PGAgent(Agent):
    '''
    Policy Gradient
    '''
    def __init__(self, *args, **kwargs):
        super(PGAgent, self).__init__(*args, **kwargs)


            
           


class DQNAgent(Agent):
    '''
    Deep Q learning
    '''
    def __init__(self, *args, **kwargs):
        super(DQNAgent, self).__init__(*args, **kwargs)
        self.batch_size = 512 # 训练批
        self.decay_rate = 0.9 # 衰减率
        self.decay_each_batch = 10 # 表示n轮衰减一次
        self.random_prob_max = 0.95 # 最大随机概率
        self.random_prob_min = 0.10 # 最小随机概率
        self.basemodel = BaseModel(self.env)# 基准模型
        self.choice_basemodel_prob = 0.8 # 基准模型的概率
        self.gamma = 0.8 # 未来奖励的衰减系数
        self.lr = 0.002 # 模型学习速率
        for key, value in kwargs.items():
            setattr(self, key, value)
        input_dim = self.env.observation_space.shape[0]
        if torch.torch.cuda.is_available():
            self.eval_net = Net(input_dim, 2).cuda()
        else:
            self.eval_net = Net(input_dim, 2)
        self.optimizer = optim.Adam(self.eval_net.parameters(), lr=self.lr)
        self.samples = [] # state1, action1, reward2, state2
        self.count = 0 # 计数
        self.estimator = Estimator(name="DQN") # 评估器


    def get_action(self, obs):
        num = self.count//(self.batch_size*self.decay_each_batch)
        if  num < 500:
            random_prob = self.random_prob_min + (self.random_prob_max - self.random_prob_min)*(self.decay_rate**(num))
        else:
            random_prob = self.random_prob_min
        if random.random() < random_prob:
            if random.random() <  self.choice_basemodel_prob:
                return self.basemodel.get_action(obs)
            return random.randrange(1) # 随机生成0,1
        # 生成tensor
        x =  torch.tensor(obs, dtype=torch.float).view(1,-1)
        if torch.torch.cuda.is_available():
            x = x.cuda()
        pre_action = torch.argmax(self.eval_net(x)).item()
        return pre_action
        
    
    def update(self):
        random.shuffle(self.samples)
        s0, a0, r1, s1 = zip(*self.samples)
        s0 = torch.tensor(s0, dtype=torch.float)
        a0 = torch.tensor(a0, dtype=torch.long).view(self.batch_size, -1)
        r1 = torch.tensor(r1, dtype=torch.float).view(self.batch_size, -1)
        s1 = torch.tensor(s1, dtype=torch.float)
        if torch.torch.cuda.is_available():
            s0 = s0.cuda()
            a0 = a0.cuda()
            r1 = r1.cuda()
            s1 = s1.cuda()
        y_true = r1 + self.gamma * torch.max(self.eval_net(s1).detach(), dim=1)[0].view(self.batch_size, -1)
        # action 是通过 state0 取最大得到，因此gather可以得到state0中最大的概率
        y_pred = self.eval_net(s0).gather(1, a0)
        if torch.torch.cuda.is_available():
            y_pred = y_pred.cuda()
            y_true = y_true.cuda()
        loss_fn = nn.MSELoss()
        loss = loss_fn(y_pred, y_true)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    

    def train(self):
        random.seed(19910308)
        self.score = [1]
        for episode in range(3000):
            self.env.reset()
            for i in range(len(self.env.stock)):
                last_obs = self.env.obs
                action = self.get_action(last_obs)
                next_obs, reward, done, info = self.env.step(action)
                last_obs = self.env.obs #运行step之前的obs
                self.samples.append([last_obs, action, reward, next_obs])
                if len(self.samples) == self.batch_size: # 模型更新
                    self.update()
                    self.samples = self.samples[self.batch_size//5:] # 每次剔除五分之一
                last_obs = next_obs
                self.count += 1
                if done:
                    break
            self.estimator.estimate(self.env)
    

    def replay(self):
        self.replay_env = StockTradingEnv("./train_data/test/", MAX_STEP=self.batch_size)
        self.replay_env.reset()
        for i in range(len(self.replay_env.stock)):
            if i == 0: # 第一天不操作
                action = 0
            else:
                action = self.get_action(last_obs)
            next_obs, reward, done, info = self.replay_env.step(action)
            last_obs = self.replay_env.obs #运行step之前的obs
            if done:
                break
        self.estimator.estimate(self.replay_env)
    


        