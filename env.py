#!/usr/bin/python
#coding=utf8
"""
# Author: shikanon
# Created Time : 2019-05-26 15:53:11

# File Name: env.py
# Description:

"""

import gym
import os
import logging
import random
import numpy as np
import pandas as pd

from gym import spaces
from collections import defaultdict
from StockRLearning.data_processing import DataProcess
from StockRLearning.setting import logger


# random.seed(19910308)

class StockData:
    def __init__(self, data_path=None):
        if data_path is None:
            process = DataProcess()
            self.data_path = process.train_data_path
        else:
            self.data_path = data_path
        self.train_filenames = os.listdir(self.data_path)
        self.start_pos = 0
        random.shuffle(self.train_filenames)
    
    def next(self):
        if self.start_pos > len(self.train_filenames) -1:
            self.start_pos = 0
            random.shuffle(self.train_filenames)
        filename = self.train_filenames[self.start_pos]
        logger.info("read file: ",filename)
        self.start_pos = self.start_pos + 1
        return pd.read_pickle(self.data_path + filename)


class StockTradingEnv(gym.Env):
    ''' 股票买卖交易环境
    '''
    def __init__(self, data_path=None, MAX_STEP = 500, TAX_RATE = 0.002, init_money=10000):
        self.DataSource = StockData(data_path)
        self.MAX_STEP = MAX_STEP
        self.TAX_RATE = TAX_RATE
        self.init_money = init_money
    

    def get_action_name(self, action, hold_stock):
        '''获取行为名称'''
        if hold_stock == 0 and action == 1:
            return "buy"
        elif hold_stock == 1 and action == 0:
            return "sell"
        else:
            return "hold"
    

    def _limit_trading_action(self, action, hold_stock):
        # 涨停/跌停限制交易
        p_change = (self.stock["close"].iloc[self.current_step] - self.stock["close"].iloc[self.current_step - 1]) / self.stock["close"].iloc[self.current_step - 1]
        if (p_change > 0.0998 and hold_stock == 0) or (p_change < -0.0998 and hold_stock == 1):
            logger.info("交易限制%f"%p_change)
            action = hold_stock # 通过改变action，强制不能交易
        return action


    def _task_action(self, action):
        '''action == 0 is sell, and action == 1 is buy,
        when action is buy:
            Returns = price(t+1) - price(t) - price(t) * taxes
        when action is sell:
            Returns = - price(t) * taxes
        when action is hold stock:
            Returns = price(t+1) - price(t)
        when action is not hold stock and is inactions:
            Returns = 0
        '''
        # hold_stock is 1 means have no money and holding stock, and 0 means keep money
        hold_stock = self.obs[-1]
        # logger.info("action and origin hold_stock:%d,%d"%(action, hold_stock))
        action = self._limit_trading_action(action, hold_stock)
        action_name = self.get_action_name(action, hold_stock)
        if hold_stock == 0 and action == 1: # have money and buy
            episode_reward = (self.stock["close"].iloc[self.current_step] - self.stock["close"].iloc[self.current_step - 1] * (1 + self.TAX_RATE))/self.stock["close"].iloc[self.current_step - 1]
            hold_stock = 1 # change hold stock state
        elif hold_stock == 1 and action == 0: # sell stock
            # 由于日级，买得时候已经计算了第二天收益，卖得时候只需要计算税率就可以了
            episode_reward = -self.TAX_RATE
            hold_stock = 0 # change hold stock state
        elif hold_stock == 1 and action == 1: # have stock and buy is hold stock
            episode_reward = (self.stock["close"].iloc[self.current_step] - self.stock["close"].iloc[self.current_step - 1])/self.stock["close"].iloc[self.current_step - 1]
        elif hold_stock == 0 and action == 0: # have money and sell is inaction
            episode_reward = 0
        self.total_reward = (1 + episode_reward)*self.total_reward
        self.obs = np.array(self.features.iloc[self.current_step].to_list() + [hold_stock])
        # 最优收益
        if hold_stock == 0:
            best_reward = max((self.stock["close"].iloc[self.current_step] - self.stock["close"].iloc[self.current_step - 1] * (1 + self.TAX_RATE))/self.stock["close"].iloc[self.current_step - 1],
            0)
        else:
            best_reward = max(-self.TAX_RATE,(self.stock["close"].iloc[self.current_step] - self.stock["close"].iloc[self.current_step - 1])/self.stock["close"].iloc[self.current_step - 1])
        loss = best_reward - episode_reward
        # 记录历史数据
        # 满仓操作收益
        self.history["full_warehouse_income"].append(
            self.init_money*((self.stock["close"].iloc[self.current_step] - self.stock["close"].iloc[0])/self.stock["close"].iloc[0] - self.TAX_RATE*2)
        )
        self.history["total_reward"].append(self.total_reward)
        self.history["episode_reward"].append(episode_reward)
        self.history["loss"].append(loss)
        self.history["action"].append(action_name)
        self.history["close"].append(self.stock["close"].iloc[self.current_step])
        return -sum(self.history["loss"]), self.obs, action_name


    def step(self, action):
        if action not in (0,1):
            raise ValueError("action is Error, action:",action)
        self.current_step += 1
        # 步数大于一次模拟 或者 资金额度剩余小于税率
        if self.current_step > self.stock_max_step or self.total_reward <= self.init_money * self.TAX_RATE:
            done = True
            # 最后一天，如果持有股票需要强制卖出, 也就是action=0
            hold_stock = self.obs[-1]
            if hold_stock == 1:
                reward = -self.TAX_RATE*self.total_reward
            else:
                reward = 0
            self.total_reward = self.total_reward + reward
            next_obs = np.zeros(self.obs.shape)
            info = {
                "totalReward": self.total_reward,
                "action": "sell",
                "afterReward": reward,
                "date": self.stock["date"].iloc[self.current_step-1],
                "code": self.code
            }
        else:
            done = False
            # task action
            reward, next_obs, current_action_name = self._task_action(action)
            # obs = self._get_next_observation(action)
            info = {"totalReward": self.total_reward,
                    "action": current_action_name,
                    "afterReward": reward,
                    "date": self.stock["date"].iloc[self.current_step-1],
                    "code": self.code}
        return next_obs, reward, done, info

    
    def reset(self):
        # default the parameter
        # load stock data
        self.stock = self.DataSource.next()
        self.code = self.stock["code"].iloc[0]
        # features
        self.features = self.stock.copy()
        self.features = self.features.drop(["code","date","open","low","high","close"],axis=1)
        logger.info(self.features.columns)
        # describe the format of valid actions 
        self.action_space = spaces.Box(
                low=0, high=1, shape=(1,), dtype=np.float16)
        high = np.array([np.inf]*(self.features.shape[1] + 1))
        self.observation_space = spaces.Box(
                low=-high, high=high, dtype=np.float16)
        self.current_step = 0
        self.stock_max_step = min(self.MAX_STEP, len(self.stock)-1)
        self.total_reward = self.init_money # 初始资金
        self.obs = np.array(self.features.iloc[0].to_list() + [0])
        self.history = defaultdict(list)
        self.history["total_reward"].append(self.total_reward)
        self.history["full_warehouse_income"].append(0)
        self.history["action"].append("hold")
        self.history["episode_reward"].append(0)
        self.history["loss"].append(0)
        self.history["close"].append(self.stock["close"].iloc[self.current_step])
