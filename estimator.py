#!/usr/bin/python
#coding=utf8
"""
# Author: shikanon
# Created Time : 2019-06-11 15:53:11

# File Name: features.py
# Description:

"""
import numpy as np
import pandas as pd

class Estimator:
    def __init__(self,name,**kwargs):
        param = "".join("%s:%s/"%(k,kwargs[k]) for k in kwargs)
        self.name = name + "/" + param
        self.baseline_rewards = []
        self.rewards = []
        self.win_probs = []
        self.loss = []
    
    def to_df(self):
        data = dict()
        data["baseline_rewards"] = self.baseline_rewards
        data["rewards"] = self.rewards
        data["win_probs"] = self.win_probs
        data["loss"] = self.loss
        return pd.DataFrame(data)
    
    def estimate(self, env):
        # 基准收益
        self.baseline_rewards.append(np.mean(env.history["full_warehouse_income"]))
        # 策略收益
        self.rewards.append(np.mean(env.history["total_reward"])-env.init_money)
        # 胜率
        win_rate = 100*len([r for r in env.history["episode_reward"] if r>0])/len(env.history["episode_reward"])
        self.win_probs.append(win_rate)
        # 最佳收益的误差
        self.loss.append(np.mean(env.history["loss"]))

    def plot(self):
        from IPython import display
        import matplotlib.pyplot as plt
        display.clear_output(wait=True)
        display.display(plt.gcf())
        plt.figure(figsize=(20,10))
        plt.clf()
    
        plt.title('Estimator %s'%self.name)
        plt.xlabel('Episode')
        plt.ylabel('Duration')

        plt.plot(self.loss)
        plt.text(len(self.loss)-1, self.loss[-1], str(self.loss[-1]))
        
        # if len(self.rewards) > 0:
        #     plt.plot(self.rewards)
        #     plt.text(len(self.rewards)-1, self.rewards[-1], str(self.rewards[-1]))
        # if len(self.baseline_rewards) > 0:
        #     plt.plot(self.baseline_rewards)
        #     plt.text(len(self.baseline_rewards)-1, self.baseline_rewards[-1], str(self.baseline_rewards[-1]))

