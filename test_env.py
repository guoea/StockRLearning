#!/usr/bin/python
#coding=utf8
"""
# Author: shikanon
# Created Time : 2019-06-07 15:53:11

# File Name: test_agent.py
# Description: 测试环境

"""
import pytest
import random
import logging
from StockRLearning.env import StockTradingEnv

logger = logging.getLogger(__name__)

def test_Env():
    random.seed(19910308)
    env = StockTradingEnv(MAX_STEP=80)
    env.reset()
    # test step 错误类型
    try:
        env.step(2)
    except ValueError:
        checkErrorType = True
    assert checkErrorType == True
    obs, reward, done, info = env.step(1)
    assert done == False
    assert env.current_step == 1
    assert info["action"] == "buy"
    # test Max step
    for i in range(80):
        obs, reward, done, info = env.step(1)
        if i > 78:
            assert done == True
    # test reward
    env.reset()
    obs, reward, done, info = env.step(1) # 买
    assert info["action"] == "buy"
    logger.info("reward:%f"%reward)
    logger.info(env.stock["close"].iloc[1])
    logger.info(env.stock["close"].iloc[0])
    diff = (env.stock["close"].iloc[1] - env.stock["close"].iloc[0])/env.stock["close"].iloc[0] - env.TAX_RATE
    assert round(env.total_reward) == round((1+diff)*10000)
    obs, reward, done, info = env.step(0) # 卖
    assert info["action"] == "sell"
    # 卖得预期收益为税率
    assert reward == -env.TAX_RATE * env.total_reward
    assert round(env.total_reward) == round((1 + diff)*10000 + reward)

