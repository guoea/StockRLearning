#!/usr/bin/python
#coding=utf8
"""
# Author: shikanon
# Created Time : 2019-06-27 15:53:11

# File Name: test_agent.py
# Description: 测试智能体

"""
import random
import pytest
import pandas as pd
from StockRLearning.agent import Agent

def test_Agent():
    random.seed(19910308)
    agent = Agent(mode="test")
    agent.choice_action()
    assert len(agent.transactions["action"]) == (len(agent.env.stock))
