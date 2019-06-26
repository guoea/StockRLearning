#!/usr/bin/python
#coding=utf8
"""
# Author: shikanon
# Created Time : 2019-05-27 15:53:11

# File Name: test_processing.py
# Description: 指标得测试文件

"""

import pytest
import datetime

from StockRLearning.data_processing import DataProcess
from StockRLearning.features import CCI_index, RSI_SMA_index

process = DataProcess()
stock_data = process.get_stock_data("000568","19900101","20190619")
stock_data.index = stock_data["date"]

def test_CCI():
    CCI_index(stock_data, 14)
    assert round(stock_data.loc[datetime.date(2019,5,10)]["CCI_14"]/0.015,2) == -40.75

def test_RSI():
    res = RSI_SMA_index(stock_data, 6)
    assert int(stock_data.loc[datetime.date(2019,5,10)][res]*100) == 51
    res = RSI_SMA_index(stock_data, 12)
    assert int(stock_data.loc[datetime.date(2019,5,10)][res]*100) == 54
    res = RSI_SMA_index(stock_data, 24)
    assert int(stock_data.loc[datetime.date(2019,5,10)][res]*100) == 58