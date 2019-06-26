#!/usr/bin/python
#coding=utf8
"""
# Author: shikanon
# Created Time : 2019-05-27 15:53:11

# File Name: features.py
# Description:

"""
import datetime
import pandas as pd
import numpy as np

def CCI_index(data, window_num=14):
    data["HLC"] = (data["high"] + data["low"] + data["close"])/3
    HLC_rolling = data["HLC"].rolling(window_num)
    data["CCI_%d"%window_num] = (data["HLC"] - HLC_rolling.mean())/(HLC_rolling.apply(lambda x: np.mean(np.absolute(x - np.mean(x))), raw=True))
    return ["CCI_%d"%window_num]


def RSI_SMA_index(data, window_num=6):
    # the data is from past to present
    # 信通达里的SMA就是EWM(window_num),或者EWM(alpha=1/(window_num+1))
    # 信通达里 EMA 等于 SMA(N+1,2), EWM(alpha=2/(window_num+2))
    diff = data["close"] - data["pre_close"]
    up = diff.copy()
    up[up < 0] = 0 # make down to zero
    roll_up = up.ewm(alpha=1/(window_num+1)).mean()
    roll_down = diff.abs().ewm(alpha=1/(window_num+1)).mean()
    data["RSI_SMA_%d"%window_num] = roll_up / roll_down
    return ["RSI_SMA_%d"%window_num]


def RSI_MA_index(data, window_num=6):
    diff = data["close"] - data["pre_close"]
    up, down = diff.copy(), diff.copy()
    up[up < 0] = 0 # make down to zero
    down[down > 0] = 0
    roll_up = up.rolling(window_num).mean()
    roll_down = down.abs().rolling(window_num).mean()
    data["RSI_MA_%d"%window_num] = 1 - (1/ (1 + roll_up / roll_down))
    return ["RSI_MA_%d"%window_num]


def MACD_index(data, short_window=12, long_window=26, mid_window=9):
    # EMA 等于 SMA(N+1,2), EWM(alpha=2/(window_num+2))
    diff = data["close"].ewm(alpha=2/(short_window+2)).mean() - data["close"].ewm(alpha=2/(long_window+2)).mean()
    dea = diff.ewm(alpha=2/(mid_window+2)).mean()
    macd_name = "MACD_%d_%d_%d"%(short_window, long_window, mid_window)
    data[macd_name] = diff - dea
    data["MACD_TREND_3"] = (data[macd_name] - data[macd_name].shift().rolling(3).mean()).rolling(3).mean()
    return [macd_name, "MACD_TREND_3"]


def KDJ_index(data, window_num=9, k_window=3, d_window=3):
    hhv = data["high"].rolling(window_num).max() # the high of high price
    llv = data["low"].rolling(window_num).min() # the low of low price
    rsv = (data["close"] - llv)/(hhv - llv)
    k = rsv.ewm(k_window).mean()
    d = k.ewm(d_window).mean()
    j = 3 * k - 2 * d
    data["KDJ_K"] = k
    data["KDJ_D"] = d
    data["KDJ_J"] = j
    return ["KDJ_K", "KDJ_D", "KDJ_J"]


def VR_index(data, n=26, m=6):
    '''成交量变异率'''
    # we use turnover instead of all volumns to reduce the impact of price.
    diff = data["close"] - data["pre_close"]
    up, down, plain = data["turnover_f"].copy(), data["turnover_f"].copy(), data["turnover_f"].copy()
    up[diff < 0] = 0
    down[diff > 0] = 0
    plain[diff != 0] = 0
    up = up.rolling(n).sum()
    down = down.sum()
    plain = plain.sum()
    data["VR_%d"%n] = (up * 2 + plain)/(down * 2 + plain)
    data["VR_MA_%d"%(m)] = data["VR_%d"%n].rolling(m).mean()
    return ["VR_%d"%n, "VR_MA_%d"%(m)]


def ZJTJ_index(data, window_num=9):
    '''通达信庄家抬轿指数'''
    double_ema_close = data["close"].ewm(alpha=2/(window_num+2)).mean().ewm(alpha=2/(window_num+2)).mean()
    pre_double_ema_close = double_ema_close.shift()
    control_line = (double_ema_close - pre_double_ema_close)/pre_double_ema_close * 100
    data["ZJTJ_%d"%window_num] = control_line
    up = control_line.copy()
    up[(up < 0)|((up - up.shift()<0))] = 0
    data["ZJTJ_uprolling_%d"%window_num] = up.rolling(3).mean()
    return ["ZJTJ_%d"%window_num, "ZJTJ_uprolling_%d"%window_num]



def LON_index(data, window_num=10, longHead=10, longTail=20):
    '''通达信龙系长线指数'''
    # we use turnover instead of all volumns to reduce the impact of price.
    hhv = data["high"].rolling(2).max() # the high of high price
    llv = data["low"].rolling(2).min() # the low of low price
    vid = data["turnover_f"].rolling(2).sum()/((hhv - llv)*100)
    rc = (data["close"] - data["pre_close"]) * vid
    long_index = rc.cumsum()# all history data cumsum
    data["LON_%d"%window_num] = (long_index.ewm(longHead).mean() - long_index.ewm(longTail).mean())
    data["LON_MA%d"%window_num] = data["LON_%d"%window_num] - data["LON_%d"%window_num].rolling(10).mean()
    return ["LON_%d"%window_num, "LON_MA%d"%window_num]


def ATR_index(data):
    data["ATR_factor_1"] = data["high"] - data["low"]
    data["ATR_factor_2"] = (data["pre_close"] - data["low"]).abs()
    data["ATR_factor_3"] = (data["pre_close"] - data["low"]).abs()
    data['ATR'] = np.maximum.reduce(data[['ATR_factor_1', 'ATR_factor_2', 'ATR_factor_3']].values, axis=1)
    number = range(1, len(data)+1)
    data["ATR_cummean"] = data["ATR"].cumsum()/number
    data["number"] = np.log(number)/10
    return ["ATR", "ATR_cummean", "number"]


def RSRS_index(data):
    pass


def RANGE_index(data):
    ''' 多头排列
    '''
    data["MA5"] = data["close"].rolling(5).mean()
    data["MA10"] = data["close"].rolling(10).mean()
    data["MA20"] = data["close"].rolling(20).mean()
    data["MA30"] = data["close"].rolling(30).mean()
    data["MA40"] = data["close"].rolling(40).mean()
    data["MA60"] = data["close"].rolling(60).mean()
    data["MA90"] = data["close"].rolling(90).mean()
    data["MA5_ratio"] = (data["MA5"] - data["close"])/data["close"]
    data["MA10_ratio"] = (data["MA10"] - data["close"])/data["close"]
    data["MA20_ratio"] = (data["MA20"] - data["close"])/data["close"]
    data["MA30_ratio"] = (data["MA30"] - data["close"])/data["close"]
    data["MA40_ratio"] = (data["MA40"] - data["close"])/data["close"]
    data["MA60_ratio"] = (data["MA60"] - data["close"])/data["close"]
    data["MA90_ratio"] = (data["MA90"] - data["close"])/data["close"]
    return ["MA5_ratio", "MA10_ratio", "MA20_ratio", "MA30_ratio",
            "MA40_ratio", "MA60_ratio", "MA90_ratio"]
    

def MASS_index(data):
    # Introducing Value Assessment
    detal_time = (data.date - datetime.date(1989,1,1)).days / 365
    value = data["float_market_value"]/10000 * (0.92**(detal_time)) # Inflation rate is 8% per year
    data["MASS_ofall"] = round(np.log(value), 1)
    return ["MASS_ofall"]


def JumpCap_index(data):
    # 主要用于大盘指标
    data["JumpCap"] = (data["high"] - data["close"].shift()).apply(lambda x: 1 if x < 0 else 0)
    data["JumpCap"] = data["close"] * data["JumpCap"]
    # 20日内如果没有补价禁止购买
    data["JumpCap_index"] = (data["close"] - data["JumpCap"].rolling(20).max()).apply(lambda x: 1 if x<0 else 0)
    return ["JumpCap_index"]


def ContinuousDrop_index(data):
    # 连续下跌,主要用于大盘指标
    pass