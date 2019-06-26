#!/usr/bin/python
#coding=utf8
"""
# Author: shikanon
# Created Time : 2019-05-27 15:53:11

# File Name: data_processing.py
# Description:

"""
import os
import shutil
import logging
import random
import pandas as pd
from StockRLearning.features import *
from StockRLearning.setting import logger
from StockRLearning.data.database import StockDatabase
from StockRLearning.data.util import create_if_not_exists




class DataProcess:
    def __init__(self):
        self.stock_cache = "./stock_cache/"
        self.feature_path = "./features/"
        self.train_data_path = "./train_data/train/"
        self.test_data_path = "./train_data/test/"
        create_if_not_exists(self.stock_cache)
        create_if_not_exists(self.feature_path)
        create_if_not_exists(self.train_data_path)
        create_if_not_exists(self.test_data_path)
        self.db = StockDatabase()
        self.db.init_db("config.conf")


    def get_stock_data(self, code, start, end, ex_right=None):
            filename = self.stock_cache + "cache_%s_%s_%s.plk"%(code, start, end)
            if not os.path.exists(filename):
                data = self.db.load_data("stock_daily_data", code, start, end)
                if len(data) != 0:
                    data.to_pickle(filename)
                else:
                    logger.fatal("the code %s have not data!"%code)
            else:
                data = pd.read_pickle(filename)
            data["p_change"] = (data["close"] - data["pre_close"])/data["pre_close"] * 100
            if ex_right == "fontend":
                data["close"] = data["close_f"]
                data["open"] = data["open_f"]
                data["high"] = data["high_f"]
                data["low"] = data["low_f"]
                data["p_change"] = data["p_change_f"]
                data["pre_close"] = data["close_f"].shift()
            data = data.drop(["close_f", "open_f", "high_f", "low_f", "p_change_f", "p_change_b", "pre_close_f"], axis=1)
            return data


    def make_features(self, code, start, end, force_update=False):
        filename = self.feature_path + "features_%s_%s_%s.plk"%(code, start, end)
        if os.path.exists(filename) and not force_update:
            return pd.read_pickle(filename)
        # data = self.get_stock_data(code, start, end, ex_right="fontend")
        data = self.get_stock_data(code, start, end)
        # features pipelines
        self.fields = ["date", "code", "open", "close", "high", "low","turnover", "p_change"]
        self.fields = self.fields + CCI_index(data, window_num=14) + \
            RSI_SMA_index(data) + \
            RSI_MA_index(data) + \
            MACD_index(data) + \
            KDJ_index(data) + \
            VR_index(data) + \
            ZJTJ_index(data) + \
            LON_index(data) + \
            ATR_index(data) + \
            RANGE_index(data)
        # save cache
        features = data[self.fields].copy()
        features.dropna(inplace=True)
        features.to_pickle(filename)
        return features


    def random_split(self, data, length=180, min_length=20):
        code = data["code"].iloc[0]
        total_length = len(data)
        if total_length > length:
            # random slice ensures data independence and various combination
            choice_length = total_length - length # random choice range
            for i in range(int(choice_length/length)*3):
                end_pos = random.randint(length, total_length)
                start_pos = end_pos - length
                filename = self.train_data_path + "random_%s_%d_%d.plk"%(code, start_pos, end_pos)
                data[start_pos:end_pos].to_pickle(filename)
            # sequential slice ensures that all data is covered
            for i in range(choice_length//length):
                filename = self.train_data_path + "seq_%s_%d_%d.plk"%(code, length*i, length*(i+1))
                data[length*i:length*(i+1)].to_pickle(filename)
            if choice_length % length >= min_length: # processing the remaining data
                filename = self.train_data_path + "seq_%s_%d_%d.plk"%(code, length*(i+1), total_length)
                data[length*(i+1):].to_pickle(filename)
        elif length >= total_length >= min_length:
            filename = self.train_data_path + "seq_%s_0_%d.plk"%(code, total_length)
            data.to_pickle(filename)
    

    def split_data(self, data, n=0.8):
        '''split time series data'''
        split_pos = int(len(data)*n)
        train = data[:split_pos]
        test = data[split_pos:]
        return train, test
    
    def clean(self):
        if os.path.exists(self.feature_path):
            shutil.rmtree(self.feature_path)
        if os.path.exists(self.stock_cache):
            shutil.rmtree(self.stock_cache)
        if os.path.exists(self.train_data_path):
            shutil.rmtree(self.train_data_path)
        if os.path.exists(self.test_data_path):
            shutil.rmtree(self.test_data_path)
        create_if_not_exists(self.stock_cache)
        create_if_not_exists(self.feature_path)
        create_if_not_exists(self.train_data_path)
        create_if_not_exists(self.test_data_path)
