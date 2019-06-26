#!/usr/bin/python
#coding=utf8
"""
# Author: shikanon
# Created Time : 2019-05-27 15:53:11

# File Name: data_processing.py
# Description:

"""
import datetime
import logging
import os
import time
import pandas as pd
import tushare as ts

from pytdx.hq import TdxHq_API
from StockRLearning.data.util import load_config, create_if_not_exists
from StockRLearning.data.database import StockDatabase, StockDailyData



# global variable
logging.basicConfig(level = logging.DEBUG, format = '[%(levelname)s] - %(name)s -%(asctime)s -  %(message)s')
logger = logging.getLogger(__name__)



class StockData:
    def __init__(self, rootPath="./stock_data"):
        # load config file
        self.parser_config("config.conf")
        # init database
        self.stock = StockDatabase()
        self.stock.init_db("config.conf")
        # tushare pro config
        ts.set_token(self.tushare_key)
        self.pro = ts.pro_api()
        # connect tdx
        self.api = TdxHq_API(auto_retry=True, raise_exception=True)
        self.rootPath = rootPath
        self._stock_cache = self.rootPath + "/stock_cache/"
        create_if_not_exists(self._stock_cache)
        self._stock_daily_data_cache_path = self.rootPath + "/daily_data/"
        create_if_not_exists(self._stock_daily_data_cache_path)
        self._all_stock_info_cache_path = self.rootPath + "/stock_info/"
        create_if_not_exists(self._all_stock_info_cache_path)
        self._stock_extend_data_cache_path = self.rootPath + "/stock_extend/"
        create_if_not_exists(self._stock_extend_data_cache_path)
        self._stock_rights_path = self.rootPath + "/stock_rights_cache/"
        create_if_not_exists(self._stock_rights_path)
        self.get_all_stock_info()


    def parser_config(self, filename):
        conf = load_config(filename)
        self.tushare_key = conf.get("tushare", "key")
    

    def _get_default_value(self, start, end):
        if start == None:
            start = "19910101"
        if end == None:
            end = datetime.datetime.now().strftime("%Y%m%d")
        return start, end
    

    def get_all_stock_info(self, force_update=False):
        '''get all stock informations
        '''
        self._all_stock_info_cache_file = self._all_stock_info_cache_path + "stock_info.csv"
        if force_update or (not os.path.exists(self._all_stock_info_cache_file)):
            self.all_stock_info = self.pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,area,industry,list_date')
            self.all_stock_info.to_csv(self._all_stock_info_cache_file, index=False)
        else: # use cache
            self.all_stock_info = pd.read_csv(self._all_stock_info_cache_file)
    

    def download_stock_daily_data(self, code, start, end, force_update=False):
        '''download a stock daily data in cache'''
        filename = self._stock_daily_data_cache_path + "%s_%s_%s.csv"%(code[:-3], start, end)
        if not os.path.exists(filename) or force_update:
            data = self.pro.daily(ts_code=code, adj='qfq', start_date=start, end_date=end)
            if len(data)>0:
                logger.debug(code, ": ", len(data))
                data.to_csv(filename)
            else:
                logger.info("no data: " + code)


    def download_all_stock_daily_data(self, start=None, end=None, force_update=False):
        '''download stock in cache
        '''
        start, end = self._get_default_value(start, end)
        n = 0 # progress management
        m = 0 # observed the tushare rules, limit the number of downloads per minute 
        total_n = len(self.all_stock_info.ts_code)
        star_time = time.time()
        for code in self.all_stock_info.ts_code:
            if n % 10 == 0:
                logger.info("start %s download...(%d/%d)"%(code, n, total_n))
            n = n + 1
            filename = self._stock_daily_data_cache_path + "%s_%s_%s.csv"%(code[:-3], start, end)
            if not os.path.exists(filename) or force_update:
                if m > 190 and (time.time() - star_time) < 60:
                    time.sleep(5)
                elif (time.time() - star_time) > 60:
                    m = 0
                    star_time = time.time()
                data = self.pro.daily(ts_code=code, adj='qfq', start_date=start, end_date=end)
                m = m + 1 # use once api
                if len(data)>0:
                    logger.debug(code, ": ", len(data))        
                    data.to_csv(filename)
                else:
                    logger.info("no data: " + code)
        logger.info("download of stock is finished!")


    def download_stock_extend_data(self, code, start, end, force_update=False):
        '''download extend stock data in cache, include pe, pb, turnover, etc.
        '''
        filename = self._stock_extend_data_cache_path + "%s_%s_%s.csv"%(code[:-3], start, end)
        if not os.path.exists(filename) or force_update:
            data = self.pro.daily_basic(ts_code=code, start_date=start, end_date=end)
            if len(data)>0:
                logger.debug(code, ": ", len(data))
                data.to_csv(filename)
            else:
                logger.info("no data: " + code)
            return True
        else:
            return False
    

    def download_all_stock_extend_data(self, start=None, end=None, force_update=False):
        '''download extend stock data in cache, include pe, pb, turnover, etc.
        '''
        start, end = self._get_default_value(start, end)
        n = 0 # progress management 
        m = 0 # observed the tushare rules, limit the number of downloads per minute 
        total_n = len(self.all_stock_info.ts_code)
        star_time = time.time()
        for code in self.all_stock_info.ts_code:
            if n % 10 == 0:
                logger.info("start %s download...(%d/%d)"%(code, n, total_n))
            n = n + 1

            if m > 190 and (time.time() - star_time) < 60:
                time.sleep(5)
            elif (time.time() - star_time) > 60:
                m = 0
                star_time = time.time()
            success = self.download_stock_extend_data(code, start, end)
            if success:
                m = m + 1 # use once api
        logger.info("download of stock is finished!")
    

    def download_rights_data(self, end=None):
        '''download stock rights to use deprive of right'''
        if end == None:
            end = datetime.datetime.now().strftime("%Y%m%d")
        
        n = 0
        for code in self.all_stock_info.ts_code:
            filename = self._stock_rights_path + "%s_%s.csv"%(code[:-3], end)
            if not os.path.exists(filename):
                data = self.pro.adj_factor(ts_code=code, trade_date='')
                data.to_csv(filename)
        logger.info("download of all stock rights data is finished!")
    # https://github.com/QUANTAXIS/QUANTAXIS/blob/master/QUANTAXIS/QAData/data_fq.py
    # data['preclose'] = (
    #     data['close'].shift(1) * 10 - data['fenhong'] +
    #     data['peigu'] * data['peigujia']
    # ) / (10 + data['peigu'] + data['songzhuangu'])

    # if fqtype in ['01', 'qfq']:
    #     data['adj'] = (data['preclose'].shift(-1) /
    #                    data['close']).fillna(1)[::-1].cumprod()
    # else:
    #     data['adj'] = (data['close'] /
    #                    data['preclose'].shift(-1)).cumprod().shift(1).fillna(1)

    # for col in ['open', 'high', 'low', 'close', 'preclose']:
    #     data[col] = data[col] * data['adj']
    # data['volume'] = data['volume'] / \
    #     data['adj'] if 'volume' in data.columns else data['vol']/data['adj']
    # try:
    #     data['high_limit'] = data['high_limit'] * data['adj']
    #     data['low_limit'] = data['high_limit'] * data['adj']
    # except:
    #     pass


    def save_db(self, code, start=None, end=None):
        start, end = self._get_default_value(start, end)
        stock_filename = self._stock_daily_data_cache_path + "%s_%s_%s.csv"%(code, start, end)
        extend_filename = self._stock_extend_data_cache_path + "%s_%s_%s.csv"%(code, start, end)
        rights_filename = self._stock_rights_path + "%s_%s.csv"%(code, end)
        if not os.path.exists(stock_filename):
            self.download_stock_daily_data(code, start, end)
        if not os.path.exists(extend_filename):
            self.download_stock_daily_data(code, start, end)
        if not os.path.exists(rights_filename):
            self.download_rights_data(end)
        stock_data = pd.read_csv(stock_filename)
        extend_data = pd.read_csv(extend_filename)
        rights_data = pd.read_csv(rights_filename)
        stock_data = pd.merge(stock_data, extend_data, on=["ts_code", "trade_date"], how="left", suffixes=('', '_2'))
        stock_data = pd.merge(stock_data, rights_data, on=["ts_code", "trade_date"], how="left", suffixes=('', '_2'))
        stock_data["code"] = stock_data["ts_code"].apply(lambda x: x[:-3])
        stock_data["date"] = stock_data["trade_date"].apply(lambda x: datetime.datetime(int(str(x)[:4]),int(str(x)[4:6]),int(str(x)[6:])))
        stock_data["p_change_f"] = stock_data["pct_chg"]
        stock_data["total_market_value"] = stock_data["total_mv"]
        stock_data["float_market_value"] = stock_data["circ_mv"]
        stock_data["turnover"] = stock_data["turnover_rate"]
        stock_data["turnover_f"] = stock_data["turnover_rate_f"]
        # deprive of right
        stock_data["close_f"] = stock_data["close"] * stock_data["adj_factor"] / stock_data.iloc[0]["adj_factor"]
        stock_data["open_f"] = stock_data["open"] * stock_data["adj_factor"] / stock_data.iloc[0]["adj_factor"]
        stock_data["high_f"] = stock_data["high"] * stock_data["adj_factor"] / stock_data.iloc[0]["adj_factor"]
        stock_data["low_f"] = stock_data["low"] * stock_data["adj_factor"] / stock_data.iloc[0]["adj_factor"]
        stock_data["pre_close_f"] = stock_data["close_f"].shift(-1) # 这里的数据还是从新到旧，所以用 shift(-1)
        # field
        save_field = ["date","code","open","close","high","low","pre_close","change",
                    "open_f","close_f","high_f","low_f","pre_close_f","p_change_f", 
                    "vol", "amount", "turnover", "turnover_f", "volume_ratio", 
                    "pe", "pe_ttm", "pb", "ps", "ps_ttm", "total_share", "float_share",
                    "free_share", "total_market_value", "float_market_value"]
        self.stock.save_db(stock_data[save_field], "stock_daily_data")
    

    def save_all_to_database(self, start=None, end=None):
        start, end = self._get_default_value(start, end)
        n = 0 # count number
        for _, _, files in os.walk(self._stock_daily_data_cache_path):
            for fname in files:
                if ".csv" in fname and len(fname.split("_")) == 3:
                    fname = fname[:-4]
                    code, code_start, code_end = fname.split("_")
                    if start == code_start and end == code_end:
                        self.save_db(code, start, end)
                        n += 1
                        logger.info("save stock data process (%d/%d)"%(n, len(self.all_stock_info.ts_code)))
    

    def get_stock_daily_data(self, code, start=None, end=None, ex_right=None):
        '''get stock'''
        start, end = self._get_default_value(start, end)
        filename = self._stock_cache + "cache_%s_%s_%s.plk"%(code, start, end)
        if not os.path.exists(filename):
            data = self.stock.load_data("stock_daily_data", code, start, end)
            if len(data) != 0:
                data.to_pickle(filename) 
            else:
                logger.fatal("the code %s have not data!"%code)   
        else:
            data = pd.read_pickle(filename)
        if ex_right == "fontend":
            data["close"] = data["close_f"]
            data["open"] = data["open_f"]
            data["high"] = data["high_f"]
            data["low"] = data["low_f"]
            data["p_change"] = data["p_change_f"]
            data.drop(["close_f", "open_f", "high_f", "low_f", "p_change_f"],axis=1)
        return data

