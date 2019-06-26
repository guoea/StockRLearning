#!/usr/bin/python
#coding=utf8
"""
# Author: shikanon
# Created Time : 2019-05-27 15:53:11

# File Name: download.py
# Description:

"""

from data.get_stock import StockData



if __name__ == "__main__":
    stock = StockData()
    stock.get_all_stock_info()
    stock.download_all_stock_daily_data()
    stock.download_all_stock_extend_data()
    stock.download_rights_data()
    stock.save_all_to_database()