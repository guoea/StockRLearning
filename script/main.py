#!/usr/bin/python
#coding=utf8
"""
# Author: shikanon
# Created Time : 2019-05-26 15:53:11

# File Name: main.py
# Description:

"""
import click
import pandas as pd
from StockRLearning.data.get_stock import StockData
from StockRLearning.agent import Agent
from StockRLearning.data_processing import DataProcess


@click.group()
def cli():
    '''创建命令组'''
    pass

@click.command()
@click.option('--one-date', default=False, help='下载一天数据', is_flag=True)
@click.option('--date', help='下载股票日期')
@click.option('--all-date', help='下载所有数据', is_flag=True)
def download(all_date, one_date, date):
    '''下载数据
    '''
    if one_date:
        print("下载所有股票一天的数据")
    else:
        stock = StockData()
        stock.get_all_stock_info()
        stock.download_all_stock_daily_data()
        stock.download_all_stock_extend_data()
        stock.download_rights_data()
        stock.save_all_to_database()


@click.command()
@click.option('--code', default="601398",type=str, help='股票代码')
@click.option('--start-time', default="19910101",type=str, help='起始日期:19910101')
@click.option('--end-time', default="20190601",type=str, help='结束日期:20190601')
def create_feature(code, start_time, end_time):
    '''生成特征字段
    '''
    process = DataProcess() 
    process.clean()
    df = process.make_features(code, start_time, end_time, True)
    print(df.columns)
    print(df.head())
    train_data, test_data = process.split_data(df)
    test_data.to_pickle(process.test_data_path+"%s.plk"%test_data["code"].iloc[0])
    process.random_split(train_data)


@click.command()
def all_for_feature():
    '''生成特征字段
    '''
    stock = StockData()
    stock.get_all_stock_info()
    process = DataProcess()
    process.clean()
    for code in stock.all_stock_info.ts_code:
        code = code[:-3]
        df = process.make_features(code, "19910101", "20190601", True)
        print(df.columns)
        print(df.head())
        train_data, test_data = process.split_data(df)
        test_data.to_pickle(process.test_data_path+"%s.plk"%test_data["code"].iloc[0])
        process.random_split(train_data)


@click.command()
def agent():
    '''运行Agent
    '''
    agent = Agent()
    agent.choice_action()
    result = pd.DataFrame(agent.transactions)
    print(result.head())

cli.add_command(download)
cli.add_command(agent)
cli.add_command(create_feature)
cli.add_command(all_for_feature)



if __name__ == "__main__":
    cli()