#!/usr/bin/python
#coding=utf8
"""
# Author: shikanon
# Created Time : 2019-05-27 15:53:11

# File Name: database.py
# Description:
"""

import logging
import pandas

from sqlalchemy import Column, CHAR, Date, Float, String, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from StockRLearning.data.util import load_config


Base = declarative_base()
logging.basicConfig(level = logging.INFO,format = '[%(levelname)s] - %(name)s -%(asctime)s -  %(message)s')
logger = logging.getLogger(__name__)


class StockDailyData(Base):
    __tablename__ = "stock_daily_data"

    # to prevent duplication of keywords, add prefix c_ to fields
    c_date = Column("date", Date, primary_key=True, nullable=True, comment="日期")
    c_code = Column("code", CHAR(6), primary_key=True, nullable=True, comment="股票代码")
    c_open = Column("open", Float, comment="开盘价/元")
    c_close = Column("close", Float, comment="收盘价/元")
    c_high = Column("high", Float, comment="最高价/元")
    c_low = Column("low", Float, comment="最低价/元")
    c_pre_close = Column("pre_close", Float, comment="昨日收价/元")
    c_change = Column("change", Float, comment="涨跌价格/元")
    c_p_change = Column("p_change", Float, comment="涨跌幅%(未复权)")
    c_open_f = Column("open_f", Float, comment="开盘价/元(前复权)")
    c_close_f = Column("close_f", Float, comment="收盘价/元(前复权)")
    c_high_f = Column("high_f", Float, comment="最高价/元(前复权)")
    c_low_f = Column("low_f", Float, comment="最低价/元(前复权)")
    c_pre_close_f = Column("pre_close_f", Float, comment="昨日收价/元(前复权)")
    c_p_change_f = Column("p_change_f", Float, comment="前复权_涨跌幅%,当日收盘价 × 当日复权因子 / 最新复权因子")
    c_p_change_b = Column("p_change_b", Float, comment="后复权_涨跌幅%,当日收盘价 × 当日复权因子")
    c_vol = Column("vol", Float, comment="成交量/手")
    c_amount = Column("amount", Float, comment="成交额/千元")
    c_turnover = Column("turnover", Float, comment="换手率%")
    c_turnover_f = Column("turnover_f", Float, comment="自由流通股换手率(%), 成交量/发行总股数")
    c_volume_ratio = Column("volume_ratio", Float, comment="量比,（现成交总手数 / 现累计开市时间(分) ）/ 过去5日平均每分钟成交量")
    c_pe = Column("pe", Float, comment="市盈率, 总市值/过去一年净利润")
    c_pe_ttm = Column("pe_ttm", Float, comment="滚动市盈率(TTM)")
    c_pb = Column("pb", Float, comment="市净率, 总市值/净资产")
    c_ps = Column("ps", Float, comment="市销率,  总市值/主营业务收入")
    c_ps_ttm = Column("ps_ttm", Float, comment="滚动市销率（TTM）")
    c_total_share = Column("total_share", Float, comment="总股本（万股）")
    c_float_share = Column("float_share", Float, comment="流通股本(万股)")
    c_free_share = Column("free_share", Float, comment="自由流通股本(万股)")
    c_total_market_value = Column("total_market_value", Float, comment="总市值(万元)")
    c_float_market_value = Column("float_market_value", Float, comment="流通市值(万元)")

    def __repr__(self):
        return "<StockDailyData(date='%s', code='%s', close='%.2f')>" % (
                                str(self.c_date), self.c_code, self.c_close)


class StockExtendData(Base):
    __tablename__ = "stock_extend_data"

    c_date = Column("date", Date, primary_key=True, nullable=True, comment="日期")
    c_code = Column("code", CHAR(6), primary_key=True, nullable=True, comment="股票代码")
    c_turnover_rate = Column("turnover_rate", Float, comment="换手率（%）, 成交量/发行总股数")
    c_turnover_rate_f = Column("turnover_rate_f", Float, comment="自由流通股换手率(%), 成交量/发行总股数")
    c_volume_ratio = Column("volume_ratio", Float, comment="量比,（现成交总手数 / 现累计开市时间(分) ）/ 过去5日平均每分钟成交量")
    c_pe = Column("pe", Float, comment="市盈率, 总市值/过去一年净利润")
    c_pe_ttm = Column("pe_ttm", Float, comment="滚动市盈率(TTM)")
    c_pb = Column("pb", Float, comment="市净率, 总市值/净资产")
    c_ps = Column("ps", Float, comment="市销率,  总市值/主营业务收入")
    c_ps_ttm = Column("ps_ttm", Float, comment="滚动市销率（TTM）")
    c_total_share = Column("total_share", Float, comment="总股本（万股）")
    c_float_share = Column("float_share", Float, comment="流通股本(万股)")
    c_free_share = Column("free_share", Float, comment="自由流通股本(万股)")
    c_total_mv = Column("total_mv", Float, comment="总市值(万元)")
    c_circ_mv = Column("circ_mv", Float, comment="流通市值(万元)")

    def __repr__(self):
        return "<StockDailyData(date='%s', code='%s', turnover_rate='%.2f')>" % (
                                str(self.c_date), self.c_code, self.c_turnover_rate_f)



class StockDatabase:

    def init_db(self, config_filename):
        conf = load_config(config_filename)
        # mysql database connect engine
        db_host = conf.get("db", "host")
        db_port = conf.get("db", "port")
        db_user = conf.get("db", "user")
        db_password = conf.get("db", "password")
        db_database = conf.get("db", "database")
        self.db_database = db_database
        # Connect to the database
        schema = "mysql+pymysql://%s:%s@%s:%s/%s"%(db_user, db_password, db_host, db_port, db_database)
        self.engine = create_engine(schema)
        self.Session = sessionmaker(bind=self.engine)
        # create table if it not exists;
        try:
            Base.metadata.create_all(self.engine)
        except Exception as e:
            logger.warn(e)


    def save_db(self, dataframe, datatype, method="append"):
        with self.engine.connect() as conn, conn.begin():
            try:
                dataframe.to_sql(name=datatype, con=conn, if_exists=method, index=False, chunksize=1000)
                logger.debug("write database success!")
            except Exception as e:
                logger.fatal(e)


    def load_data(self, datatype, code, start, end):
        select_sql = "select * from %s where code='%s' and date>='%s' and date<='%s'"%(
                datatype, code, start, end)
        return pandas.read_sql_query(select_sql, self.engine)