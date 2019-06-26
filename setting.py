#!/usr/bin/python
#coding=utf8
"""
# Author: shikanon
# Created Time : 2019-05-27 15:53:11

# File Name: setting.py
# Description:配置文件

"""
import logging

LOGLEVEL = logging.ERROR

logging.basicConfig(level = LOGLEVEL, format = '[%(levelname)s] - %(name)s -%(asctime)s -  %(message)s')
logger = logging.getLogger(__name__)