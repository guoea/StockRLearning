#!/usr/bin/python
#coding=utf8
"""
# Author: shikanon
# Created Time : 2019-05-27 15:53:11

# File Name: util.py
# Description:
"""
import os
import configparser


def load_config(confile): 
    conf = configparser.ConfigParser()
    conf.read(confile)
    return conf

def create_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)