"""
Filename: main.py
Author: szh
Contact: 980536943@qq.com
"""
from core.frame import *
import pandas as pd
# import numpy as np
import warnings
from model.risk_prediction.udf import UdfFunc
from model.risk_prediction.ufd2 import UdfFunc2

warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)  # 显示所有列
# pd.set_option('display.max_rows', None)  # 显示所有行

if __name__ == '__main__':
    # df = base_framework("./data/AIWIN70/ent_info.csv")
    # print(df.head())
    base_framework("./data/AIWIN70/ent_info.csv", UdfFunc())
