"""
Filename: BaseFeatureEmbedding.py
Author: szh
Contact: 980536943@qq.com
"""

import pandas as pd
# import numpy as np
from pandas import DataFrame


class BaseFeatureEmbedding:
    """
    数据embedding基础框架
    """
    def text_embedding(self, df: DataFrame, col: list):
        """
        文本处理
        :param df: dataframe类型的数据
        :param col: 文本列表
        :return: 预处理完的数据
        """
        return df

    def num_process(self, df: DataFrame, col: list):
        """
        数字处理
        :param df: dataframe类型的数据
        :param col: 文本列表
        :return: 预处理完的数据
        """
        return df
