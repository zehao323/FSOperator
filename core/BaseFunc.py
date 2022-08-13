"""
Filename: BaseFunc.py
Author: szh
Contact: 980536943@qq.com
"""
import pandas as pd
# import numpy as np
from pandas import DataFrame


class BaseFunc:
    """
    数据处理基础框架
    """
    def read_file(self, path: str):
        """
        读取数据
        :param path: 文件地址
        :return: dataframe类型的数据
        """
        return pd.read_csv(path)

    def preprocess(self, df: DataFrame):
        """
        特征选择前的预处理
        :param df: dataframe类型的数据
        :return: 预处理完的数据
        """
        return df

    def fs(self, df: DataFrame):
        """
        特征选择/数据降维
        :param df: dataframe类型的数据
        :return: 特征选择后的数据
        """
        return df

    def postprocess(self, df: DataFrame):
        """
        特征选择后的特征工程
        :param df: dataframe类型的数据
        :return: 特征工程处理完毕的数据
        """
        return df

    def train(self, df: DataFrame):
        """
        模型训练
        :param df: dataframe类型的数据
        :return: 训练完的数据
        """
        return df

    def predict(self, df: DataFrame):
        """
        模型评价/预测
        :param df: dataframe类型的数据
        :return: 预测结果
        """
        return df
