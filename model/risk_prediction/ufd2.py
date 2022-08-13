"""
Filename: utf2.py
Author: szh
Contact: 980536943@qq.com
"""
from core.BaseFunc import BaseFunc
from pandas import DataFrame
import pandas as pd
from utils.import_data import from_mysql, from_mysql_with_sql
import model.risk_prediction.udf as udf


class UdfFunc2(BaseFunc):

    def __init__(self):
        self.func_class = udf.UdfFunc()

    def read_file(self, path: str):
        sql = "SELECT * FROM fs_demo_num2 LIMIT 50"
        # sql = "SELECT * FROM fs_demo_num2 ORDER BY RAND() LIMIT 5000"
        sql = "SELECT * FROM fs_demo_num2"
        df = from_mysql_with_sql("127.0.0.1",
                                 3306,
                                 "root",
                                 "0dubinglian",
                                 "feature_selection",
                                 sql)
        # df = from_mysql("127.0.0.1",
        #                 3306,
        #                 "root",
        #                 "0dubinglian",
        #                 "feature_selection",
        #                 "fs_demo_num")
        return df

    def preprocess(self, df: DataFrame):
        test: DataFrame = df.loc[:, (df.dtypes == 'float64').values]
        test2 = df.loc[:, (df.dtypes == 'object').values]
        print(test.shape)
        return df

    def fs(self, df: DataFrame):
        df = self.func_class.fs(df)
        return df

    def train(self, df: DataFrame):
        # df = self.func_class.train(df)
        return df

    def predict(self, df: DataFrame):
        return self.func_class.predict(df)
