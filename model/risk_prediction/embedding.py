from pandas import DataFrame

from core.BaseFeatureEmbedding import BaseFeatureEmbedding


class Embedding(BaseFeatureEmbedding):

    def text_embedding(self, df: DataFrame, col: list):
        """
        文本处理
        :param df: dataframe类型的数据
        :param col: 文本列表
        :return: 预处理完的数据
        """
        # 自然数编码,lgb用到（文本转换成数字枚举）
        for col_item in col:
            df[col_item] = Embedding.label_encode(Embedding, df[col_item])
        return df

    def num_process(self, df: DataFrame, col: list):
        """
        数字处理
        :param df: dataframe类型的数据
        :param col: 文本列表
        :return: 预处理完的数据
        """
        return df

    # 自然数编码,lgb用到
    def label_encode(self, series):
        unique = list(series.unique())
        # unique.sort()
        return series.map(dict(zip(unique, range(series.nunique()))))