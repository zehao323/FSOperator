"""
Filename: frame.py
Author: szh
Contact: 980536943@qq.com
"""

from core.BaseFunc import BaseFunc


def base_framework(path: str = "", func_class: BaseFunc = BaseFunc()):
    """
    数据分析基础运行框架
    :param path: 文件地址
    :param func_class: 自定义方法对象
    :return: 运行结果
    """
    df = func_class.read_file(path)
    df = func_class.preprocess(df)
    df = func_class.fs(df)
    df = func_class.postprocess(df)
    df = func_class.train(df)
    result = func_class.predict(df)
    return result
