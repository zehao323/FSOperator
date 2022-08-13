"""
Filename: import_data.py
Author: szh
Contact: 980536943@qq.com
"""
import pandas as pd
import pymysql


def from_mysql(host, port, user, password, database, table):
    """
    从mysql中读取文件
    :param host: mysql的ip地址
    :param port: mysql的端口
    :param user: mysql的用户名
    :param password: mysql的密码
    :param database: mysql的数据集
    :param table: 查询的表
    :return: dataframe类型的数据
    """
    db_conn = pymysql.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        database=database,
        charset='utf8'
    )
    # 执行sql操作
    sql = "select * from " + table
    df = pd.read_sql(sql, con=db_conn)
    return df


def from_mysql_with_sql(host, port, user, password, database, sql):
    """
    从mysql中读取文件
    :param host: mysql的ip地址
    :param port: mysql的端口
    :param user: mysql的用户名
    :param password: mysql的密码
    :param database: mysql的数据集
    :param sql: 查询语句
    :return: dataframe类型的数据
    """
    db_conn = pymysql.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        database=database,
        charset='utf8'
    )
    # 执行sql操作
    df = pd.read_sql(sql, con=db_conn)
    return df
