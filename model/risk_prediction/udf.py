"""
Filename: utf.py
Author: szh
Contact: 980536943@qq.com
"""
from core.BaseFunc import BaseFunc
import pandas as pd
import os
import gc
import xgboost as xgb
# import lightgbm as lgb
# from catboost import CatBoostRegressor
from sklearn.linear_model import SGDRegressor, LinearRegression, Ridge
from sklearn.preprocessing import MinMaxScaler
from gensim.models import Word2Vec
import math
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, log_loss
import matplotlib.pyplot as plt
import time
import warnings
from sklearn import metrics
from pandas import DataFrame
from model.risk_prediction.embedding import Embedding


class UdfFunc(BaseFunc):

    def read_file(self, path: str):
        news_list = []
        base_url = '/Users/shenzehao/Desktop/daslab/FSOpr/project/FSOperator/data/AIWIN70/'
        # 2018-2020年的舆情信息
        for idx, line in enumerate(open(base_url + 'ent_news.csv', encoding='utf-8')):
            if idx == 0:
                cols = line.split('|')
            else:
                line_list = line.split('|')
                line_list = line_list[:8] + [''.join(line_list[8:]).replace('\n', '')]
                news_list.append(line_list)
        news_df = pd.DataFrame(news_list, columns=cols)
        # 2019-2020年的违约记录
        ent_default = pd.read_csv(base_url + 'ent_default.csv', sep='|')
        # 2018-2020年的财务指标数据
        ent_fina = pd.read_csv(base_url + 'ent_financial_indicator.csv', sep='|')
        # 企业的基本信息（只包含发债企业）
        ent_info = pd.read_csv(base_url + 'ent_info.csv', sep='|')
        # 答案
        answer = pd.read_csv(base_url + 'answer.csv', sep='|')

        # 时间处理（将yyyyMMdd转换成yyyy）
        ent_default['year'] = ent_default['acu_date'].apply(lambda x: x // 10000)
        ent_fina['year'] = ent_fina['report_period'].apply(lambda x: x // 10000)
        news_df['year'] = news_df['publishdate'].apply(lambda x: int(x) // 10000)

        ent_default['ent_id_year'] = ent_default['ent_id'] + '_' + (ent_default['year'] - 1).astype(str)  # 拼接历史数据
        ent_fina['ent_id_year'] = ent_fina['ent_id'] + '_' + ent_fina['year'].astype(str)
        news_df['ent_id_year'] = news_df['ent_id'] + '_' + news_df['year'].astype(str)
        answer['ent_id_year'] = answer['ent_id'].apply(lambda x: x + '_2020')

        del ent_fina['year'], news_df['year']

        # 去重
        ent_default_new = ent_default.drop_duplicates(subset=['ent_id_year'], keep='last')

        # 合并
        ent_default_new['default_score'] = 1
        answer['year'] = 2021  # 违约数据2019、2020，提交样例中测试数据是2021年
        # 将答案和违约记录拼接起来(行拼接)
        data = pd.concat([ent_default_new[['ent_id', 'ent_id_year', 'year', 'default_score']], answer], axis=0,
                         ignore_index=True)
        del ent_default_new

        print(data.shape)
        # 构建负样本
        ent_ids = [i for i in answer['ent_id'].unique() if
                   i not in ent_default[ent_default['year'] == 2019]['ent_id'].unique().tolist()]
        ent_ids_df = pd.DataFrame({'ent_id': ent_ids})
        ent_ids_df['year'] = 2019
        ent_ids_df['default_score'] = 0
        ent_ids_df['ent_id_year'] = ent_ids_df['ent_id'].apply(lambda x: x + '_2018')
        data = pd.concat([data, ent_ids_df], axis=0, ignore_index=True)

        ent_ids = [i for i in answer['ent_id'].unique() if
                   i not in ent_default[ent_default['year'] == 2020]['ent_id'].unique().tolist()]
        ent_ids_df = pd.DataFrame({'ent_id': ent_ids})
        ent_ids_df['year'] = 2020
        ent_ids_df['default_score'] = 0
        ent_ids_df['ent_id_year'] = ent_ids_df['ent_id'].apply(lambda x: x + '_2019')
        data = pd.concat([data, ent_ids_df], axis=0, ignore_index=True)
        print(data.shape)

        # 合并ent_info、ent_fina
        ent_info_new = ent_info.drop_duplicates()
        data = data.merge(ent_info_new, on=['ent_id'], how='left')

        ent_fina_new = ent_fina.sort_values('report_period').drop_duplicates(subset=['ent_id_year'], keep='last')
        data = data.merge(ent_fina_new, on=['ent_id', 'ent_id_year'], how='left')

        data = data.drop_duplicates()
        # print("合并后的维度为：", end="")
        # print(data.shape)
        # print(data.columns)
        return data

    def preprocess(self, df: DataFrame):
        # 特征工程
        # ent_info.csv(企业的基本信息)
        ent_info_cat_cols = ['industryphy', 'industryco', 'enttype', 'entstatus', 'prov', 'city', 'county',
                             'is_bondissuer']
        embedding = Embedding()
        df = embedding.text_embedding(df, ent_info_cat_cols)
        df = embedding.num_process(df, [])
        # 时间相关特征构建
        # data = df
    #     # data['opfrom_year'] = data['opfrom'].fillna('0000').apply(lambda x: int(x[:4]))
    #     # data['opto_year'] = data['opto'].fillna('0000').apply(lambda x: int(x[:4]))
    #     # data['esdate_year'] = data['esdate'].fillna('0000').apply(lambda x: int(x[:4]))
    #     # data['apprdate_year'] = data['apprdate'].fillna('0000').apply(lambda x: int(x[:4]))
    #     #
    #     # data.loc[data.opfrom.isnull(), 'opfrom'] = data.loc[data.opfrom.isnull(), 'esdate']
    #     # data.loc[data.apprdate.isnull(), 'apprdate'] = data.loc[data.apprdate.isnull(), 'esdate']
    #     #
    #     # # ['opfrom','optp','esdate','apprdate']
    #     # # 经营(驻在)期限自、经营(驻在)期限至、成立日期、核准日期
    #     # data['opfrom_esdate_diff'] = data['opfrom'].apply(lambda x: int(x[:4])) - data['esdate'].apply(
    #     #     lambda x: int(x[:4]))
    #     # data['apprdate_esdate_diff'] = data['apprdate'].apply(lambda x: int(x[:4])) - data['esdate'].apply(
    #     #     lambda x: int(x[:4]))
    #     #
    #     # # 预测年与经营(驻在)期限自、成立日期、核准日期、时间差
    #     # data['year_opfrom_diff'] = data['year'] - data['opfrom'].apply(lambda x: int(x[:4]))
    #     # data['year_esdate_diff'] = data['year'] - data['esdate'].apply(lambda x: int(x[:4]))
    #     # data['year_apprdate_diff'] = data['year'] - data['apprdate'].apply(lambda x: int(x[:4]))
    #     #
    #     # # 企业财务指标报告期
    #     # data['report_period_year'] = data['report_period'].apply(lambda x: x // 10000)
    #     # data['report_period_month'] = data['report_period'].apply(lambda x: x // 10000)
    #     #
    #     # tmp_df = news_df.groupby(['ent_id_year'])['newssource'].agg({list}).reset_index()
    #     # tmp_df['list'] = tmp_df['list'].apply(lambda x: ' '.join([i for i in x]))
    #     #
    #     # from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
    #     # from sklearn.decomposition import TruncatedSVD, SparsePCA
    #     # # TfidfVectorizer
    #     # tfidf = TfidfVectorizer()
    #     # tf = tfidf.fit_transform(tmp_df['list'].fillna("##").values)
    #     #
    #     # ###TfidfVectorizer
    #     # decom = TruncatedSVD(n_components=128, random_state=1024)
    #     # decom_x = decom.fit_transform(tf)
    #     # decom_feas = pd.DataFrame(decom_x)
    #     # decom_feas.columns = ['newssource_svd' + str(i) for i in range(decom_feas.shape[1])]
    #     # decom_feas['ent_id_year'] = tmp_df['ent_id_year']
    #     #
    #     # data = data.merge(decom_feas, on=['ent_id_year'], how='left')
    #     # print(data.shape)
        return df

    def train(self, df: DataFrame):
        print(df.shape)
        data = df

        # ent_info.csv(企业的基本信息)
        ent_info_time_cols = ['opfrom', 'opto', 'esdate', 'apprdate']  # 经营(驻在)期限自、经营(驻在)期限至、成立日期、核准日期
        # ent_financial_indicator.csv(企业的财务指标数据)
        ent_fina_time_cols = ['report_period']  # 报告期
        # 训练数据/测试数据准备
        features = [f for f in data.columns if f not in ['ent_id', 'ent_id_year', 'default_score', 'is_bondissuer'] + \
                    ent_info_time_cols + ent_fina_time_cols]
        print(features)
        # import random
        # features = random.sample(features, 50)

        train = data[data.year != 2021].reset_index(drop=True)
        test = data[data.year == 2021].reset_index(drop=True)
        x_train = train[features]
        x_test = test[features]
        y_train = train['default_score']

        lgb_train, lgb_test = self.xgb_model(x_train, y_train, x_test)
        return df

    # 建模
    def cv_model(self, clf, train_x, train_y, test_x, clf_name):
        folds = 5
        seed = 1234
        kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

        train = np.zeros(train_x.shape[0])
        test = np.zeros(test_x.shape[0])

        cv_scores = []

        for i, (train_index, valid_index) in enumerate(kf.split(train_x, train_y)):
            print('************************************ {} ************************************'.format(str(i + 1)))
            trn_x, trn_y, val_x, val_y = train_x.iloc[train_index], train_y[train_index], train_x.iloc[valid_index], \
                                         train_y[valid_index]

            if clf_name == "lgb":
                train_matrix = clf.Dataset(trn_x, label=trn_y)
                valid_matrix = clf.Dataset(val_x, label=val_y)

                params = {
                    'boosting_type': 'gbdt',
                    'objective': 'binary',
                    'metric': 'auc',
                    'min_child_weight': 5,
                    'num_leaves': 2 ** 5,
                    'lambda_l2': 10,
                    'feature_fraction': 0.8,
                    'bagging_fraction': 0.8,
                    'bagging_freq': 4,
                    'learning_rate': 0.01,
                    'seed': 1234,
                    'n_jobs': -1,
                    'verbose': -1,
                }

                model = clf.train(params, train_matrix, 50000, valid_sets=[train_matrix, valid_matrix],
                                  categorical_feature=[], verbose_eval=500, early_stopping_rounds=500)
                val_pred = model.predict(val_x, num_iteration=model.best_iteration)
                test_pred = model.predict(test_x, num_iteration=model.best_iteration)

            if clf_name == "xgb":
                train_matrix = clf.DMatrix(trn_x, label=trn_y)
                valid_matrix = clf.DMatrix(val_x, label=val_y)
                test_matrix = clf.DMatrix(test_x)

                params = {
                    'booster': 'gbtree',
                    'objective': 'binary:logistic',
                    'eval_metric': 'auc',
                    'gamma': 1,
                    'min_child_weight': 1.5,
                    'max_depth': 5,
                    'lambda': 10,
                    'subsample': 0.7,
                    'colsample_bytree': 0.7,
                    'colsample_bylevel': 0.7,
                    'eta': 0.01,
                    'tree_method': 'exact',
                    'seed': 1234,
                    'nthread': 36,
                }
                watchlist = [(train_matrix, 'train'), (valid_matrix, 'eval')]
                model = clf.train(params, train_matrix, num_boost_round=50000, evals=watchlist, verbose_eval=500,
                                  early_stopping_rounds=500)
                val_pred = model.predict(valid_matrix, ntree_limit=model.best_iteration)
                test_pred = model.predict(test_matrix, ntree_limit=model.best_iteration)

            if clf_name == "cat":
                params = {
                    'learning_rate': 0.01, 'depth': 5, 'l2_leaf_reg': 10, 'bootstrap_type': 'Bernoulli',
                    'od_type': 'Iter', 'od_wait': 50, 'random_seed': 11, 'allow_writing_files': False
                    # od参数防止过拟合
                    # l2_leaf_reg正则化前系数
                }
                model = clf(iterations=20000, **params)
                model.fit(trn_x, trn_y, eval_set=(val_x, val_y),
                          cat_features=[], use_best_model=True, verbose=500)
                val_pred = model.predict(val_x)
                test_pred = model.predict(test_x)

            train[valid_index] = val_pred
            test += test_pred / kf.n_splits
            cv_scores.append(roc_auc_score(val_y, val_pred))

            print(cv_scores)

        print("%s_scotrainre_list:" % clf_name, cv_scores)
        print("%s_score_mean:" % clf_name, np.mean(cv_scores))
        print("%s_score_std:" % clf_name, np.std(cv_scores))
        return train, test

    def xgb_model(self, x_train, y_train, x_test):
        xgb_train, xgb_test = self.cv_model(xgb, x_train, y_train, x_test, "xgb")
        return xgb_train, xgb_test

    # def lgb_model(self, x_train, y_train, x_test):
    #     lgb_train, lgb_test = self.cv_model(lgb, x_train, y_train, x_test, "lgb")
    #     return lgb_train, lgb_test

    # def cat_model(self, x_train, y_train, x_test):
    #     cat_train, cat_test = self.cv_model(CatBoostRegressor, x_train, y_train, x_test, "cat")
    #     return cat_train, cat_test
