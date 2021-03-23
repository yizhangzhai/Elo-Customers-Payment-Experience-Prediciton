import datetime
import gc
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import time
import warnings

from contextlib import contextmanager
from pandas.core.common import SettingWithCopyWarning
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold

warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

FEATS_EXCLUDED = ['first_active_month', 'target', 'card_id', 'outliers', 'index',
                  'hist_purchase_date_max', 'hist_purchase_date_min', 'hist_card_id_size',
                  'new_purchase_date_max', 'new_purchase_date_min', 'new_card_id_size',
                  'OOF_PRED', 'month_0']

def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

# rmse
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# One-hot encoding for categorical columns with get_dummies
def one_hot_encoder(df, nan_as_category = True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns

# Display/plot feature importance
def display_importances(feature_importance_df_):
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:40].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]

    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig('lgbm_importances.png')

# reduce memory
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df

# preprocessing train & test
def train_test(num_rows=None):

    # load csv
    train_df = pd.read_csv('train.csv', index_col=['card_id'], nrows=num_rows)
    test_df = pd.read_csv('test.csv', index_col=['card_id'], nrows=num_rows)

    print("Train samples: {}, test samples: {}".format(len(train_df), len(test_df)))

    # outlier
    train_df['outliers'] = 0
    train_df.loc[train_df['target'] < -30, 'outliers'] = 1

    # set target as nan
    test_df['target'] = np.nan

    # merge
    df = train_df.append(test_df)

    del train_df, test_df
    gc.collect()

    # to datetime
    df['first_active_month'] = pd.to_datetime(df['first_active_month'])

    # datetime features
    df['quarter'] = df['first_active_month'].dt.quarter
    df['elapsed_time'] = (datetime.datetime.today() - df['first_active_month']).dt.days

    df['days_feature1'] = df['elapsed_time'] * df['feature_1']
    df['days_feature2'] = df['elapsed_time'] * df['feature_2']
    df['days_feature3'] = df['elapsed_time'] * df['feature_3']

    df['days_feature1_ratio'] = df['feature_1'] / df['elapsed_time']
    df['days_feature2_ratio'] = df['feature_2'] / df['elapsed_time']
    df['days_feature3_ratio'] = df['feature_3'] / df['elapsed_time']

    # one hot encoding
    df, cols = one_hot_encoder(df, nan_as_category=False)

    for f in ['feature_1','feature_2','feature_3']:
        order_label = df.groupby([f])['outliers'].mean()
        df[f] = df[f].map(order_label)

    df['feature_sum'] = df['feature_1'] + df['feature_2'] + df['feature_3']
    df['feature_mean'] = df['feature_sum']/3
    df['feature_max'] = df[['feature_1', 'feature_2', 'feature_3']].max(axis=1)
    df['feature_min'] = df[['feature_1', 'feature_2', 'feature_3']].min(axis=1)
    df['feature_var'] = df[['feature_1', 'feature_2', 'feature_3']].std(axis=1)

    return df

# preprocessing historical transactions
def historical_transactions(num_rows=None):
    # load csv
    hist_df = pd.read_csv('historical_transactions.csv', nrows=num_rows)

    # fillna
    hist_df['category_2'].fillna(1.0,inplace=True)
    hist_df['category_3'].fillna('A',inplace=True)
    hist_df['merchant_id'].fillna('M_ID_00a6ca8a8a',inplace=True)
    hist_df['installments'].replace(-1, np.nan,inplace=True)
    hist_df['installments'].replace(999, np.nan,inplace=True)

    # trim
    hist_df['purchase_amount'] = hist_df['purchase_amount'].apply(lambda x: min(x, 0.8))

    # Y/N to 1/0
    hist_df['authorized_flag'] = hist_df['authorized_flag'].map({'Y': 1, 'N': 0}).astype(int)
    hist_df['category_1'] = hist_df['category_1'].map({'Y': 1, 'N': 0}).astype(int)
    hist_df['category_3'] = hist_df['category_3'].map({'A':0, 'B':1, 'C':2})

    # datetime features
    hist_df['purchase_date'] = pd.to_datetime(hist_df['purchase_date'])
    hist_df['month'] = hist_df['purchase_date'].dt.month
    hist_df['day'] = hist_df['purchase_date'].dt.day
    hist_df['hour'] = hist_df['purchase_date'].dt.hour
    hist_df['weekofyear'] = hist_df['purchase_date'].dt.weekofyear
    hist_df['weekday'] = hist_df['purchase_date'].dt.weekday
    hist_df['weekend'] = (hist_df['purchase_date'].dt.weekday >=5).astype(int)

    # additional features
    hist_df['price'] = hist_df['purchase_amount'] / hist_df['installments']

    #Christmas : December 25 2017
    hist_df['Christmas_Day_2017']=(pd.to_datetime('2017-12-25')-hist_df['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
    #Mothers Day: May 14 2017
    hist_df['Mothers_Day_2017']=(pd.to_datetime('2017-06-04')-hist_df['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
    #fathers day: August 13 2017
    hist_df['fathers_day_2017']=(pd.to_datetime('2017-08-13')-hist_df['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
    #Childrens day: October 12 2017
    hist_df['Children_day_2017']=(pd.to_datetime('2017-10-12')-hist_df['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
    #Valentine's Day : 12th June, 2017
    hist_df['Valentine_Day_2017']=(pd.to_datetime('2017-06-12')-hist_df['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
    #Black Friday : 24th November 2017
    hist_df['Black_Friday_2017']=(pd.to_datetime('2017-11-24') - hist_df['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)

    #2018
    #Mothers Day: May 13 2018
    hist_df['Mothers_Day_2018']=(pd.to_datetime('2018-05-13')-hist_df['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)

    hist_df['month_diff'] = ((datetime.datetime.today() - hist_df['purchase_date']).dt.days)//30
    hist_df['month_diff'] += hist_df['month_lag']

    # additional features
    hist_df['duration'] = hist_df['purchase_amount']*hist_df['month_diff']
    hist_df['amount_month_ratio'] = hist_df['purchase_amount']/hist_df['month_diff']

    # reduce memory usage
    hist_df = reduce_mem_usage(hist_df)

    col_unique =['subsector_id', 'merchant_id', 'merchant_category_id']
    col_seas = ['month', 'hour', 'weekofyear', 'weekday', 'day']

    aggs = {}
    for col in col_unique:
        aggs[col] = ['nunique']

    for col in col_seas:
        aggs[col] = ['nunique', 'mean', 'min', 'max']

    aggs['purchase_amount'] = ['sum','max','min','mean','var','skew']
    aggs['installments'] = ['sum','max','mean','var','skew']
    aggs['purchase_date'] = ['max','min']
    aggs['month_lag'] = ['max','min','mean','var','skew']
    aggs['month_diff'] = ['max','min','mean','var','skew']
    aggs['authorized_flag'] = ['mean']
    aggs['weekend'] = ['mean'] # overwrite
    aggs['weekday'] = ['mean'] # overwrite
    aggs['day'] = ['nunique', 'mean', 'min'] # overwrite
    aggs['category_1'] = ['mean']
    aggs['category_2'] = ['mean']
    aggs['category_3'] = ['mean']
    aggs['card_id'] = ['size','count']
    aggs['price'] = ['sum','mean','max','min','var']
    aggs['Christmas_Day_2017'] = ['mean']
    aggs['Mothers_Day_2017'] = ['mean']
    aggs['fathers_day_2017'] = ['mean']
    aggs['Children_day_2017'] = ['mean']
    aggs['Valentine_Day_2017'] = ['mean']
    aggs['Black_Friday_2017'] = ['mean']
    aggs['Mothers_Day_2018'] = ['mean']
    aggs['duration']=['mean','min','max','var','skew']
    aggs['amount_month_ratio']=['mean','min','max','var','skew']

    for col in ['category_2','category_3']:
        hist_df[col+'_mean'] = hist_df.groupby([col])['purchase_amount'].transform('mean')
        hist_df[col+'_min'] = hist_df.groupby([col])['purchase_amount'].transform('min')
        hist_df[col+'_max'] = hist_df.groupby([col])['purchase_amount'].transform('max')
        hist_df[col+'_sum'] = hist_df.groupby([col])['purchase_amount'].transform('sum')
        aggs[col+'_mean'] = ['mean']

    hist_df = hist_df.reset_index().groupby('card_id').agg(aggs)

    # change column name
    hist_df.columns = pd.Index([e[0] + "_" + e[1] for e in hist_df.columns.tolist()])
    hist_df.columns = ['hist_'+ c for c in hist_df.columns]

    hist_df['hist_purchase_date_diff'] = (hist_df['hist_purchase_date_max']-hist_df['hist_purchase_date_min']).dt.days
    hist_df['hist_purchase_date_average'] = hist_df['hist_purchase_date_diff']/hist_df['hist_card_id_size']
    hist_df['hist_purchase_date_uptonow'] = (datetime.datetime.today()-hist_df['hist_purchase_date_max']).dt.days
    hist_df['hist_purchase_date_uptomin'] = (datetime.datetime.today()-hist_df['hist_purchase_date_min']).dt.days

    # reduce memory usage
    hist_df = reduce_mem_usage(hist_df)

    return hist_df

# preprocessing new_merchant_transactions
def new_merchant_transactions(num_rows=None):
    # load csv
    new_merchant_df = pd.read_csv('new_merchant_transactions.csv', nrows=num_rows)

    # fillna
    new_merchant_df['category_2'].fillna(1.0,inplace=True)
    new_merchant_df['category_3'].fillna('A',inplace=True)
    new_merchant_df['merchant_id'].fillna('M_ID_00a6ca8a8a',inplace=True)
    new_merchant_df['installments'].replace(-1, np.nan,inplace=True)
    new_merchant_df['installments'].replace(999, np.nan,inplace=True)

    # trim
    new_merchant_df['purchase_amount'] = new_merchant_df['purchase_amount'].apply(lambda x: min(x, 0.8))

    # Y/N to 1/0
    new_merchant_df['authorized_flag'] = new_merchant_df['authorized_flag'].map({'Y': 1, 'N': 0}).astype(int)
    new_merchant_df['category_1'] = new_merchant_df['category_1'].map({'Y': 1, 'N': 0}).astype(int)
    new_merchant_df['category_3'] = new_merchant_df['category_3'].map({'A':0, 'B':1, 'C':2}).astype(int)

    # datetime features
    new_merchant_df['purchase_date'] = pd.to_datetime(new_merchant_df['purchase_date'])
    new_merchant_df['month'] = new_merchant_df['purchase_date'].dt.month
    new_merchant_df['day'] = new_merchant_df['purchase_date'].dt.day
    new_merchant_df['hour'] = new_merchant_df['purchase_date'].dt.hour
    new_merchant_df['weekofyear'] = new_merchant_df['purchase_date'].dt.weekofyear
    new_merchant_df['weekday'] = new_merchant_df['purchase_date'].dt.weekday
    new_merchant_df['weekend'] = (new_merchant_df['purchase_date'].dt.weekday >=5).astype(int)

    # additional features
    new_merchant_df['price'] = new_merchant_df['purchase_amount'] / new_merchant_df['installments']

    #Christmas : December 25 2017
    new_merchant_df['Christmas_Day_2017']=(pd.to_datetime('2017-12-25')-new_merchant_df['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
    #Childrens day: October 12 2017
    new_merchant_df['Children_day_2017']=(pd.to_datetime('2017-10-12')-new_merchant_df['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
    #Black Friday : 24th November 2017
    new_merchant_df['Black_Friday_2017']=(pd.to_datetime('2017-11-24') - new_merchant_df['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)

    #Mothers Day: May 13 2018
    new_merchant_df['Mothers_Day_2018']=(pd.to_datetime('2018-05-13')-new_merchant_df['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)

    new_merchant_df['month_diff'] = ((datetime.datetime.today() - new_merchant_df['purchase_date']).dt.days)//30
    new_merchant_df['month_diff'] += new_merchant_df['month_lag']

    # additional features
    new_merchant_df['duration'] = new_merchant_df['purchase_amount']*new_merchant_df['month_diff']
    new_merchant_df['amount_month_ratio'] = new_merchant_df['purchase_amount']/new_merchant_df['month_diff']

    # reduce memory usage
    new_merchant_df = reduce_mem_usage(new_merchant_df)

    col_unique =['subsector_id', 'merchant_id', 'merchant_category_id']
    col_seas = ['month', 'hour', 'weekofyear', 'weekday', 'day']

    aggs = {}
    for col in col_unique:
        aggs[col] = ['nunique']

    for col in col_seas:
        aggs[col] = ['nunique', 'mean', 'min', 'max']

    aggs['purchase_amount'] = ['sum','max','min','mean','var','skew']
    aggs['installments'] = ['sum','max','mean','var','skew']
    aggs['purchase_date'] = ['max','min']
    aggs['month_lag'] = ['max','min','mean','var','skew']
    aggs['month_diff'] = ['mean','var','skew']
    aggs['weekend'] = ['mean']
    aggs['month'] = ['mean', 'min', 'max']
    aggs['weekday'] = ['mean', 'min', 'max']
    aggs['category_1'] = ['mean']
    aggs['category_2'] = ['mean']
    aggs['category_3'] = ['mean']
    aggs['card_id'] = ['size','count']
    aggs['price'] = ['mean','max','min','var']
    aggs['Christmas_Day_2017'] = ['mean']
    aggs['Children_day_2017'] = ['mean']
    aggs['Black_Friday_2017'] = ['mean']
    aggs['Mothers_Day_2018'] = ['mean']
    aggs['duration']=['mean','min','max','var','skew']
    aggs['amount_month_ratio']=['mean','min','max','var','skew']

    for col in ['category_2','category_3']:
        new_merchant_df[col+'_mean'] = new_merchant_df.groupby([col])['purchase_amount'].transform('mean')
        new_merchant_df[col+'_min'] = new_merchant_df.groupby([col])['purchase_amount'].transform('min')
        new_merchant_df[col+'_max'] = new_merchant_df.groupby([col])['purchase_amount'].transform('max')
        new_merchant_df[col+'_sum'] = new_merchant_df.groupby([col])['purchase_amount'].transform('sum')
        aggs[col+'_mean'] = ['mean']

    new_merchant_df = new_merchant_df.reset_index().groupby('card_id').agg(aggs)

    # change column name
    new_merchant_df.columns = pd.Index([e[0] + "_" + e[1] for e in new_merchant_df.columns.tolist()])
    new_merchant_df.columns = ['new_'+ c for c in new_merchant_df.columns]

    new_merchant_df['new_purchase_date_diff'] = (new_merchant_df['new_purchase_date_max']-new_merchant_df['new_purchase_date_min']).dt.days
    new_merchant_df['new_purchase_date_average'] = new_merchant_df['new_purchase_date_diff']/new_merchant_df['new_card_id_size']
    new_merchant_df['new_purchase_date_uptonow'] = (datetime.datetime.today()-new_merchant_df['new_purchase_date_max']).dt.days
    new_merchant_df['new_purchase_date_uptomin'] = (datetime.datetime.today()-new_merchant_df['new_purchase_date_min']).dt.days

    # reduce memory usage
    new_merchant_df = reduce_mem_usage(new_merchant_df)

    return new_merchant_df

# additional features
def additional_features(df):
    df['hist_first_buy'] = (df['hist_purchase_date_min'] - df['first_active_month']).dt.days
    df['hist_last_buy'] = (df['hist_purchase_date_max'] - df['first_active_month']).dt.days
    df['new_first_buy'] = (df['new_purchase_date_min'] - df['first_active_month']).dt.days
    df['new_last_buy'] = (df['new_purchase_date_max'] - df['first_active_month']).dt.days

    date_features=['hist_purchase_date_max','hist_purchase_date_min',
                   'new_purchase_date_max', 'new_purchase_date_min']

    for f in date_features:
        df[f] = df[f].astype(np.int64) * 1e-9

    df['card_id_total'] = df['new_card_id_size']+df['hist_card_id_size']
    df['card_id_cnt_total'] = df['new_card_id_count']+df['hist_card_id_count']
    df['card_id_cnt_ratio'] = df['new_card_id_count']/df['hist_card_id_count']
    df['purchase_amount_total'] = df['new_purchase_amount_sum']+df['hist_purchase_amount_sum']
    df['purchase_amount_mean'] = df['new_purchase_amount_mean']+df['hist_purchase_amount_mean']
    df['purchase_amount_max'] = df['new_purchase_amount_max']+df['hist_purchase_amount_max']
    df['purchase_amount_min'] = df['new_purchase_amount_min']+df['hist_purchase_amount_min']
    df['purchase_amount_ratio'] = df['new_purchase_amount_sum']/df['hist_purchase_amount_sum']
    df['month_diff_mean'] = df['new_month_diff_mean']+df['hist_month_diff_mean']
    df['month_diff_ratio'] = df['new_month_diff_mean']/df['hist_month_diff_mean']
    df['month_lag_mean'] = df['new_month_lag_mean']+df['hist_month_lag_mean']
    df['month_lag_max'] = df['new_month_lag_max']+df['hist_month_lag_max']
    df['month_lag_min'] = df['new_month_lag_min']+df['hist_month_lag_min']
    df['category_1_mean'] = df['new_category_1_mean']+df['hist_category_1_mean']
    df['installments_total'] = df['new_installments_sum']+df['hist_installments_sum']
    df['installments_mean'] = df['new_installments_mean']+df['hist_installments_mean']
    df['installments_max'] = df['new_installments_max']+df['hist_installments_max']
    df['installments_ratio'] = df['new_installments_sum']/df['hist_installments_sum']
    df['price_total'] = df['purchase_amount_total'] / df['installments_total']
    df['price_mean'] = df['purchase_amount_mean'] / df['installments_mean']
    df['price_max'] = df['purchase_amount_max'] / df['installments_max']
    df['duration_mean'] = df['new_duration_mean']+df['hist_duration_mean']
    df['duration_min'] = df['new_duration_min']+df['hist_duration_min']
    df['duration_max'] = df['new_duration_max']+df['hist_duration_max']
    df['amount_month_ratio_mean']=df['new_amount_month_ratio_mean']+df['hist_amount_month_ratio_mean']
    df['amount_month_ratio_min']=df['new_amount_month_ratio_min']+df['hist_amount_month_ratio_min']
    df['amount_month_ratio_max']=df['new_amount_month_ratio_max']+df['hist_amount_month_ratio_max']
    df['new_CLV'] = df['new_card_id_count'] * df['new_purchase_amount_sum'] / df['new_month_diff_mean']
    df['hist_CLV'] = df['hist_card_id_count'] * df['hist_purchase_amount_sum'] / df['hist_month_diff_mean']
    df['CLV_ratio'] = df['new_CLV'] / df['hist_CLV']

    return df


def delta_days():
    ############ delta_purchase_days

    temp1 = historical_transactions.groupby('card_id')['purchase_date'].max().to_frame('hist_latest_purchase_date')
    temp2 = new_transactions.groupby('card_id')['purchase_date'].min().to_frame('new_earliest_purchase_date')

    delta_purchase_days = temp1.merge(temp2,how='left',left_index=True,right_index=True)
    delta_purchase_days["delta_purchase_days_1"] = delta_purchase_days.new_earliest_purchase_date - delta_purchase_days.hist_latest_purchase_date
    delta_purchase_days["delta_purchase_days_1"] = delta_purchase_days["delta_purchase_days_1"].dt.days
    delta_purchase_days["delta_purchase_days_1"].fillna(0,inplace=True)

    #
    temp1 = historical_transactions.groupby('card_id')['purchase_date'].max().to_frame('hist_latest_purchase_date')
    temp2 = new_transactions.groupby('card_id')['purchase_date'].max().to_frame('new_earliest_purchase_date')

    _delta_ = temp1.merge(temp2,how='left',left_index=True,right_index=True)
    delta_purchase_days["delta_purchase_days_2"] = _delta_.new_earliest_purchase_date - _delta_.hist_latest_purchase_date
    delta_purchase_days["delta_purchase_days_2"] = delta_purchase_days["delta_purchase_days_2"].dt.days
    delta_purchase_days["delta_purchase_days_2"].fillna(0,inplace=True)

    #
    temp1 = historical_transactions.groupby('card_id')['purchase_date'].min().to_frame('hist_latest_purchase_date')
    temp2 = new_transactions.groupby('card_id')['purchase_date'].min().to_frame('new_earliest_purchase_date')

    _delta_ = temp1.merge(temp2,how='left',left_index=True,right_index=True)
    delta_purchase_days["delta_purchase_days_3"] = _delta_.new_earliest_purchase_date - _delta_.hist_latest_purchase_date
    delta_purchase_days["delta_purchase_days_3"] = delta_purchase_days["delta_purchase_days_3"].dt.days
    delta_purchase_days["delta_purchase_days_3"].fillna(0,inplace=True)

    #
    temp1 = historical_transactions.groupby('card_id')['purchase_date_int'].mean().to_frame('hist_latest_purchase_date')
    temp2 = historical_transactions.groupby('card_id')['purchase_date_int'].max().to_frame('new_earliest_purchase_date')

    _delta_ = temp1.merge(temp2,how='left',left_index=True,right_index=True)
    delta_purchase_days["delta_purchase_days_6"] = _delta_.new_earliest_purchase_date - _delta_.hist_latest_purchase_date
    delta_purchase_days["delta_purchase_days_6"].fillna(0,inplace=True)

    #
    temp1 = new_transactions.groupby('card_id')['purchase_date_int'].mean().to_frame('hist_latest_purchase_date')
    temp2 = new_transactions.groupby('card_id')['purchase_date_int'].max().to_frame('new_earliest_purchase_date')

    _delta_ = temp1.merge(temp2,how='left',left_index=True,right_index=True)
    delta_purchase_days["delta_purchase_days_7"] = _delta_.new_earliest_purchase_date - _delta_.hist_latest_purchase_date
    delta_purchase_days["delta_purchase_days_7"].fillna(0,inplace=True)

    #
    temp1 = new_transactions.groupby('card_id')['purchase_date_int'].max().to_frame('hist_latest_purchase_date')
    temp2 = new_transactions.groupby('card_id')['purchase_date_int'].min().to_frame('new_earliest_purchase_date')

    _delta_ = temp1.merge(temp2,how='left',left_index=True,right_index=True)
    delta_purchase_days["delta_purchase_days_8"] = _delta_.new_earliest_purchase_date - _delta_.hist_latest_purchase_date
    delta_purchase_days["delta_purchase_days_8"].fillna(0,inplace=True)

    #
    temp1 = historical_transactions.groupby('card_id')['purchase_date_int'].max().to_frame('hist_latest_purchase_date')
    temp2 = historical_transactions.groupby('card_id')['purchase_date_int'].min().to_frame('new_earliest_purchase_date')

    _delta_ = temp1.merge(temp2,how='left',left_index=True,right_index=True)
    delta_purchase_days["delta_purchase_days_9"] = _delta_.new_earliest_purchase_date - _delta_.hist_latest_purchase_date
    delta_purchase_days["delta_purchase_days_9"].fillna(0,inplace=True)

    #
    temp1 = historical_transactions.groupby('card_id')['purchase_date_int'].var().to_frame('hist_latest_purchase_date')
    temp2 = new_transactions.groupby('card_id')['purchase_date_int'].var().to_frame('new_earliest_purchase_date')

    _delta_ = temp1.merge(temp2,how='left',left_index=True,right_index=True)
    delta_purchase_days["delta_purchase_days_10"] = _delta_.hist_latest_purchase_date
    delta_purchase_days["delta_purchase_days_11"] = _delta_.new_earliest_purchase_date

    #
    temp1 = historical_transactions.loc[historical_transactions.category_1==1].groupby('card_id')['purchase_date_int'].mean().to_frame('hist_latest_purchase_date')
    temp2 = historical_transactions.loc[historical_transactions.category_1==0].groupby('card_id')['purchase_date_int'].mean().to_frame('new_earliest_purchase_date')

    _delta_ = temp1.merge(temp2,how='left',left_index=True,right_index=True)
    delta_purchase_days["delta_purchase_days_12"] = _delta_.hist_latest_purchase_date - _delta_.new_earliest_purchase_date


    return delta_purchase_days

def recency():
    historical_transactions['category_1'] = historical_transactions['category_1'].map({'N':0,'Y':1})
    new_transactions['category_1'] = new_transactions['category_1'].map({'N':0,'Y':1})
    historical_transactions['authorized_flag'] = historical_transactions['authorized_flag'].map({'N':0,'Y':1})
    new_transactions['authorized_flag'] = new_transactions['authorized_flag'].map({'N':0,'Y':1})

    agg_card = {
                'card_id':['count'],
                'purchase_amount':['min','max','mean','std','sum','nunique','median'],
                'month_lag':['mean','nunique','min','max','sum','median'],
                'authorized_flag':['sum','mean'],
                'category_1':['sum','mean','nunique'],
                'merchant_id':['nunique'],
                'city_id':['nunique'],
                'purchase_date_int':['mean','max','sum','min'],
                'installments':['sum','mean','nunique','median'],
                }
    # Recency
    hist_agg_card_Recency = historical_transactions.loc[historical_transactions.month_lag>-1].groupby('card_id').agg(agg_card)
    hist_agg_card_Recency.columns = ['hist_agg_card_Recency_'+a+'_'+b for a,b in hist_agg_card_Recency.columns]
    hist_agg_card_Recency.fillna(0,inplace=True)

    new_agg_card_Recency = new_transactions.loc[new_transactions.month_lag<2].groupby('card_id').agg(agg_card)
    new_agg_card_Recency.columns = ['new_agg_card_Recency_'+a+'_'+b for a,b in new_agg_card_Recency.columns]
    new_agg_card_Recency.fillna(0,inplace=True)

    hist_agg_card_Recency_12_mont_lag = historical_transactions.groupby('card_id')['month_lag'].apply(lambda x: x.tail(12).mean())
    hist_agg_card_Recency_12_mont_lag = hist_agg_card_Recency_12_mont_lag.to_frame('hist_agg_card_Recency_12_mont_lag')

    temp = hist_agg_card_Recency.merge(new_agg_card_Recency,how='outer',left_index=True,right_index=True).merge(hist_agg_card_Recency_12_mont_lag,how='outer',left_index=True,right_index=True)

    return temp


def aggs():
    historical_transactions['days_diff'] = ((datetime.datetime.today() - historical_transactions['purchase_date']).dt.days)
    new_transactions['days_diff'] = ((datetime.datetime.today() - new_transactions['purchase_date']).dt.days)

    agg4_hist = historical_transactions.groupby(['card_id','merchant_id'])['days_diff'].nunique().to_frame('agg4_hist')
    agg4_new = new_transactions.groupby(['card_id','merchant_id'])['days_diff'].nunique().to_frame('agg4_new')

    agg4_hist = agg4_hist.groupby('card_id')['agg4_hist'].agg(['mean','var'])
    agg4_hist.columns = ['agg4_hist_' + a for a in agg4_hist.columns]
    agg4_new = agg4_new.groupby('card_id')['agg4_new'].agg(['mean','var'])
    agg4_new.columns = ['agg4_new_' + a for a in agg4_new.columns]

    ######################################## agg 5 ##################################################################################
    agg5_hist = historical_transactions.groupby(['card_id','merchant_id'])['days_diff'].median().to_frame('agg5_hist')
    agg5_new = new_transactions.groupby(['card_id','merchant_id'])['days_diff'].median().to_frame('agg5_new')

    agg5_hist = agg5_hist.groupby('card_id')['agg5_hist'].agg(['mean','var'])
    agg5_hist.columns = ['agg5_hist_' + a for a in agg5_hist.columns]

    agg5_new = agg5_new.groupby('card_id')['agg5_new'].agg(['mean','var'])
    agg5_new.columns = ['agg5_new_' + a for a in agg5_new.columns]

    ######################################## agg 6 ##################################################################################
    agg6_hist = historical_transactions.groupby(['card_id','merchant_id'])['purchase_amount'].sum().to_frame('agg6_hist')
    agg6_new = new_transactions.groupby(['card_id','merchant_id'])['purchase_amount'].sum().to_frame('agg6_new')

    agg6_hist = agg6_hist.groupby('card_id')['agg6_hist'].agg(['mean','var'])
    agg6_hist.columns = ['agg6_hist_' + a for a in agg6_hist.columns]

    agg6_new = agg6_new.groupby('card_id')['agg6_new'].agg(['mean','var'])
    agg6_new.columns = ['agg6_new_' + a for a in agg6_new.columns]

    ######################################## agg 7 ##################################################################################
    agg7_hist = historical_transactions.groupby(['card_id','city_id'])['purchase_amount'].var().to_frame('agg7_hist')
    agg7_new = new_transactions.groupby(['card_id','city_id'])['purchase_amount'].var().to_frame('agg7_new')

    agg7_hist = agg7_hist.groupby('card_id')['agg7_hist'].agg(['mean','var'])
    agg7_hist.columns = ['agg7_hist_' + a for a in agg7_hist.columns]

    agg7_new = agg7_new.groupby('card_id')['agg7_new'].agg(['mean','var'])
    agg7_new.columns = ['agg7_new_' + a for a in agg7_new.columns]

    ######################################## agg 8 ##################################################################################
    agg8_hist = historical_transactions.groupby(['card_id','month_lag'])['purchase_amount'].mean().to_frame('agg8_hist')
    agg8_new = new_transactions.groupby(['card_id','month_lag'])['purchase_amount'].mean().to_frame('agg8_new')

    agg8_hist = agg8_hist.groupby('card_id')['agg8_hist'].agg(['mean','var'])
    agg8_hist.columns = ['agg8_hist_' + a for a in agg8_hist.columns]

    agg8_new = agg8_new.groupby('card_id')['agg8_new'].agg(['mean','var'])
    agg8_new.columns = ['agg8_new_' + a for a in agg8_new.columns]

    ######################################## agg 9 ##################################################################################
    agg9_hist = historical_transactions.groupby(['card_id','category_1'])['category_2'].nunique().to_frame('agg9_hist')
    agg9_new = new_transactions.groupby(['card_id','category_1'])['category_2'].nunique().to_frame('agg9_new')

    agg9_hist = agg9_hist.groupby('card_id')['agg9_hist'].agg(['mean','var'])
    agg9_hist.columns = ['agg9_hist_' + a for a in agg9_hist.columns]

    agg9_new = agg9_new.groupby('card_id')['agg9_new'].agg(['mean','var'])
    agg9_new.columns = ['agg9_new_' + a for a in agg9_new.columns]

    ######################################## agg 10 ##################################################################################
    agg10_hist = historical_transactions.groupby(['card_id','category_1'])['purchase_amount'].agg(['mean','min','max','sum','var']).unstack()
    agg10_new = new_transactions.groupby(['card_id','category_1'])['purchase_amount'].agg(['mean','min','max','sum','var']).unstack()

    agg10_hist.columns = ['agg10_hist_' + str(a)+'_'+str(b) for a,b in agg10_hist.columns]
    agg10_new.columns = ['agg10_new_' + str(a) +'_'+str(b) for a,b in agg10_new.columns]

    ######################################## agg 11 ##################################################################################
    historical_transactions['month_diff'] = ((datetime.datetime.today() - historical_transactions['purchase_date']).dt.days)//30
    historical_transactions['month_diff'] += historical_transactions['month_lag']
    new_transactions['month_diff'] = ((datetime.datetime.today() - historical_transactions['purchase_date']).dt.days)//30
    new_transactions['month_diff'] += new_transactions['month_lag']

    agg11_hist = historical_transactions.groupby(['card_id','category_1'])['month_diff'].agg(['mean','min','max','sum','var']).unstack()
    agg11_new = new_transactions.groupby(['card_id','category_1'])['month_diff'].agg(['mean','min','max','sum','var']).unstack()

    agg11_hist.columns = ['agg11_hist_' + str(a)+'_'+str(b) for a,b in agg11_hist.columns]
    agg11_new.columns = ['agg11_new_' + str(a) +'_'+str(b) for a,b in agg11_new.columns]


    aggs = agg4_hist.merge(agg4_new,how='outer',left_index=True,right_index=True)\
                    .merge(agg5_hist,how='outer',left_index=True,right_index=True).merge(agg5_new,how='outer',left_index=True,right_index=True)\
                    .merge(agg6_hist,how='outer',left_index=True,right_index=True).merge(agg6_new,how='outer',left_index=True,right_index=True)\
                    .merge(agg7_hist,how='outer',left_index=True,right_index=True).merge(agg7_new,how='outer',left_index=True,right_index=True)\
                    .merge(agg8_hist,how='outer',left_index=True,right_index=True).merge(agg8_new,how='outer',left_index=True,right_index=True)\
                    .merge(agg9_hist,how='outer',left_index=True,right_index=True).merge(agg9_new,how='outer',left_index=True,right_index=True)\
                    .merge(agg10_hist,how='outer',left_index=True,right_index=True).merge(agg10_new,how='outer',left_index=True,right_index=True)\
                    .merge(agg11_hist,how='outer',left_index=True,right_index=True).merge(agg11_new,how='outer',left_index=True,right_index=True)

    return aggs


def aggs_plus():
    historical_transactions['days_diff'] = ((datetime.datetime.today() - historical_transactions['purchase_date']).dt.days)
    new_transactions['days_diff'] = ((datetime.datetime.today() - new_transactions['purchase_date']).dt.days)

    historical_transactions['month'] = historical_transactions['purchase_date'].dt.month
    historical_transactions['day'] = historical_transactions['purchase_date'].dt.day
    historical_transactions['hour'] = historical_transactions['purchase_date'].dt.hour
    historical_transactions['weekofyear'] = historical_transactions['purchase_date'].dt.weekofyear
    historical_transactions['weekday'] = historical_transactions['purchase_date'].dt.weekday
    historical_transactions['weekend'] = (historical_transactions['purchase_date'].dt.weekday >=5).astype(int)

    new_transactions['month'] = new_transactions['purchase_date'].dt.month
    new_transactions['day'] = new_transactions['purchase_date'].dt.day
    new_transactions['hour'] = new_transactions['purchase_date'].dt.hour
    new_transactions['weekofyear'] = new_transactions['purchase_date'].dt.weekofyear
    new_transactions['weekday'] = new_transactions['purchase_date'].dt.weekday
    new_transactions['weekend'] = (new_transactions['purchase_date'].dt.weekday >=5).astype(int)
    #######################################################################################################################
    agg12_hist = historical_transactions.groupby(['card_id','authorized_flag'])['days_diff'].agg(['mean','min','max','sum','var']).unstack()
    agg12_new = new_transactions.groupby(['card_id','authorized_flag'])['days_diff'].agg(['mean','min','max','sum','var']).unstack()

    agg12_hist.columns = ['agg12_hist_' + str(a)+'_'+str(b) for a,b in agg12_hist.columns]
    agg12_new.columns = ['agg12_new_' + str(a) +'_'+str(b) for a,b in agg12_new.columns]

    #######################################################################################################################
    agg13_hist = historical_transactions.groupby(['card_id','authorized_flag'])['weekofyear'].agg(['mean','min','max','sum','var']).unstack()
    agg13_new = new_transactions.groupby(['card_id','authorized_flag'])['weekofyear'].agg(['mean','min','max','sum','var']).unstack()

    agg13_hist.columns = ['agg13_hist_' + str(a)+'_'+str(b) for a,b in agg13_hist.columns]
    agg13_new.columns = ['agg13_new_' + str(a) +'_'+str(b) for a,b in agg13_new.columns]

    #######################################################################################################################
    agg14_hist = historical_transactions.groupby(['card_id','authorized_flag'])['purchase_amount'].agg(['mean','min','max','sum','var']).unstack()
    agg14_new = new_transactions.groupby(['card_id','authorized_flag'])['purchase_amount'].agg(['mean','min','max','sum','var']).unstack()

    agg14_hist.columns = ['agg14_hist_' + str(a)+'_'+str(b) for a,b in agg14_hist.columns]
    agg14_new.columns = ['agg14_new_' + str(a) +'_'+str(b) for a,b in agg14_new.columns]

    agg_plus = agg12_hist.merge(agg12_new,how='outer',left_index=True,right_index=True)\
                    .merge(agg13_hist,how='outer',left_index=True,right_index=True).merge(agg13_new,how='outer',left_index=True,right_index=True)\
                    .merge(agg14_hist,how='outer',left_index=True,right_index=True).merge(agg14_new,how='outer',left_index=True,right_index=True)\

    return agg_plus

def agg_plus_2():
        ######################################## agg 15 ##################################################################################
        agg15_hist = historical_transactions.groupby(['card_id','category_1'])['weekday'].agg(['mean','min','max','sum','nunique']).unstack()
        agg15_new = new_transactions.groupby(['card_id','category_1'])['weekday'].agg(['mean','min','max','sum','nunique']).unstack()

        agg15_hist.columns = ['agg15_hist_' + str(a)+'_'+str(b) for a,b in agg15_hist.columns]
        agg15_new.columns = ['agg15_new_' + str(a) +'_'+str(b) for a,b in agg15_new.columns]

        ######################################## agg 16 ##################################################################################
        agg16_hist = historical_transactions.groupby(['card_id','category_1'])['installments'].sum().to_frame('agg16_hist')
        agg16_new = new_transactions.groupby(['card_id','category_1'])['installments'].sum().to_frame('agg16_new')

        agg16_hist = agg16_hist.groupby('card_id')['agg16_hist'].agg(['mean','var'])
        agg16_hist.columns = ['agg16_hist_' + a for a in agg16_hist.columns]

        agg16_new = agg16_new.groupby('card_id')['agg16_new'].agg(['mean','var'])
        agg16_new.columns = ['agg16_new_' + a for a in agg16_new.columns]

        ####################################### merchants ##################################################################################
        #
        # temp = historical_transactions.groupby(['merchant_id'])['purchase_amount'].median().to_frame('merchant_median_purchase_amount')
        # hist_purchase = historical_transactions[['card_id','merchant_id','purchase_amount']].merge(temp.reset_index(),how='inner',on='merchant_id')
        # hist_purchase['purchase_amount_more'] = [1 if x>=y else 0 for x,y in hist_purchase[['purchase_amount','merchant_median_purchase_amount']].values]
        #
        # hist_purchase = hist_purchase.groupby('card_id').agg({'merchant_median_purchase_amount':['mean','var','sum','min','max','nunique'],\
        #                                     'purchase_amount_more':['mean','var','sum']})
        # hist_purchase.columns = [a+'_'+b for a,b in hist_purchase.columns]
        #
        #
        # temp = new_transactions.groupby(['merchant_id'])['purchase_amount'].median().to_frame('merchant_median_purchase_amount')
        # new_purchase = new_transactions[['card_id','merchant_id','purchase_amount']].merge(temp.reset_index(),how='inner',on='merchant_id')
        # new_purchase['purchase_amount_more'] = [1 if x>=y else 0 for x,y in new_purchase[['purchase_amount','merchant_median_purchase_amount']].values]
        #
        # new_purchase = new_purchase.groupby('card_id').agg({'merchant_median_purchase_amount':['mean','var','sum','min','max','nunique'],\
        #                                     'purchase_amount_more':['mean','var','sum']})
        # new_purchase.columns = [a+'_'+b for a,b in new_purchase.columns]


        ####################################### agg 17 ##################################################################################
        agg17_hist = historical_transactions.groupby(['card_id','category_1'])['days_diff'].agg(['mean','min','max','sum','var']).unstack()
        agg17_new = new_transactions.groupby(['card_id','category_1'])['days_diff'].agg(['mean','min','max','sum','var']).unstack()

        agg17_hist.columns = ['agg17_hist_' + str(a)+'_'+str(b) for a,b in agg17_hist.columns]
        agg17_new.columns = ['agg17_new_' + str(a) +'_'+str(b) for a,b in agg17_new.columns]

        ####################################### agg 18 ##################################################################################
        agg18_hist = historical_transactions.groupby(['card_id','category_3','authorized_flag'])['purchase_amount'].max().to_frame('agg18_hist')
        agg18_new = new_transactions.groupby(['card_id','category_3','authorized_flag'])['purchase_amount'].max().to_frame('agg18_new')

        agg18_hist = agg18_hist.groupby('card_id')['agg18_hist'].agg(['mean','min','max','sum','var'])
        agg18_hist.columns = ['agg18_hist_' + a for a in agg18_hist.columns]

        agg18_new = agg18_new.groupby('card_id')['agg18_new'].agg(['mean','min','max','sum','var'])
        agg18_new.columns = ['agg18_new_' + a for a in agg18_new.columns]

        ######################################## agg 19 ##################################################################################
        agg19_hist = historical_transactions.groupby(['card_id','authorized_flag','merchant_id'])['days_diff'].mean().to_frame('agg19_hist')
        agg19_new = new_transactions.groupby(['card_id','authorized_flag','merchant_id'])['days_diff'].mean().to_frame('agg19_new')

        agg19_hist = agg19_hist.groupby('card_id')['agg19_hist'].agg(['mean','min','max','sum','var'])
        agg19_hist.columns = ['agg19_hist_' + a for a in agg19_hist.columns]

        agg19_new = agg19_new.groupby('card_id')['agg19_new'].agg(['mean','min','max','sum','var'])
        agg19_new.columns = ['agg19_new_' + a for a in agg19_new.columns]
        ######################################## agg 20 ##################################################################################
        agg20_hist = historical_transactions.groupby(['card_id','category_2','merchant_id'])['month_diff'].mean().to_frame('agg20_hist')
        agg20_new = new_transactions.groupby(['card_id','category_2','merchant_id'])['month_diff'].mean().to_frame('agg20_new')

        agg20_hist = agg20_hist.groupby('card_id')['agg20_hist'].agg(['mean','min','max','sum','var'])
        agg20_hist.columns = ['agg20_hist_' + a for a in agg20_hist.columns]

        agg20_new = agg20_new.groupby('card_id')['agg20_new'].agg(['mean','min','max','sum','var'])
        agg20_new.columns = ['agg20_new_' + a for a in agg20_new.columns]

        ######################################## agg 21 ##################################################################################
        agg21_hist = historical_transactions.groupby(['card_id','category_3','merchant_id'])['days_diff'].sum().to_frame('agg21_hist')
        agg21_new = new_transactions.groupby(['card_id','category_3','merchant_id'])['days_diff'].sum().to_frame('agg21_new')

        agg21_hist = agg21_hist.groupby('card_id')['agg21_hist'].agg(['mean','min','max','sum','var'])
        agg21_hist.columns = ['agg21_hist_' + a for a in agg21_hist.columns]

        agg21_new = agg21_new.groupby('card_id')['agg21_new'].agg(['mean','min','max','sum','var'])
        agg21_new.columns = ['agg21_new_' + a for a in agg21_new.columns]

        ######################################## agg 22 ##################################################################################
        agg22_hist = historical_transactions.groupby(['card_id','category_3','merchant_id'])['month_diff'].nunique().to_frame('agg22_hist')
        agg22_new = new_transactions.groupby(['card_id','category_3','merchant_id'])['month_diff'].nunique().to_frame('agg22_new')

        agg22_hist = agg22_hist.groupby('card_id')['agg22_hist'].agg(['mean','min','max','sum','var'])
        agg22_hist.columns = ['agg22_hist_' + a for a in agg22_hist.columns]

        agg22_new = agg22_new.groupby('card_id')['agg22_new'].agg(['mean','min','max','sum','var'])
        agg22_new.columns = ['agg22_new_' + a for a in agg22_new.columns]

        ######################################## agg 23 ##################################################################################
        agg23_hist = historical_transactions.groupby(['card_id','category_3','merchant_id'])['purchase_amount'].nunique().to_frame('agg23_hist')
        agg23_new = new_transactions.groupby(['card_id','category_3','merchant_id'])['purchase_amount'].nunique().to_frame('agg23_new')

        agg23_hist = agg23_hist.groupby('card_id')['agg23_hist'].agg(['mean','min','max','sum','var'])
        agg23_hist.columns = ['agg23_hist_' + a for a in agg23_hist.columns]

        agg23_new = agg23_new.groupby('card_id')['agg23_new'].agg(['mean','min','max','sum','var'])
        agg23_new.columns = ['agg23_new_' + a for a in agg23_new.columns]

        ######################################## agg 24 ##################################################################################
        agg24_hist = historical_transactions.groupby(['card_id','city_id','merchant_id'])['purchase_amount'].min().to_frame('agg24_hist')
        agg24_new = new_transactions.groupby(['card_id','city_id','merchant_id'])['purchase_amount'].min().to_frame('agg24_new')

        agg24_hist = agg24_hist.groupby('card_id')['agg24_hist'].agg(['mean','min','max','sum','var'])
        agg24_hist.columns = ['agg24_hist_' + a for a in agg24_hist.columns]

        agg24_new = agg24_new.groupby('card_id')['agg24_new'].agg(['mean','min','max','sum','var'])
        agg24_new.columns = ['agg24_new_' + a for a in agg24_new.columns]

        ######################################## agg 25 ##################################################################################
        agg25_hist = historical_transactions.groupby(['card_id','category_1'])['weekofyear'].agg(['mean','min','max','sum','nunique']).unstack()
        agg25_new = new_transactions.groupby(['card_id','category_1'])['weekofyear'].agg(['mean','min','max','sum','nunique']).unstack()

        agg25_hist.columns = ['agg25_hist_' + str(a)+'_'+str(b) for a,b in agg25_hist.columns]
        agg25_new.columns = ['agg25_new_' + str(a) +'_'+str(b) for a,b in agg25_new.columns]

        ######################################## agg 25 ##################################################################################
        agg26_hist = historical_transactions.groupby(['card_id','category_1'])['month'].agg(['mean','min','max','sum','nunique']).unstack()
        agg26_new = new_transactions.groupby(['card_id','category_1'])['month'].agg(['mean','min','max','sum','nunique']).unstack()

        agg26_hist.columns = ['agg26_hist_' + str(a)+'_'+str(b) for a,b in agg26_hist.columns]
        agg26_new.columns = ['agg26_new_' + str(a) +'_'+str(b) for a,b in agg26_new.columns]

        ###########################################################################################################
        agg_plus_2 = agg15_hist.merge(agg15_new,how='outer',left_index=True,right_index=True)\
                        .merge(agg16_hist,how='outer',left_index=True,right_index=True).merge(agg16_new,how='outer',left_index=True,right_index=True)\
                        .merge(agg17_hist,how='outer',left_index=True,right_index=True).merge(agg17_new,how='outer',left_index=True,right_index=True)\
                        .merge(agg18_hist,how='outer',left_index=True,right_index=True).merge(agg18_new,how='outer',left_index=True,right_index=True)\
                        .merge(agg19_hist,how='outer',left_index=True,right_index=True).merge(agg19_new,how='outer',left_index=True,right_index=True)\
                        .merge(agg21_hist,how='outer',left_index=True,right_index=True).merge(agg21_new,how='outer',left_index=True,right_index=True)\
                        .merge(agg20_hist,how='outer',left_index=True,right_index=True).merge(agg20_new,how='outer',left_index=True,right_index=True)\
                        .merge(agg22_hist,how='outer',left_index=True,right_index=True).merge(agg22_new,how='outer',left_index=True,right_index=True)\
                        .merge(agg23_hist,how='outer',left_index=True,right_index=True).merge(agg23_new,how='outer',left_index=True,right_index=True)\
                        .merge(agg24_hist,how='outer',left_index=True,right_index=True).merge(agg24_new,how='outer',left_index=True,right_index=True)\
                        .merge(agg25_hist,how='outer',left_index=True,right_index=True).merge(agg25_new,how='outer',left_index=True,right_index=True)\
                        .merge(agg26_hist,how='outer',left_index=True,right_index=True).merge(agg26_new,how='outer',left_index=True,right_index=True)\

        return agg_plus_2


# LightGBM GBDT with KFold or Stratified KFold
def kfold_lightgbm(train_df, test_df, num_folds, stratified = False, debug= False):

    keeping=['agg11_hist_mean_1', 'hist_month_nunique', 'delta_purchase_days_2',
           'agg11_hist_max_1', 'new_amount_month_ratio_max',
           'agg11_hist_sum_1', 'agg10_hist_sum_1', 'hist_duration_min',
           'hist_purchase_date_diff', 'new_month_lag_mean',
           'hist_weekofyear_nunique', 'hist_authorized_flag_mean',
           'hist_month_diff_max',
           'new_agg_card_Recency_purchase_date_int_mean',
           'hist_month_diff_mean', 'agg14_hist_sum_0', 'agg14_new_max_1',
           'agg12_hist_sum_0', 'agg4_hist_mean', 'agg13_hist_sum_0',
           'hist_installments_sum', 'agg5_new_var', 'delta_purchase_days_11',
           'new_purchase_amount_max', 'new_day_mean', 'new_duration_max',
           'hist_category_1_mean', 'agg10_hist_min_1', 'category_1_mean',
           'agg11_hist_min_1', 'agg11_hist_mean_0', 'agg11_new_sum_1',
           'hist_agg_card_Recency_category_1_nunique',
           'hist_agg_card_Recency_purchase_amount_sum', 'agg11_hist_sum_0',
           'agg10_new_sum_1', 'agg13_new_var_1', 'agg12_new_sum_1',
           'hist_agg_card_Recency_card_id_count',
           'hist_agg_card_Recency_purchase_date_int_mean', 'hist_price_var',
           'hist_purchase_date_average',
           'new_agg_card_Recency_purchase_date_int_max',
           'hist_fathers_day_2017_mean',
           'hist_agg_card_Recency_installments_sum',
           'hist_agg_card_Recency_purchase_date_int_max', 'agg12_new_var_1',
           'hist_agg_card_Recency_purchase_amount_max', 'agg10_new_min_1',
           'agg11_hist_var_0', 'delta_purchase_days_8', 'agg14_hist_min_0',
           'hist_duration_mean', 'hist_hour_mean', 'agg12_hist_var_1',
           'agg12_hist_var_0', 'agg4_hist_var',
           'new_agg_card_Recency_purchase_amount_max', 'card_id_cnt_ratio',
           'hist_last_buy', 'agg11_hist_var_1', 'new_day_max',
           'hist_Children_day_2017_mean', 'new_purchase_date_uptonow',
           'agg14_hist_max_0', 'agg5_hist_var', 'agg10_hist_max_0',
           'hist_first_buy', 'agg10_hist_mean_1', 'days_feature2',
           'agg10_hist_var_1', 'hist_agg_card_Recency_authorized_flag_mean',
           'new_purchase_date_diff', 'hist_Valentine_Day_2017_mean',
           'hist_month_lag_var', 'new_agg_card_Recency_purchase_amount_std',
           'hist_merchant_id_nunique', 'agg14_hist_var_0', 'agg7_hist_var',
           'agg10_hist_max_1', 'agg12_hist_max_0', 'days_feature1',
           'month_diff_ratio', 'hist_weekend_mean', 'agg14_hist_max_1',
           'hist_agg_card_Recency_purchase_amount_median',
           'new_weekofyear_max', 'hist_weekday_mean', 'agg10_hist_min_0',
           'agg13_hist_var_1', 'hist_agg_card_Recency_category_1_sum',
           'new_weekofyear_mean', 'agg10_hist_sum_0', 'new_category_1_mean',
           'hist_month_lag_skew', 'new_month_lag_max', 'agg6_hist_mean',
           'agg13_hist_var_0', 'hist_agg_card_Recency_purchase_amount_mean',
           'agg14_new_var_1', 'new_agg_card_Recency_purchase_amount_sum',
           'hist_agg_card_Recency_merchant_id_nunique',
           'hist_subsector_id_nunique', 'new_category_3_mean_mean',
           'new_purchase_date_average', 'hist_month_lag_mean',
           'hist_agg_card_Recency_purchase_amount_min',
           'hist_agg_card_Recency_purchase_amount_nunique',
           'agg12_hist_min_0', 'agg12_hist_max_1', 'days_feature1_ratio',
           'hist_price_sum', 'new_month_mean', 'agg11_new_mean_1',
           'hist_Mothers_Day_2017_mean', 'hist_day_mean',
           'installments_ratio', 'new_duration_var', 'agg10_hist_mean_0',
           'agg12_hist_min_1', 'hist_merchant_category_id_nunique',
           'agg6_hist_var', 'agg10_new_max_0',
           'hist_agg_card_Recency_installments_mean', 'new_last_buy',
           'agg14_hist_mean_1', 'agg11_new_sum_0', 'agg10_hist_var_0',
           'agg14_hist_mean_0', 'hist_month_diff_var', 'new_hour_mean',
           'hist_purchase_amount_skew', 'agg10_new_mean_0',
           'hist_amount_month_ratio_skew', 'hist_amount_month_ratio_min',
           'hist_CLV', 'hist_agg_card_Recency_category_1_mean',
           'hist_price_mean', 'hist_purchase_amount_sum', 'month_lag_mean',
           'new_agg_card_Recency_purchase_amount_median', 'agg14_hist_min_1',
           'new_first_buy', 'new_amount_month_ratio_var',
           'hist_month_diff_skew', 'agg13_hist_sum_1',
           'hist_agg_card_Recency_12_mont_lag', 'hist_duration_skew',
           'hist_category_3_mean', 'new_agg_card_Recency_purchase_amount_min',
           'new_price_var', 'hist_agg_card_Recency_purchase_amount_std',
           'agg13_new_sum_1', 'agg5_hist_mean',
           'new_agg_card_Recency_purchase_amount_mean', 'agg12_hist_mean_0',
           'agg6_new_mean', 'hist_category_3_mean_mean', 'agg14_hist_var_1',
           'hist_purchase_date_uptonow', 'month_lag_max',
           'hist_installments_mean', 'agg8_hist_mean', 'agg6_new_var',
           'hist_Black_Friday_2017_mean', 'agg10_new_min_0',
           'agg12_hist_mean_1', 'new_day_min', 'hist_installments_var',
           'hist_installments_skew', 'new_Christmas_Day_2017_mean',
           'installments_total', 'new_month_max', 'CLV_ratio',
           'new_weekofyear_min', 'delta_purchase_days_7',
           'hist_purchase_date_uptomin', 'agg12_new_min_1', 'new_price_min',
           'agg10_new_sum_0', 'agg8_hist_var', 'purchase_amount_ratio']


    # Cross validation model
    if stratified:
        folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=15)
    else:
        folds = KFold(n_splits= num_folds, shuffle=True, random_state=15)

    # Create arrays and dataframes to store results
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in train_df.columns if f not in FEATS_EXCLUDED]
    feats = [c for c in feats if c in keeping]

    from itertools import chain
    feats = list(chain(*[feats,new_vars.tolist()]))

    print("Starting LightGBM. Train shape: {}, test shape: {}".format(len(feats), len(feats)))

    # k-fold
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['outliers'])):
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['target'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['target'].iloc[valid_idx]

        # set data structure
        lgb_train = lgb.Dataset(train_x,
                                label=train_y,
                                free_raw_data=False)
        lgb_test = lgb.Dataset(valid_x,
                               label=valid_y,
                               free_raw_data=False)

        # params optimized by optuna
        params ={
                'task': 'train',
                'boosting': 'goss',
                'objective': 'regression',
                'metric': 'rmse',
                'learning_rate': 0.01,
                'subsample': 0.9855232997390695,
                'max_depth': 6,
                'top_rate': 0.9064148448434349,
                'num_leaves': 63,
                'min_child_weight': 41.9612869171337,
                'other_rate': 0.0721768246018207,
                'reg_alpha': 9.677537745007898,
                'colsample_bytree': 0.5665320670155495,
                'min_split_gain': 9.820197773625843,
                'reg_lambda': 8.2532317400459,
                'min_data_in_leaf': 21,
                'verbose': -1,
                'seed':int(2**n_fold),
                'bagging_seed':int(2**n_fold),
                'drop_seed':int(2**n_fold)
                }

        reg = lgb.train(
                        params,
                        lgb_train,
                        valid_sets=[lgb_train, lgb_test],
                        valid_names=['train', 'test'],
                        num_boost_round=10000,
                        early_stopping_rounds= 200,
                        verbose_eval=100
                        )

        oof_preds[valid_idx] = reg.predict(valid_x, num_iteration=reg.best_iteration)
        sub_preds += reg.predict(test_df[feats], num_iteration=reg.best_iteration) / folds.n_splits

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = np.log1p(reg.feature_importance(importance_type='gain', iteration=reg.best_iteration))
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        print('Fold %2d RMSE : %.6f' % (n_fold + 1, rmse(valid_y, oof_preds[valid_idx])))
        del reg, train_x, train_y, valid_x, valid_y
        gc.collect()



    # display importances
    display_importances(feature_importance_df)
    # return feature_importance_df
    if not debug:
        # save submission file
        test_df.loc[:,'target'] = sub_preds
        test_df = test_df.reset_index()
        # test_df[['card_id', 'target']].to_csv('SUBMISSION.csv', index=False)
        test_df = test_df[['card_id', 'target']]
        return test_df

# Classify outlier
def outlier_identify(train_df, test_df,num_folds):
    outlier_vars = ['hist_month_nunique', 'hist_month_diff_mean', 'agg11_hist_mean_1',
               'delta_purchase_days_2', 'hist_duration_min',
               'hist_month_diff_max', 'new_purchase_date_uptonow',
               'hist_purchase_date_max', 'new_purchase_date_max',
               'hist_agg_card_Recency_purchase_date_int_max', 'agg11_hist_max_1',
               'agg12_hist_min_1', 'category_1_mean', 'hist_authorized_flag_mean',
               'new_month_lag_mean', 'hist_purchase_date_diff',
               'delta_purchase_days_8', 'hist_purchase_date_uptonow',
               'hist_weekofyear_nunique', 'agg17_new_min_0', 'agg11_hist_mean_0',
               'agg14_hist_sum_0', 'agg12_hist_sum_0', 'agg15_hist_sum_1',
               'agg17_new_var_0', 'agg10_hist_sum_1', 'agg12_hist_var_1',
               'agg17_hist_sum_1', 'new_month_lag_max', 'agg13_new_var_1',
               'agg12_new_min_1', 'agg11_new_mean_0', 'agg5_new_var',
               'agg13_hist_sum_0', 'agg10_new_sum_1', 'hist_price_var',
               'new_day_mean', 'agg5_new_mean', 'agg11_hist_var_0',
               'agg10_new_min_1', 'new_purchase_date_diff', 'hist_month_lag_min',
               'agg11_hist_sum_1', 'hist_installments_sum', 'agg14_hist_var_0',
               'agg15_hist_mean_0', 'hist_agg_card_Recency_purchase_amount_sum',
               'agg6_hist_var', 'hist_fathers_day_2017_mean', 'agg4_hist_var',
               'agg18_hist_mean', 'new_purchase_date_average',
               'hist_agg_card_Recency_authorized_flag_mean', 'hist_weekday_mean',
               'delta_purchase_days_7',
               'hist_agg_card_Recency_category_1_nunique',
               'hist_agg_card_Recency_purchase_date_int_min',
               'delta_purchase_days_11', 'agg10_hist_min_1', 'agg17_hist_var_0',
               'agg17_hist_var_1', 'hist_month_diff_min', 'new_hour_mean',
               'agg15_hist_nunique_1', 'hist_agg_card_Recency_installments_sum',
               'agg11_hist_min_1', 'hist_last_buy',
               'hist_agg_card_Recency_purchase_date_int_sum', 'days_feature2',
               'new_day_max', 'agg12_hist_max_0', 'hist_month_lag_mean',
               'agg14_hist_max_0', 'agg5_hist_var', 'agg14_hist_mean_0',
               'month_lag_mean', 'new_agg_card_Recency_purchase_amount_min',
               'month_diff_ratio', 'hist_agg_card_Recency_purchase_amount_mean',
               'hist_day_mean', 'agg16_hist_mean', 'agg12_hist_var_0',
               'hist_month_lag_skew', 'agg10_hist_mean_1',
               'hist_Christmas_Day_2017_mean', 'agg13_hist_var_0',
               'hist_purchase_date_average', 'agg15_hist_max_1', 'agg4_hist_mean',
               'agg19_hist_var', 'agg12_hist_mean_1', 'agg17_new_sum_1',
               'hist_agg_card_Recency_category_1_mean', 'agg18_new_mean',
               'hist_hour_mean', 'agg11_new_var_0']

    # Cross validation model
    folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=15)

    # Create arrays and dataframes to store results
    train_df['outlier'] = [1 if x<-30 else 0 for x in train_df.target.values]
    outlier = train_df.outlier

    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])

    feats = [c for c in outlier_vars]

    threshold = []
    # k-fold
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['outlier'])):
            train_x, train_y = train_df[feats].iloc[train_idx], outlier.iloc[train_idx]
            valid_x, valid_y = train_df[feats].iloc[valid_idx], outlier.iloc[valid_idx]

            # set data structure
            lgb_train = lgb.Dataset(train_x,
                                    label=train_y,
                                    free_raw_data=False)
            lgb_test = lgb.Dataset(valid_x,
                                   label=valid_y,
                                   free_raw_data=False)

            # params optimized by optuna

            num_round = 3000

            param = {'num_leaves': 80,
                     'min_data_in_leaf': 149,
                     'objective':'binary',
                     'max_depth': 6,
                     'learning_rate': 0.005,
                     "boosting": "gbdt",
                     "feature_fraction": 0.75,
                     "bagging_freq": 1,
                     "bagging_fraction": 0.68 ,
                     "bagging_seed": 11,
                     "metric": 'auc',
                     "lambda_l1": 0.2634,
                     "random_state": 133,
                     "verbosity": -1}

            classifier = lgb.train(param,
                                     lgb_train,
                                     num_round,
                                     valid_sets = [lgb_train, lgb_test],
                                     verbose_eval=100,
                                     early_stopping_rounds = 200)

            oof_preds[valid_idx] = classifier.predict(valid_x, num_iteration=classifier.best_iteration)
            sub_preds += classifier.predict(test_df[feats], num_iteration=classifier.best_iteration) / folds.n_splits
            threshold.append(pd.Series(oof_preds[valid_idx]).quantile(0.989))

    test_df.loc[:,'outlier'] = sub_preds
    # test_df['outlier'] = [1 if x >np.mean(threshold) else 0 for x in test_df['outlier']]
    test_df = test_df.reset_index()
    return test_df[['card_id', 'outlier']]


# Light On Denoised
def kfold_lightgbm_DNS(train_df, test_df, num_folds, stratified = False, debug= False):

    # Create arrays and dataframes to store results
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    feature_importance_df = pd.DataFrame()

    # k-fold
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df, train_df['outliers'])):
        train_x, train_y = train_df.iloc[train_idx].drop(['card_id','outliers','target'],axis=1), train_df['target'].iloc[train_idx]
        valid_x, valid_y = train_df.iloc[valid_idx].drop(['card_id','outliers','target'],axis=1), train_df['target'].iloc[valid_idx]

        # set data structure
        lgb_train = lgb.Dataset(train_x,
                                label=train_y,
                                free_raw_data=False)
        lgb_test = lgb.Dataset(valid_x,
                               label=valid_y,
                               free_raw_data=False)

        # params optimized by optuna
        params ={
                'task': 'train',
                'boosting': 'goss',
                'objective': 'regression',
                'metric': 'rmse',
                'learning_rate': 0.01,
                'subsample': 0.9855232997390695,
                'max_depth': 6,
                'top_rate': 0.9064148448434349,
                'num_leaves': 63,
                'min_child_weight': 41.9612869171337,
                'other_rate': 0.0721768246018207,
                'reg_alpha': 9.677537745007898,
                'colsample_bytree': 0.5665320670155495,
                'min_split_gain': 9.820197773625843,
                'reg_lambda': 8.2532317400459,
                'min_data_in_leaf': 21,
                'verbose': -1,
                'seed':int(2**n_fold),
                'bagging_seed':int(2**n_fold),
                'drop_seed':int(2**n_fold)
                }

        reg = lgb.train(
                        params,
                        lgb_train,
                        valid_sets=[lgb_train, lgb_test],
                        valid_names=['train', 'test'],
                        num_boost_round=10000,
                        early_stopping_rounds= 200,
                        verbose_eval=100
                        )

        oof_preds[valid_idx] = reg.predict(valid_x, num_iteration=reg.best_iteration)
        sub_preds += reg.predict(test_df, num_iteration=reg.best_iteration) / folds.n_splits

    # return feature_importance_df
    if not debug:
        # save submission file
        test_df.loc[:,'target'] = sub_preds
        test_df = test_df.reset_index()
        # test_df[['card_id', 'target']].to_csv('SUBMISSION.csv', index=False)
        test_df = test_df[['card_id', 'target']]
        return test_df

############################################ Execution
num_rows = None
with timer("train & test"):
    df = train_test(num_rows)
with timer("historical transactions"):
    df = pd.merge(df.reset_index(), historical_transactions(num_rows).reset_index(), on='card_id', how='outer')
with timer("new merchants"):
    df = pd.merge(df.reset_index(), new_merchant_transactions(num_rows).reset_index(), on='card_id', how='outer')
with timer("additional features"):
    df = additional_features(df)
with timer("additional features"):

    historical_transactions = pd.read_csv('historical_transactions.csv', parse_dates=['purchase_date'])
    new_transactions = pd.read_csv('new_merchant_transactions.csv',parse_dates=['purchase_date'])
    historical_transactions['purchase_date_int'] = pd.DatetimeIndex(historical_transactions['purchase_date']).astype(np.int64) * 1e-9
    new_transactions['purchase_date_int'] = pd.DatetimeIndex(new_transactions['purchase_date']).astype(np.int64) * 1e-9

    delta_days_dat = delta_days()
    recency_dat = recency()
    aggs_dat = aggs()
    aggs_plus_dat = aggs_plus()

    agg_plus_2_dat = agg_plus_2()
    df2 = pd.merge(df, delta_days_dat.reset_index()[["card_id",'delta_purchase_days_2',"delta_purchase_days_7","delta_purchase_days_8",
            "delta_purchase_days_11","delta_purchase_days_12"]], on='card_id', how='outer')\
            .merge(recency_dat.reset_index(),how='outer',on='card_id')\
            .merge(aggs_dat.reset_index(),how='outer',on='card_id')\
            .merge(aggs_plus_dat.reset_index(),how='outer',on='card_id')\
            .merge(agg_plus_2_dat.reset_index(),how='outer',on='card_id')\
            # .merge(others.reset_index(),how='outer',on='card_id')\
            # .merge(other_agg_new.reset_index(),how='outer',on='card_id')\

    # df.to_pickle('df.pkl')
    # delta_days_dat.to_pickle('delta_days_dat.pkl')
    # recency_dat.to_pickle('recency_dat.pkl')
    # aggs_dat.to_pickle('aggs_dat.pkl')
    # aggs_plus_dat.to_pickle('aggs_plus_dat.pkl')
    # agg_plus_2_dat.to_pickle('agg_plus_2_dat.pkl')

    new_vars = agg_plus_2_dat.columns
    # .append(others.columns)
    # .append(other_agg_new.columns)
with timer("split train & test"):
    train_df = df2[df2['target'].notnull()]
    test_df = df2[df2['target'].isnull()]
    # df = pd.concat([train_df,test_df],axis=0)
    gc.collect()
# with timer("Run LightGBM for Outlier with kfold"):
#     test_outlier = outlier_identify(train_df, test_df, num_folds=5)
with timer("Run LightGBM with kfold"):
    test_df2 = kfold_lightgbm(train_df, test_df, num_folds=5, stratified=False, debug=False)
    test_df3 = kfold_lightgbm(train_df.loc[train_df.target>-30], test_df, num_folds=5, stratified=False, debug=False)


W    test_df2_c = kfold_lightgbm(train_df, test_df, num_folds=5, stratified=False, debug=False)

    test_df3_c = kfold_lightgbm(train_df.loc[train_df.target>-30], test_df, num_folds=5, stratified=False, debug=False)

###
    train_denoised = pd.read_pickle('train_denoised.pkl')
    test_denoised = pd.read_pickle('test_denoised.pkl')

    train_denoised = pd.concat([train_df[['card_id','outliers','target']], train_denoised],axis=1)
    test_denoised = pd.concat([test_df[['card_id']], test_denoised],axis=1)


    test_df2_d = kfold_lightgbm_DNS(train_denoised, test_denoised, num_folds=5, stratified=False, debug=False)


test_submission = test_df2.merge(test_df3,how='outer',on='card_id')
test_submission.columns = ['card_id','target_W_outlier','target_Wn_outlier']
outlier_id = test_outlier.sort_values(by='outlier',ascending=False).head(20000).card_id.values
test_submission['target'] = [a if id in outlier_id else b for id,a,b in test_submission.values]

test_submission[['card_id', 'target']].to_csv('SUBMISSION.csv', index=False)


# imp = feature_importance_df.groupby('feature')['importance'].mean().reset_index()
# imp = imp.sort_values(by='importance',ascending=False)
# imp
#
# imp.importance.describe()
#
# sum(imp.importance<10.8)
#
#  for c in imp.loc[imp.importance>=10.8].feature:
#       print('%s' %c)
#
# keeping = imp.loc[imp.importance>=10.8].feature

# train_df.to_pickle('training_output.pkl')
# test_df.to_pickle('testing_output.pkl')



Starting LightGBM. Train shape: 206, test shape: 206
Training until validation scores don't improve for 200 rounds.
[100]	train's rmse: 3.65624	test's rmse: 3.72722
[200]	train's rmse: 3.58648	test's rmse: 3.68563
[300]	train's rmse: 3.5497	test's rmse: 3.67092
[400]	train's rmse: 3.52235	test's rmse: 3.66465
[500]	train's rmse: 3.50224	test's rmse: 3.66136
[600]	train's rmse: 3.48371	test's rmse: 3.65965
[700]	train's rmse: 3.46702	test's rmse: 3.65817
[800]	train's rmse: 3.45246	test's rmse: 3.65761
[900]	train's rmse: 3.43867	test's rmse: 3.65773
[1000]	train's rmse: 3.42493	test's rmse: 3.65751
[1100]	train's rmse: 3.41284	test's rmse: 3.65765
[1200]	train's rmse: 3.39968	test's rmse: 3.6571
[1300]	train's rmse: 3.38739	test's rmse: 3.65698
[1400]	train's rmse: 3.3749	test's rmse: 3.65684
[1500]	train's rmse: 3.36134	test's rmse: 3.65689
[1600]	train's rmse: 3.34879	test's rmse: 3.65708
Early stopping, best iteration is:
[1472]	train's rmse: 3.36519	test's rmse: 3.65666
Fold  1 RMSE : 3.656661
Training until validation scores don't improve for 200 rounds.
[100]	train's rmse: 3.6697	test's rmse: 3.66747
[200]	train's rmse: 3.59636	test's rmse: 3.6323
[300]	train's rmse: 3.55557	test's rmse: 3.62219
[400]	train's rmse: 3.52964	test's rmse: 3.61778
[500]	train's rmse: 3.50894	test's rmse: 3.61566
[600]	train's rmse: 3.48969	test's rmse: 3.61391
[700]	train's rmse: 3.47323	test's rmse: 3.6128
[800]	train's rmse: 3.45759	test's rmse: 3.61252
[900]	train's rmse: 3.44335	test's rmse: 3.61208
[1000]	train's rmse: 3.42879	test's rmse: 3.61154
[1100]	train's rmse: 3.41456	test's rmse: 3.61137
[1200]	train's rmse: 3.40057	test's rmse: 3.61071
[1300]	train's rmse: 3.38787	test's rmse: 3.6106
[1400]	train's rmse: 3.37451	test's rmse: 3.61035
[1500]	train's rmse: 3.3621	test's rmse: 3.61087
Early stopping, best iteration is:
[1399]	train's rmse: 3.3746	test's rmse: 3.61033
Fold  2 RMSE : 3.610334
Training until validation scores don't improve for 200 rounds.
[100]	train's rmse: 3.67584	test's rmse: 3.63663
[200]	train's rmse: 3.60181	test's rmse: 3.60256
[300]	train's rmse: 3.56068	test's rmse: 3.59055
[400]	train's rmse: 3.5345	test's rmse: 3.58642
[500]	train's rmse: 3.51461	test's rmse: 3.58392
[600]	train's rmse: 3.49766	test's rmse: 3.58267
[700]	train's rmse: 3.48171	test's rmse: 3.58221
[800]	train's rmse: 3.46635	test's rmse: 3.58182
[900]	train's rmse: 3.45207	test's rmse: 3.58153
[1000]	train's rmse: 3.43614	test's rmse: 3.58174
[1100]	train's rmse: 3.42128	test's rmse: 3.5817
Early stopping, best iteration is:
[957]	train's rmse: 3.44317	test's rmse: 3.58145
Fold  3 RMSE : 3.581455
Training until validation scores don't improve for 200 rounds.
[100]	train's rmse: 3.62785	test's rmse: 3.83508
[200]	train's rmse: 3.55419	test's rmse: 3.79897
[300]	train's rmse: 3.5146	test's rmse: 3.78747
[400]	train's rmse: 3.48834	test's rmse: 3.78288
[500]	train's rmse: 3.46802	test's rmse: 3.78093
[600]	train's rmse: 3.45134	test's rmse: 3.77992
[700]	train's rmse: 3.43384	test's rmse: 3.77938
[800]	train's rmse: 3.42053	test's rmse: 3.77892
[900]	train's rmse: 3.40548	test's rmse: 3.77837
[1000]	train's rmse: 3.39064	test's rmse: 3.77829
[1100]	train's rmse: 3.37553	test's rmse: 3.7783
[1200]	train's rmse: 3.36123	test's rmse: 3.77827
Early stopping, best iteration is:
[1033]	train's rmse: 3.38569	test's rmse: 3.77805
Fold  4 RMSE : 3.778052
Training until validation scores don't improve for 200 rounds.
[100]	train's rmse: 3.67154	test's rmse: 3.65174
[200]	train's rmse: 3.59911	test's rmse: 3.61855
[300]	train's rmse: 3.56081	test's rmse: 3.60987
[400]	train's rmse: 3.54241	test's rmse: 3.61551
[500]	train's rmse: 3.51321	test's rmse: 3.6075
[600]	train's rmse: 3.49111	test's rmse: 3.60577
[700]	train's rmse: 3.47433	test's rmse: 3.60534
[800]	train's rmse: 3.46119	test's rmse: 3.60472
[900]	train's rmse: 3.44621	test's rmse: 3.60477
[1000]	train's rmse: 3.43067	test's rmse: 3.60501
Early stopping, best iteration is:
[858]	train's rmse: 3.45301	test's rmse: 3.60429
Fold  5 RMSE : 3.604295








Starting LightGBM. Train shape: 215, test shape: 215
Training until validation scores don't improve for 200 rounds.
[100]	train's rmse: 3.63218	test's rmse: 3.72469
[200]	train's rmse: 3.54446	test's rmse: 3.6811
[300]	train's rmse: 3.49586	test's rmse: 3.66576
[400]	train's rmse: 3.46076	test's rmse: 3.65928
[500]	train's rmse: 3.43323	test's rmse: 3.6558
[600]	train's rmse: 3.4093	test's rmse: 3.65317
[700]	train's rmse: 3.38675	test's rmse: 3.65241
[800]	train's rmse: 3.36486	test's rmse: 3.65162
[900]	train's rmse: 3.34569	test's rmse: 3.65166
[1000]	train's rmse: 3.32664	test's rmse: 3.65168
[1100]	train's rmse: 3.30792	test's rmse: 3.65199
[1200]	train's rmse: 3.29027	test's rmse: 3.6523
Early stopping, best iteration is:
[1012]	train's rmse: 3.32464	test's rmse: 3.65152
Fold  1 RMSE : 3.651523
Training until validation scores don't improve for 200 rounds.
[100]	train's rmse: 3.64514	test's rmse: 3.66361
[200]	train's rmse: 3.55434	test's rmse: 3.62809
[300]	train's rmse: 3.50017	test's rmse: 3.61798
[400]	train's rmse: 3.46511	test's rmse: 3.61288
[500]	train's rmse: 3.43786	test's rmse: 3.61034
[600]	train's rmse: 3.41441	test's rmse: 3.60956
[700]	train's rmse: 3.38929	test's rmse: 3.60833
[800]	train's rmse: 3.36804	test's rmse: 3.60715
[900]	train's rmse: 3.34749	test's rmse: 3.60655
[1000]	train's rmse: 3.32736	test's rmse: 3.60622
[1100]	train's rmse: 3.3081	test's rmse: 3.60573
[1200]	train's rmse: 3.2887	test's rmse: 3.60558
[1300]	train's rmse: 3.27013	test's rmse: 3.6055
[1400]	train's rmse: 3.25206	test's rmse: 3.60525
[1500]	train's rmse: 3.23329	test's rmse: 3.60477
[1600]	train's rmse: 3.21593	test's rmse: 3.60462
[1700]	train's rmse: 3.19752	test's rmse: 3.60498
Early stopping, best iteration is:
[1584]	train's rmse: 3.21844	test's rmse: 3.60455
Fold  2 RMSE : 3.604550
Training until validation scores don't improve for 200 rounds.
[100]	train's rmse: 3.65065	test's rmse: 3.63209
[200]	train's rmse: 3.56115	test's rmse: 3.59813
[300]	train's rmse: 3.5073	test's rmse: 3.58622
[400]	train's rmse: 3.47197	test's rmse: 3.58172
[500]	train's rmse: 3.4456	test's rmse: 3.57915
[600]	train's rmse: 3.4217	test's rmse: 3.57797
[700]	train's rmse: 3.39915	test's rmse: 3.57792
[800]	train's rmse: 3.37858	test's rmse: 3.57849
Early stopping, best iteration is:
[656]	train's rmse: 3.40839	test's rmse: 3.57771
Fold  3 RMSE : 3.577713
Training until validation scores don't improve for 200 rounds.
[100]	train's rmse: 3.60233	test's rmse: 3.8309
[200]	train's rmse: 3.51246	test's rmse: 3.79198
[300]	train's rmse: 3.46125	test's rmse: 3.77918
[400]	train's rmse: 3.42755	test's rmse: 3.77436
[500]	train's rmse: 3.3996	test's rmse: 3.77206
[600]	train's rmse: 3.37523	test's rmse: 3.7704
[700]	train's rmse: 3.35252	test's rmse: 3.76944
[800]	train's rmse: 3.32997	test's rmse: 3.76932
[900]	train's rmse: 3.30728	test's rmse: 3.76911
[1000]	train's rmse: 3.28452	test's rmse: 3.76903
[1100]	train's rmse: 3.2637	test's rmse: 3.76928
Early stopping, best iteration is:
[931]	train's rmse: 3.30059	test's rmse: 3.76887
Fold  4 RMSE : 3.768866
Training until validation scores don't improve for 200 rounds.
[100]	train's rmse: 3.64575	test's rmse: 3.64976
[200]	train's rmse: 3.55449	test's rmse: 3.61742
[300]	train's rmse: 3.50092	test's rmse: 3.60756
[400]	train's rmse: 3.4675	test's rmse: 3.60435
[500]	train's rmse: 3.43925	test's rmse: 3.60266
[600]	train's rmse: 3.41328	test's rmse: 3.6013
[700]	train's rmse: 3.38907	test's rmse: 3.60108
[800]	train's rmse: 3.36641	test's rmse: 3.60139
Early stopping, best iteration is:
[685]	train's rmse: 3.39315	test's rmse: 3.60093
Fold  5 RMSE : 3.600934







Starting LightGBM. Train shape: 244, test shape: 244
Training until validation scores don't improve for 200 rounds.
[100]	train's rmse: 3.65452	test's rmse: 3.72801
[200]	train's rmse: 3.58354	test's rmse: 3.68539
[300]	train's rmse: 3.54517	test's rmse: 3.67053
[400]	train's rmse: 3.51869	test's rmse: 3.66364
[500]	train's rmse: 3.49679	test's rmse: 3.66028
[600]	train's rmse: 3.4778	test's rmse: 3.6583
[700]	train's rmse: 3.4599	test's rmse: 3.65721
[800]	train's rmse: 3.44418	test's rmse: 3.65696
[900]	train's rmse: 3.42951	test's rmse: 3.65648
[1000]	train's rmse: 3.41576	test's rmse: 3.65601
[1100]	train's rmse: 3.4029	test's rmse: 3.65595
[1200]	train's rmse: 3.39015	test's rmse: 3.65596
[1300]	train's rmse: 3.37878	test's rmse: 3.65581
[1400]	train's rmse: 3.36579	test's rmse: 3.65588
[1500]	train's rmse: 3.3532	test's rmse: 3.65634
Early stopping, best iteration is:
[1340]	train's rmse: 3.37341	test's rmse: 3.65572
Fold  1 RMSE : 3.655719
Training until validation scores don't improve for 200 rounds.
[100]	train's rmse: 3.66978	test's rmse: 3.66828
[200]	train's rmse: 3.59487	test's rmse: 3.63332
[300]	train's rmse: 3.55191	test's rmse: 3.62211
[400]	train's rmse: 3.52597	test's rmse: 3.61707
[500]	train's rmse: 3.5054	test's rmse: 3.61452
[600]	train's rmse: 3.48692	test's rmse: 3.61323
[700]	train's rmse: 3.47136	test's rmse: 3.61221
[800]	train's rmse: 3.45686	test's rmse: 3.61156
[900]	train's rmse: 3.44274	test's rmse: 3.61103
[1000]	train's rmse: 3.42932	test's rmse: 3.6106
[1100]	train's rmse: 3.41497	test's rmse: 3.61008
[1200]	train's rmse: 3.40098	test's rmse: 3.61025
Early stopping, best iteration is:
[1070]	train's rmse: 3.41918	test's rmse: 3.61
Fold  2 RMSE : 3.609997
Training until validation scores don't improve for 200 rounds.
[100]	train's rmse: 3.67504	test's rmse: 3.6375
[200]	train's rmse: 3.5997	test's rmse: 3.60444
[300]	train's rmse: 3.55645	test's rmse: 3.59345
[400]	train's rmse: 3.53012	test's rmse: 3.58832
[500]	train's rmse: 3.50943	test's rmse: 3.58611
[600]	train's rmse: 3.49195	test's rmse: 3.58471
[700]	train's rmse: 3.47461	test's rmse: 3.58413
[800]	train's rmse: 3.45901	test's rmse: 3.58412
[900]	train's rmse: 3.44451	test's rmse: 3.58381
[1000]	train's rmse: 3.4292	test's rmse: 3.58401
Early stopping, best iteration is:
[845]	train's rmse: 3.45219	test's rmse: 3.58376
Fold  3 RMSE : 3.583756
Training until validation scores don't improve for 200 rounds.
[100]	train's rmse: 3.62611	test's rmse: 3.83357
[200]	train's rmse: 3.5507	test's rmse: 3.79773
[300]	train's rmse: 3.51111	test's rmse: 3.78624
[400]	train's rmse: 3.48389	test's rmse: 3.78191
[500]	train's rmse: 3.46311	test's rmse: 3.78001
[600]	train's rmse: 3.44449	test's rmse: 3.7791
[700]	train's rmse: 3.42894	test's rmse: 3.77841
[800]	train's rmse: 3.41275	test's rmse: 3.77811
[900]	train's rmse: 3.39699	test's rmse: 3.77741
[1000]	train's rmse: 3.38156	test's rmse: 3.77683
[1100]	train's rmse: 3.36698	test's rmse: 3.77642
[1200]	train's rmse: 3.35235	test's rmse: 3.77638
[1300]	train's rmse: 3.33874	test's rmse: 3.77693
Early stopping, best iteration is:
[1182]	train's rmse: 3.35482	test's rmse: 3.77632
Fold  4 RMSE : 3.776322
Training until validation scores don't improve for 200 rounds.
[100]	train's rmse: 3.67133	test's rmse: 3.65314
[200]	train's rmse: 3.59838	test's rmse: 3.62087
[300]	train's rmse: 3.55916	test's rmse: 3.61025
[400]	train's rmse: 3.53394	test's rmse: 3.60653
[500]	train's rmse: 3.51172	test's rmse: 3.60451
[600]	train's rmse: 3.49255	test's rmse: 3.60406
[700]	train's rmse: 3.47533	test's rmse: 3.6037
[800]	train's rmse: 3.45881	test's rmse: 3.60393
[900]	train's rmse: 3.44247	test's rmse: 3.60428
Early stopping, best iteration is:
[710]	train's rmse: 3.47353	test's rmse: 3.60361
Fold  5 RMSE : 3.603614

##################################################################################################
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train.target.describe()

#####################################
ep = train[['card_id','target']].merge(historical_transactions,how='left',on='card_id')
epp = ep.groupby(['category_1','city_id','subsector_id'])['target'].mean().reset_index()
others = historical_transactions.merge(epp,how='left',on=['category_1','city_id','subsector_id'])
others = others.groupby('card_id')['target'].agg({'others_mean':'mean','others_var':'var','others_min':'min','others_max':'max'})

########################################
### Other Agg of response by keys
temp = train[['card_id','target']].merge(historical_transactions,how='left',on='card_id')
other_agg_city_mean = temp.groupby('city_id')['target'].mean().to_frame('other_agg_city_mean')
other_agg_state_mean = temp.groupby('state_id')['target'].mean().to_frame('other_agg_state_mean')
other_agg_subsector_mean = temp.groupby('subsector_id')['target'].mean().to_frame('other_agg_subsector_mean')
other_agg_merchant_cat_mean = temp.groupby('merchant_category_id')['target'].mean().to_frame('other_agg_merchant_cat_mean')
other_agg_cat3_mean = temp.groupby('category_3')['target'].mean().to_frame('other_agg_cat3_mean')
other_agg_cat2_mean = temp.groupby('category_2')['target'].mean().to_frame('other_agg_cat2_mean')
other_agg_instal_mean = temp.groupby('installments')['target'].mean().to_frame('other_agg_instal_mean')
other_agg_month_lag_mean = temp.groupby('month_lag')['target'].mean().to_frame('other_agg_month_lag_mean')
other_agg_cat1_mean = temp.groupby('category_1')['target'].mean().to_frame('other_agg_cat1_mean')
other_agg_auth_mean = temp.groupby('authorized_flag')['target'].mean().to_frame('other_agg_auth_mean')
# other_agg_merchant_mean = temp.groupby('merchant_id')['target'].median().to_frame('other_agg_merchant_mean')


temp = historical_transactions.merge(other_agg_city_mean,how='left',left_on='city_id',right_index=True)
temp = temp.merge(other_agg_state_mean,how='left',left_on='state_id',right_index=True)
temp = temp.merge(other_agg_subsector_mean,how='left',left_on='subsector_id',right_index=True)
temp = temp.merge(other_agg_merchant_cat_mean,how='left',left_on='merchant_category_id',right_index=True)
temp = temp.merge(other_agg_cat3_mean,how='left',left_on='category_3',right_index=True)
temp = temp.merge(other_agg_cat2_mean,how='left',left_on='category_2',right_index=True)
temp = temp.merge(other_agg_instal_mean,how='left',left_on='installments',right_index=True)
temp = temp.merge(other_agg_month_lag_mean,how='left',left_on='month_lag',right_index=True)
temp = temp.merge(other_agg_cat1_mean,how='left',left_on='category_1',right_index=True)
temp = temp.merge(other_agg_auth_mean,how='left',left_on='authorized_flag',right_index=True)
# temp = temp.merge(other_agg_merchant_mean,how='left',left_on='merchant_id',right_index=True)

other_agg = temp.groupby('card_id').agg({'other_agg_city_mean':'mean','other_agg_state_mean':'mean',\
                                         'other_agg_state_mean':'mean','other_agg_subsector_mean':'mean',\
                                         'other_agg_merchant_cat_mean':'mean','other_agg_cat3_mean':'mean',\
                                         'other_agg_cat2_mean':'mean','other_agg_instal_mean':'mean',\
                                         'other_agg_cat1_mean':'mean','other_agg_auth_mean':'mean'})

removal_as_overfitting_var = [
                        "other_agg_subsector",
                        "new_agg_card_purchase_weekofyear_mean",
                        "hist_agg_card_Recency_purchase_weekofyear_mean",
                        "hist_agg_card_Recency_purchase_amount_max",
                        "new_agg_card_purchase_amount_std",
                        "new_agg_card_Recency_purchase_date_gap1_mean",
                        "other_agg_city_new",
                        "other_agg_state",
                        "hist_agg_card_Recency_purchase_date_int_mean",
                        "other_agg_month_new",
                        "other_agg_merchant_cat_new",
                        "other_agg_weekofyear_new",
                        "hist_agg_card_Recency_purchase_date_gap1_mean",
                        "hist_agg_card_Recency_1_purchase_AMT",
                        "new_agg_card_Recency_purchase_weekofyear_mean",
                        "card_UNinstallments_date_max",
                        "other_agg_weekday_new"]


other_agg_vars = [v for v in other_agg.columns if v not in removal_as_overfitting_var]
other_agg = other_agg[other_agg_vars]




other_agg_min = temp.groupby('card_id').agg({'other_agg_city_mean':'min','other_agg_state_mean':'min',\
                                         'other_agg_state_mean':'min','other_agg_subsector_mean':'min',\
                                         'other_agg_merchant_cat_mean':'min','other_agg_cat3_mean':'min',\
                                         'other_agg_cat2_mean':'min','other_agg_instal_mean':'min',\
                                         'other_agg_month_mean':'min', 'other_agg_weekday_mean':'min',\
                                         'other_agg_weekofyear_mean':'min', 'other_agg_month_lag_mean':'min',\
                                         'other_agg_cat1_mean':'min','other_agg_cat4_mean':'min','other_agg_auth_mean':'min'})


other_agg_max = temp.groupby('card_id').agg({'other_agg_city_mean':'max','other_agg_state_mean':'max',\
                                          'other_agg_state_mean':'max','other_agg_subsector_mean':'max',\
                                          'other_agg_merchant_cat_mean':'max','other_agg_cat3_mean':'max',\
                                          'other_agg_cat2_mean':'max','other_agg_instal_mean':'max',\
                                          'other_agg_month_mean':'max', 'other_agg_weekday_mean':'max',\
                                          'other_agg_weekofyear_mean':'max', 'other_agg_month_lag_mean':'max',\
                                          'other_agg_cat1_mean':'max','other_agg_cat4_mean':'max','other_agg_auth_mean':'max'})
gc.collect()
########################################################################################################################
temp = train[['card_id','target']].merge(new_transactions,how='left',on='card_id')
other_agg_city_new = temp.groupby('city_id')['target'].var().to_frame('other_agg_city_new_mean')
other_agg_state_new = temp.groupby('state_id')['target'].var().to_frame('other_agg_state_new_mean')
other_agg_subsector_new = temp.groupby('subsector_id')['target'].var().to_frame('other_agg_subsector_new_mean')
other_agg_merchant_cat_new = temp.groupby('merchant_category_id')['target'].var().to_frame('other_agg_merchant_cat_new_mean')
other_agg_cat3_new = temp.groupby('category_3')['target'].var().to_frame('other_agg_cat3_new_mean')
other_agg_cat2_new = temp.groupby('category_2')['target'].var().to_frame('other_agg_cat2_new_mean')
other_agg_instal_new = temp.groupby('installments')['target'].var().to_frame('other_agg_instal_new_mean')
other_agg_month_new = temp.groupby('purchase_month')['target'].var().to_frame('other_agg_month_new_mean')
other_agg_weekday_new = temp.groupby('purchase_weekofyear')['target'].var().to_frame('other_agg_weekday_new_mean')
other_agg_weekofyear_new = temp.groupby('purchase_weekday')['target'].var().to_frame('other_agg_weekofyear_new_mean')
other_agg_cat1_new = temp.groupby('category_1')['target'].var().to_frame('other_agg_cat1_new_mean')
other_agg_cat4_new = temp.groupby('category_4')['target'].var().to_frame('other_agg_cat4_new_mean')
other_agg_auth_new = temp.groupby('authorized_flag')['target'].var().to_frame('other_agg_auth_new_mean')

temp = new_transactions.merge(other_agg_city_new,how='left',left_on='city_id',right_index=True)
temp = temp.merge(other_agg_state_new,how='left',left_on='state_id',right_index=True)
temp = temp.merge(other_agg_subsector_new,how='left',left_on='subsector_id',right_index=True)
temp = temp.merge(other_agg_merchant_cat_new,how='left',left_on='merchant_category_id',right_index=True)
temp = temp.merge(other_agg_cat3_new,how='left',left_on='category_3',right_index=True)
temp = temp.merge(other_agg_cat2_new,how='left',left_on='category_2',right_index=True)
temp = temp.merge(other_agg_instal_new,how='left',left_on='installments',right_index=True)
temp = temp.merge(other_agg_month_new,how='left',left_on='purchase_month',right_index=True)
temp = temp.merge(other_agg_weekday_new,how='left',left_on='purchase_weekofyear',right_index=True)
temp = temp.merge(other_agg_weekofyear_new,how='left',left_on='purchase_weekday',right_index=True)
temp = temp.merge(other_agg_cat1_new,how='left',left_on='category_1',right_index=True)
temp = temp.merge(other_agg_cat4_new,how='left',left_on='category_4',right_index=True)
temp = temp.merge(other_agg_auth_new,how='left',left_on='authorized_flag',right_index=True)


other_agg_new = temp.groupby('card_id').agg({'other_agg_city_new_mean':'mean','other_agg_state_new_mean':'mean',\
                                             'other_agg_state_new_mean':'mean','other_agg_subsector_new_mean':'mean',\
                                             'other_agg_merchant_cat_new_mean':'mean','other_agg_cat3_new_mean':'mean',\
                                             'other_agg_cat2_new_mean':'mean','other_agg_instal_new_mean':'mean',\
                                             # 'other_agg_month_new_mean':'mean', 'other_agg_weekday_new_mean':'mean',\
                                             # 'other_agg_weekofyear_new_mean':'mean',
                                             'other_agg_cat1_new_mean':'mean',\
                                             # 'other_agg_cat4_new_mean':'mean',
                                             'other_agg_auth_new_mean':'mean'})

other_agg_new_min = temp.groupby('card_id').agg({'other_agg_city_new_mean':'min','other_agg_state_new_mean':'min',\
                                         'other_agg_state_new_mean':'min','other_agg_subsector_new_mean':'min',\
                                         'other_agg_merchant_cat_new_mean':'min','other_agg_cat3_new_mean':'min',\
                                         'other_agg_cat2_new_mean':'min','other_agg_instal_new_mean':'min',
                                         'other_agg_month_new_mean':'min', 'other_agg_weekday_new_mean':'min',
                                          'other_agg_weekofyear_new_mean':'min','other_agg_cat1_new_mean':'min',\
                                          'other_agg_cat4_new_mean':'min','other_agg_auth_new_mean':'min'})

other_agg_new_max = temp.groupby('card_id').agg({'other_agg_city_new_mean':'max','other_agg_state_new_mean':'max',\
                                         'other_agg_state_new_mean':'max','other_agg_subsector_new_mean':'max',\
                                         'other_agg_merchant_cat_new_mean':'max','other_agg_cat3_new_mean':'max',\
                                         'other_agg_cat2_new_mean':'max','other_agg_instal_new_mean':'max',
                                         'other_agg_month_new_mean':'max', 'other_agg_weekday_new_mean':'max',
                                         'other_agg_weekofyear_new_mean':'max','other_agg_cat1_new_mean':'max',\
                                         'other_agg_cat4_new_mean':'max','other_agg_auth_new_mean':'max'})


####################################################################################################
temp = train[['card_id','target']].merge(historical_transactions,how='left',on='card_id')
other_agg_city_var = temp.groupby('city_id')['target'].var().to_frame('other_agg_city_var')
other_agg_state_var = temp.groupby('state_id')['target'].var().to_frame('other_agg_state_var')
other_agg_subsector_var = temp.groupby('subsector_id')['target'].var().to_frame('other_agg_subsector_var')
other_agg_merchant_cat_var = temp.groupby('merchant_category_id')['target'].var().to_frame('other_agg_merchant_cat_var')
other_agg_cat3_var = temp.groupby('category_3')['target'].var().to_frame('other_agg_cat3_var')
other_agg_cat2_var = temp.groupby('category_2')['target'].var().to_frame('other_agg_cat2_var')
other_agg_instal_var = temp.groupby('installments')['target'].var().to_frame('other_agg_instal_var')
other_agg_month_var = temp.groupby('purchase_month')['target'].var().to_frame('other_agg_month_var')
other_agg_weekday_var = temp.groupby('purchase_weekofyear')['target'].var().to_frame('other_agg_weekday_var')
other_agg_weekofyear_var = temp.groupby('purchase_weekday')['target'].var().to_frame('other_agg_weekofyear_var')
other_agg_month_lag_var = temp.groupby('month_lag')['target'].var().to_frame('other_agg_month_lag_var')
other_agg_cat1_var = temp.groupby('category_1')['target'].var().to_frame('other_agg_cat1_var')
other_agg_cat4_var = temp.groupby('category_4')['target'].var().to_frame('other_agg_cat4_var')
other_agg_auth_var = temp.groupby('authorized_flag')['target'].var().to_frame('other_agg_auth_var')

temp = historical_transactions.merge(other_agg_city_var,how='left',left_on='city_id',right_index=True)
temp = temp.merge(other_agg_state_var,how='left',left_on='state_id',right_index=True)
temp = temp.merge(other_agg_subsector_var,how='left',left_on='subsector_id',right_index=True)
temp = temp.merge(other_agg_merchant_cat_var,how='left',left_on='merchant_category_id',right_index=True)
temp = temp.merge(other_agg_cat3_var,how='left',left_on='category_3',right_index=True)
temp = temp.merge(other_agg_cat2_var,how='left',left_on='category_2',right_index=True)
temp = temp.merge(other_agg_instal_var,how='left',left_on='installments',right_index=True)
temp = temp.merge(other_agg_month_var,how='left',left_on='purchase_month',right_index=True)
temp = temp.merge(other_agg_weekday_var,how='left',left_on='purchase_weekofyear',right_index=True)
temp = temp.merge(other_agg_weekofyear_var,how='left',left_on='purchase_weekday',right_index=True)
temp = temp.merge(other_agg_month_lag_var,how='left',left_on='month_lag',right_index=True)
temp = temp.merge(other_agg_cat1_var,how='left',left_on='category_1',right_index=True)
temp = temp.merge(other_agg_cat4_var,how='left',left_on='category_4',right_index=True)
temp = temp.merge(other_agg_auth_var,how='left',left_on='authorized_flag',right_index=True)

other_agg_var = temp.groupby('card_id').agg({'other_agg_city_var':'mean','other_agg_state_var':'mean',\
                                         'other_agg_state_var':'mean','other_agg_subsector_var':'mean',\
                                         'other_agg_merchant_cat_var':'mean','other_agg_cat3_var':'mean',\
                                         'other_agg_cat2_var':'mean','other_agg_instal_var':'mean',\
                                         'other_agg_month_var':'mean', 'other_agg_weekday_var':'mean', \
                                         'other_agg_weekofyear_var':'mean', 'other_agg_month_lag_var':'mean',\
                                         'other_agg_cat1_var':'mean',\
                                         'other_agg_cat4_var':'mean','other_agg_auth_var':'mean'})

# other_agg_var_min = temp.groupby('card_id').agg({'other_agg_city_var':'min','other_agg_state_var':'min',\
#                                          'other_agg_state_var':'min','other_agg_subsector_var':'min',\
#                                          'other_agg_merchant_cat_var':'min','other_agg_cat3_var':'min',\
#                                          'other_agg_cat2_var':'min','other_agg_instal_var':'min',\
#                                          'other_agg_month_var':'min', 'other_agg_weekday_var':'min', \
#                                          'other_agg_weekofyear_var':'min', 'other_agg_month_lag_var':'min'})
#
# other_agg_var_max = temp.groupby('card_id').agg({'other_agg_city_var':'max','other_agg_state_var':'max',\
#                                          'other_agg_state_var':'max','other_agg_subsector_var':'max',\
#                                          'other_agg_merchant_cat_var':'max','other_agg_cat3_var':'max',\
#                                          'other_agg_cat2_var':'max','other_agg_instal_var':'max',\
#                                          'other_agg_month_var':'max', 'other_agg_weekday_var':'max', \
#                                          'other_agg_weekofyear_var':'max', 'other_agg_month_lag_var':'max'})

#############
temp = train[['card_id','target']].merge(new_transactions,how='left',on='card_id')
other_agg_city_new_var = temp.groupby('city_id')['target'].var().to_frame('other_agg_city_new_var')
other_agg_state_new_var = temp.groupby('state_id')['target'].var().to_frame('other_agg_state_new_var')
other_agg_subsector_new_var = temp.groupby('subsector_id')['target'].var().to_frame('other_agg_subsector_new_var')
other_agg_merchant_cat_new_var = temp.groupby('merchant_category_id')['target'].var().to_frame('other_agg_merchant_cat_new_var')
other_agg_cat3_new_var = temp.groupby('category_3')['target'].var().to_frame('other_agg_cat3_new_var')
other_agg_cat2_new_var = temp.groupby('category_2')['target'].var().to_frame('other_agg_cat2_new_var')
other_agg_instal_new_var = temp.groupby('installments')['target'].var().to_frame('other_agg_instal_new_var')
other_agg_month_new_var = temp.groupby('purchase_month')['target'].var().to_frame('other_agg_month_new_var')
other_agg_weekday_new_var = temp.groupby('purchase_weekofyear')['target'].var().to_frame('other_agg_weekday_new_var')
other_agg_weekofyear_new_var = temp.groupby('purchase_weekday')['target'].var().to_frame('other_agg_weekofyear_new_var')
other_agg_cat1_new_var = temp.groupby('category_1')['target'].var().to_frame('other_agg_cat1_new_var')
other_agg_cat4_new_var = temp.groupby('category_4')['target'].var().to_frame('other_agg_cat4_new_var')
other_agg_auth_new_var = temp.groupby('authorized_flag')['target'].var().to_frame('other_agg_auth_new_var')


temp = new_transactions.merge(other_agg_city_new_var,how='left',left_on='city_id',right_index=True)
temp = temp.merge(other_agg_state_new_var,how='left',left_on='state_id',right_index=True)
temp = temp.merge(other_agg_subsector_new_var,how='left',left_on='subsector_id',right_index=True)
temp = temp.merge(other_agg_merchant_cat_new_var,how='left',left_on='merchant_category_id',right_index=True)
temp = temp.merge(other_agg_cat3_new_var,how='left',left_on='category_3',right_index=True)
temp = temp.merge(other_agg_cat2_new_var,how='left',left_on='category_2',right_index=True)
temp = temp.merge(other_agg_instal_new_var,how='left',left_on='installments',right_index=True)
temp = temp.merge(other_agg_month_new_var,how='left',left_on='purchase_month',right_index=True)
temp = temp.merge(other_agg_weekday_new_var,how='left',left_on='purchase_weekofyear',right_index=True)
temp = temp.merge(other_agg_weekofyear_new_var,how='left',left_on='purchase_weekday',right_index=True)
temp = temp.merge(other_agg_cat1_new_var,how='left',left_on='category_1',right_index=True)
temp = temp.merge(other_agg_cat4_new_var,how='left',left_on='category_4',right_index=True)
temp = temp.merge(other_agg_auth_new_var,how='left',left_on='authorized_flag',right_index=True)


other_agg_new_var = temp.groupby('card_id').agg({'other_agg_city_new_var':'mean','other_agg_state_new_var':'mean',\
                                         'other_agg_state_new_var':'mean','other_agg_subsector_new_var':'mean',\
                                         'other_agg_merchant_cat_new_var':'mean','other_agg_cat3_new_var':'mean',\
                                         'other_agg_cat2_new_var':'mean','other_agg_instal_new_var':'mean',
                                         'other_agg_month_new_var':'mean', 'other_agg_weekday_new_var':'mean', \
                                         'other_agg_weekofyear_new_var':'mean','other_agg_cat1_new_var':'mean',\
                                         'other_agg_cat4_new_var':'mean','other_agg_auth_new_var':'mean'})







Starting LightGBM. Train shape: (201917, 288), test shape: (123623, 288)
Training until validation scores don't improve for 200 rounds.
[100]	train's rmse: 3.64153	test's rmse: 3.7275
[200]	train's rmse: 3.55761	test's rmse: 3.68853
[300]	train's rmse: 3.50776	test's rmse: 3.6743
[400]	train's rmse: 3.472	test's rmse: 3.66861
[500]	train's rmse: 3.44731	test's rmse: 3.66653
[600]	train's rmse: 3.42521	test's rmse: 3.66591
[700]	train's rmse: 3.4045	test's rmse: 3.66544
[800]	train's rmse: 3.38386	test's rmse: 3.6659
Early stopping, best iteration is:
[687]	train's rmse: 3.40735	test's rmse: 3.66542
Fold  1 RMSE : 3.665416
Training until validation scores don't improve for 200 rounds.
[100]	train's rmse: 3.65602	test's rmse: 3.6658
[200]	train's rmse: 3.56769	test's rmse: 3.63289
[300]	train's rmse: 3.51418	test's rmse: 3.62251
[400]	train's rmse: 3.47775	test's rmse: 3.61775
[500]	train's rmse: 3.44995	test's rmse: 3.61542
[600]	train's rmse: 3.42713	test's rmse: 3.61417
[700]	train's rmse: 3.40597	test's rmse: 3.61321
[800]	train's rmse: 3.38741	test's rmse: 3.61263
[900]	train's rmse: 3.36813	test's rmse: 3.61257
[1000]	train's rmse: 3.34949	test's rmse: 3.61239
[1100]	train's rmse: 3.32976	test's rmse: 3.61263
Early stopping, best iteration is:
[959]	train's rmse: 3.35664	test's rmse: 3.61216
Fold  2 RMSE : 3.612162
Training until validation scores don't improve for 200 rounds.
[100]	train's rmse: 3.66469	test's rmse: 3.63507
[200]	train's rmse: 3.58019	test's rmse: 3.60033
[300]	train's rmse: 3.52668	test's rmse: 3.58801
[400]	train's rmse: 3.49075	test's rmse: 3.58381
[500]	train's rmse: 3.46538	test's rmse: 3.58429
[600]	train's rmse: 3.43448	test's rmse: 3.58165
[700]	train's rmse: 32.2611	test's rmse: 32.3282
[800]	train's rmse: 15.8542	test's rmse: 16.0271
Early stopping, best iteration is:
[612]	train's rmse: 3.43135	test's rmse: 3.5815
Fold  3 RMSE : 3.581503
Training until validation scores don't improve for 200 rounds.
[100]	train's rmse: 3.61397	test's rmse: 3.83742
[200]	train's rmse: 3.52892	test's rmse: 3.80146
[300]	train's rmse: 3.47852	test's rmse: 3.79102
[400]	train's rmse: 3.44265	test's rmse: 3.78632
[500]	train's rmse: 3.41888	test's rmse: 3.78618
[600]	train's rmse: 3.39493	test's rmse: 3.7852
[700]	train's rmse: 3.37443	test's rmse: 3.78425
[800]	train's rmse: 3.35588	test's rmse: 3.78358
[900]	train's rmse: 3.33767	test's rmse: 3.78396
[1000]	train's rmse: 3.31978	test's rmse: 3.78408
Early stopping, best iteration is:
[800]	train's rmse: 3.35588	test's rmse: 3.78358
Fold  4 RMSE : 3.783581
Training until validation scores don't improve for 200 rounds.
[100]	train's rmse: 3.65815	test's rmse: 3.65124
[200]	train's rmse: 3.57195	test's rmse: 3.61869
[300]	train's rmse: 3.51989	test's rmse: 3.60813
[400]	train's rmse: 3.48438	test's rmse: 3.60549
[500]	train's rmse: 3.45675	test's rmse: 3.60417
[600]	train's rmse: 3.43186	test's rmse: 3.60421
[700]	train's rmse: 3.40956	test's rmse: 3.60411
Early stopping, best iteration is:
[548]	train's rmse: 3.44506	test's rmse: 3.60375
Fold  5 RMSE : 3.603755


#############################################################################

Starting LightGBM. Train shape: (201917, 328), test shape: (123623, 328)
Training until validation scores don't improve for 200 rounds.
[100]	train's rmse: 3.6353	test's rmse: 3.72761
[200]	train's rmse: 3.54732	test's rmse: 3.68511
[300]	train's rmse: 3.49753	test's rmse: 3.6711
[400]	train's rmse: 3.46076	test's rmse: 3.66516
[500]	train's rmse: 3.43187	test's rmse: 3.66236
[600]	train's rmse: 3.41057	test's rmse: 3.66132
[700]	train's rmse: 3.38896	test's rmse: 3.66102
[800]	train's rmse: 3.36859	test's rmse: 3.66092
[900]	train's rmse: 3.34838	test's rmse: 3.66088
[1000]	train's rmse: 3.32963	test's rmse: 3.66115
Early stopping, best iteration is:
[862]	train's rmse: 3.35617	test's rmse: 3.66067
Fold  1 RMSE : 3.660670
Training until validation scores don't improve for 200 rounds.
[100]	train's rmse: 3.6489	test's rmse: 3.66574
[200]	train's rmse: 3.55938	test's rmse: 3.63159
[300]	train's rmse: 3.50306	test's rmse: 3.62227
[400]	train's rmse: 3.46722	test's rmse: 3.6186
[500]	train's rmse: 3.43709	test's rmse: 3.61717
[600]	train's rmse: 3.41003	test's rmse: 3.61543
[700]	train's rmse: 3.38813	test's rmse: 3.61501
[800]	train's rmse: 3.36741	test's rmse: 3.61502
[900]	train's rmse: 3.34709	test's rmse: 3.61447
[1000]	train's rmse: 3.32752	test's rmse: 3.61441
[1100]	train's rmse: 3.3075	test's rmse: 3.61477
Early stopping, best iteration is:
[965]	train's rmse: 3.33429	test's rmse: 3.61417
Fold  2 RMSE : 3.614167
Training until validation scores don't improve for 200 rounds.
[100]	train's rmse: 3.65549	test's rmse: 3.63422
[200]	train's rmse: 3.56846	test's rmse: 3.60148
[300]	train's rmse: 3.51235	test's rmse: 3.58857
[400]	train's rmse: 3.47566	test's rmse: 3.58391
[500]	train's rmse: 3.44876	test's rmse: 3.58175
[600]	train's rmse: 3.42622	test's rmse: 3.58061
[700]	train's rmse: 3.40576	test's rmse: 3.5809
Early stopping, best iteration is:
[590]	train's rmse: 3.4286	test's rmse: 3.58048
Fold  3 RMSE : 3.580484
Training until validation scores don't improve for 200 rounds.
[100]	train's rmse: 3.60684	test's rmse: 3.8345
[200]	train's rmse: 3.51808	test's rmse: 3.79925
[300]	train's rmse: 3.46305	test's rmse: 3.78657
[400]	train's rmse: 3.4261	test's rmse: 3.78236
[500]	train's rmse: 3.39863	test's rmse: 3.78064
[600]	train's rmse: 3.37637	test's rmse: 3.77972
[700]	train's rmse: 3.35676	test's rmse: 3.7794
[800]	train's rmse: 3.33763	test's rmse: 3.77918
[900]	train's rmse: 3.31744	test's rmse: 3.77994
Early stopping, best iteration is:
[778]	train's rmse: 3.3422	test's rmse: 3.77913
Fold  4 RMSE : 3.779127
Training until validation scores don't improve for 200 rounds.
[100]	train's rmse: 3.65164	test's rmse: 3.65252
[200]	train's rmse: 3.5622	test's rmse: 3.62015
[300]	train's rmse: 3.50619	test's rmse: 3.60919
[400]	train's rmse: 3.47125	test's rmse: 3.60585
[500]	train's rmse: 3.4426	test's rmse: 3.60497
[600]	train's rmse: 3.41477	test's rmse: 3.60679
[700]	train's rmse: 3.38448	test's rmse: 3.60546
Early stopping, best iteration is:
[519]	train's rmse: 3.4379	test's rmse: 3.60478
Fold  5 RMSE : 3.604778

########################################## mat_depth=6
Starting LightGBM. Train shape: (201917, 373), test shape: (123623, 373)
Training until validation scores don't improve for 200 rounds.
[100]	train's rmse: 3.65695	test's rmse: 3.7275
[200]	train's rmse: 3.58787	test's rmse: 3.68482
[300]	train's rmse: 3.54756	test's rmse: 3.66988
[400]	train's rmse: 3.51852	test's rmse: 3.66346
[500]	train's rmse: 3.49787	test's rmse: 3.66049
[600]	train's rmse: 3.48051	test's rmse: 3.65913
[700]	train's rmse: 3.46436	test's rmse: 3.65813
[800]	train's rmse: 3.45089	test's rmse: 3.65784
[900]	train's rmse: 3.43758	test's rmse: 3.65757
[1000]	train's rmse: 3.42405	test's rmse: 3.65721
[1100]	train's rmse: 3.4089	test's rmse: 3.65758
[1200]	train's rmse: 3.394	test's rmse: 3.6581
Early stopping, best iteration is:
[1053]	train's rmse: 3.41642	test's rmse: 3.65703
Fold  1 RMSE : 3.657026
Training until validation scores don't improve for 200 rounds.
[100]	train's rmse: 3.67067	test's rmse: 3.66697
[200]	train's rmse: 3.59833	test's rmse: 3.6321
[300]	train's rmse: 3.55553	test's rmse: 3.62242
[400]	train's rmse: 3.52952	test's rmse: 3.61795
[500]	train's rmse: 3.50764	test's rmse: 3.61578
[600]	train's rmse: 3.48854	test's rmse: 3.61386
[700]	train's rmse: 3.47125	test's rmse: 3.61314
[800]	train's rmse: 3.45665	test's rmse: 3.61239
[900]	train's rmse: 3.44345	test's rmse: 3.61187
[1000]	train's rmse: 3.42929	test's rmse: 3.61172
[1100]	train's rmse: 3.41409	test's rmse: 3.61169
[1200]	train's rmse: 3.40103	test's rmse: 3.61187
Early stopping, best iteration is:
[1098]	train's rmse: 3.4145	test's rmse: 3.61158
Fold  2 RMSE : 3.611576
Training until validation scores don't improve for 200 rounds.
[100]	train's rmse: 3.67758	test's rmse: 3.63741
[200]	train's rmse: 3.63341	test's rmse: 3.63061
[300]	train's rmse: 3.57149	test's rmse: 3.59787
[400]	train's rmse: 3.54054	test's rmse: 3.5888
[500]	train's rmse: 3.51887	test's rmse: 3.58483
[600]	train's rmse: 3.50037	test's rmse: 3.58329
[700]	train's rmse: 3.48368	test's rmse: 3.5826
[800]	train's rmse: 3.46771	test's rmse: 3.58189
[900]	train's rmse: 3.45279	test's rmse: 3.58191
[1000]	train's rmse: 3.43884	test's rmse: 3.58176
[1100]	train's rmse: 3.42471	test's rmse: 3.58223
Early stopping, best iteration is:
[986]	train's rmse: 3.44052	test's rmse: 3.58162
Fold  3 RMSE : 3.581619
Training until validation scores don't improve for 200 rounds.
[100]	train's rmse: 3.62793	test's rmse: 3.83513
[200]	train's rmse: 3.55562	test's rmse: 3.79927
[300]	train's rmse: 3.5149	test's rmse: 3.7875
[400]	train's rmse: 3.49172	test's rmse: 3.78461
[500]	train's rmse: 3.46768	test's rmse: 3.78156
[600]	train's rmse: 3.44883	test's rmse: 3.78016
[700]	train's rmse: 3.43358	test's rmse: 3.7793
[800]	train's rmse: 3.4191	test's rmse: 3.77904
[900]	train's rmse: 3.40257	test's rmse: 3.77895
[1000]	train's rmse: 3.38801	test's rmse: 3.77862
[1100]	train's rmse: 3.37355	test's rmse: 3.77851
[1200]	train's rmse: 3.35981	test's rmse: 3.77884
Early stopping, best iteration is:
[1029]	train's rmse: 3.38368	test's rmse: 3.77837
Fold  4 RMSE : 3.778370
Training until validation scores don't improve for 200 rounds.
[100]	train's rmse: 3.67248	test's rmse: 3.65245
[200]	train's rmse: 3.59949	test's rmse: 3.61971
[300]	train's rmse: 3.55912	test's rmse: 3.61009
[400]	train's rmse: 3.53255	test's rmse: 3.60726
[500]	train's rmse: 3.51038	test's rmse: 3.60561
[600]	train's rmse: 3.4908	test's rmse: 3.60502
[700]	train's rmse: 3.47255	test's rmse: 3.6046
[800]	train's rmse: 3.45532	test's rmse: 3.60444
[900]	train's rmse: 3.43943	test's rmse: 3.6049
[1000]	train's rmse: 3.42503	test's rmse: 3.60579
Early stopping, best iteration is:
[818]	train's rmse: 3.45263	test's rmse: 3.60432
Fold  5 RMSE : 3.604318
