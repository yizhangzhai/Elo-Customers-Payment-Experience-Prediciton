import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.model_selection import KFold
import warnings
import time
import sys
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.set_option('display.max_columns', 500)
from sklearn.model_selection import train_test_split
import itertools
##########

def binarize(df):
    for col in ['authorized_flag', 'category_1']:
        df[col] = df[col].map({'Y':1, 'N':0})
    return df

def encoding_purchase_date(dat):
    dat['month_diff'] = ((datetime.datetime.today() - dat['purchase_date']).dt.days)//30
    dat['month_diff'] += dat['month_lag']
    dat['purchase_month'] = dat['purchase_date'].dt.month
    dat['purchase_date_last'] = dat.groupby('card_id')['purchase_date'].shift(1)
    dat['purchase_amount_last'] = dat.groupby('card_id')['purchase_amount'].shift(1)
    dat['purchase_date_gap'] = dat.purchase_date - dat.purchase_date_last
    dat['purchase_amount_gap'] = dat.purchase_amount - dat.purchase_amount_last
    dat['purchase_date_gap'].fillna(0,inplace=True)
    dat['purchase_amount_gap'].fillna(0,inplace=True)
    return dat

def delta_purchase_date():
    temp1 = historical_transactions.groupby('card_id')['purchase_date'].max().to_frame('hist_latest_purchase_date')
    temp2 = new_transactions.groupby('card_id')['purchase_date'].min().to_frame('new_earliest_purchase_date')
    temp = temp1.merge(temp2,how='left',left_index=True,right_index=True)
    temp["delta_purchase_days"] = temp.new_earliest_purchase_date - temp.hist_latest_purchase_date
    temp["delta_purchase_days"] = temp["delta_purchase_days"].dt.days
    temp["delta_purchase_days"].fillna(9999,inplace=True)
    return temp

def category_dummies():
    historical_transactions = pd.get_dummies(historical_transactions, columns=['category_2','category_3','category_1_2','category_1_3','category_1_2_3','category_1_auth'])
    new_transactions = pd.get_dummies(new_transactions, columns=['category_2', 'category_3','category_1_2','category_1_3','category_1_2_3','category_1_auth'])
    return historical_transactions,new_transactions

def authorized_mean():
    agg_fun = {'authorized_flag': ['mean']}
    auth_mean = historical_transactions.groupby(['card_id']).agg(agg_fun)
    auth_mean.columns = ['_'.join(col).strip() for col in auth_mean.columns.values]
    auth_mean.reset_index(inplace=True)
    return auth_mean

def aggregate_per_month(history):
    grouped = history.groupby(['card_id', 'month_lag'])

    agg_func = {
            'purchase_amount': ['count', 'sum', 'mean', 'min', 'max', 'std'],
            'installments': ['count', 'sum', 'mean', 'min', 'max', 'std'],
            }

    intermediate_group = grouped.agg(agg_func)
    intermediate_group.columns = ['_'.join(col).strip() for col in intermediate_group.columns.values]
    intermediate_group.reset_index(inplace=True)

    final_group = intermediate_group.groupby('card_id').agg(['mean', 'std'])
    final_group.columns = ['_'.join(col).strip() for col in final_group.columns.values]
    final_group.reset_index(inplace=True)

    return final_group


def main():
    ### Read Data
    train = read_data('D:/ML/Elo/train.csv')
    test = read_data('D:/ML/Elo/test.csv')

    target = train['target']
    del train['target']
    historical_transactions = pd.read_csv('D:/ML/Elo/historical_transactions.csv',parse_dates=['purchase_date'])
    new_transactions = pd.read_csv('D:/ML/Elo/new_merchant_transactions.csv',parse_dates=['purchase_date'])

    ###
    historical_transactions = binarize(historical_transactions)
    new_transactions = binarize(new_transactions)

    historical_transactions = encoding_purchase_date(historical_transactions)
    new_transactions = encoding_purchase_date(new_transactions)

    DATA_delta_purchase_date = delta_purchase_date()
    DATA_authorized_mean = authorized_mean()


















if __name__ == 'main':
    main()
