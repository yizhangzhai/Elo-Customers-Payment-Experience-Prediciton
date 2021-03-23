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

####
df = pd.read_pickle('df.pkl')
df = df.set_index('card_id')
delta_days_dat = pd.read_pickle('delta_days_dat.pkl')
recency_dat = pd.read_pickle('recency_dat.pkl')
aggs_dat = pd.read_pickle('aggs_dat.pkl')
aggs_plus_dat = pd.read_pickle('aggs_plus_dat.pkl')
agg_plus_2_dat = pd.read_pickle('agg_plus_2_dat.pkl')

######
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def display_importances(feature_importance_df_):
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:40].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]

    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig('lgbm_importances.png')

def create_dat_combination(dat_list):
    df2 = dat_list[0]
    i=1
    while i<len(dat_list):
        df2 = df2.reset_index().merge(dat_list[i].reset_index(),how='outer',on='card_id').set_index('card_id')
        i+=1
    return df2

def kfold_lightgbm(train_df, test_df, num_folds, stratified = False, debug= False):
    FEATS_EXCLUDED = ['first_active_month', 'target', 'card_id', 'outliers', 'index',
                  'hist_purchase_date_max', 'hist_purchase_date_min', 'hist_card_id_size',
                  'new_purchase_date_max', 'new_purchase_date_min', 'new_card_id_size',
                  'OOF_PRED', 'month_0']

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
           'agg15_hist_mean_0', 'agg15_hist_mean_1', 'agg15_hist_min_0',
           'agg15_hist_min_1', 'agg15_hist_max_0', 'agg15_hist_max_1',
           'agg15_hist_sum_0', 'agg15_hist_sum_1', 'agg15_hist_nunique_0',
           'agg15_hist_nunique_1', 'agg15_new_mean_0', 'agg15_new_mean_1',
           'agg15_new_min_0', 'agg15_new_min_1', 'agg15_new_max_0',
           'agg15_new_max_1', 'agg15_new_sum_0', 'agg15_new_sum_1',
           'agg15_new_nunique_0', 'agg15_new_nunique_1', 'agg16_hist_mean',
           'agg16_hist_var', 'agg16_new_mean', 'agg16_new_var',
           'agg17_hist_mean_0', 'agg17_hist_mean_1', 'agg17_hist_min_0',
           'agg17_hist_min_1', 'agg17_hist_max_0', 'agg17_hist_max_1',
           'agg17_hist_sum_0', 'agg17_hist_sum_1', 'agg17_hist_var_0',
           'agg17_hist_var_1', 'agg17_new_mean_0', 'agg17_new_mean_1',
           'agg17_new_min_0', 'agg17_new_min_1', 'agg17_new_max_0',
           'agg17_new_max_1', 'agg17_new_sum_0', 'agg17_new_sum_1',
           'agg17_new_var_0', 'agg17_new_var_1']


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



############################    Training    ##############################

df2 = create_dat_combination(dat_list=[df,delta_days_dat,recency_dat,aggs_dat,aggs_plus_dat,agg_plus_2_dat])

train_df = df2[df2['target'].notnull()]
test_df = df2[df2['target'].isnull()]
gc.collect()

test_df_wt_Outliers_1 = kfold_lightgbm(train_df.loc[train_df.target>-30], test_df, num_folds=5, stratified=False, debug=False)
test_df_wt_Outliers_1_b = kfold_lightgbm(train_df.loc[train_df.target>-30], test_df, num_folds=5, stratified=True, debug=False)
#######################

df2 = create_dat_combination(dat_list=[df,delta_days_dat,recency_dat,aggs_dat,aggs_plus_dat])

train_df = df2[df2['target'].notnull()]
test_df = df2[df2['target'].isnull()]
gc.collect()

test_df_wt_Outliers_2 = kfold_lightgbm(train_df.loc[train_df.target>-30], test_df, num_folds=5, stratified=False, debug=False)
test_df_wt_Outliers_2_b = kfold_lightgbm(train_df.loc[train_df.target>-30], test_df, num_folds=5, stratified=True, debug=False)


#######################

df2 = create_dat_combination(dat_list=[df,delta_days_dat,recency_dat,aggs_dat,agg_plus_2_dat])


train_df = df2[df2['target'].notnull()]
test_df = df2[df2['target'].isnull()]
gc.collect()

test_df_wt_Outliers_3 = kfold_lightgbm(train_df.loc[train_df.target>-30], test_df, num_folds=5, stratified=False, debug=False)
test_df_wt_Outliers_3_b = kfold_lightgbm(train_df.loc[train_df.target>-30], test_df, num_folds=5, stratified=True, debug=False)



#################################################################
test_df_wt_Outliers = test_df_wt_Outliers_1.merge(test_df_wt_Outliers_1_b,how='outer',on='card_id')\
                                           .merge(test_df_wt_Outliers_2,how='outer',on='card_id').merge(test_df_wt_Outliers_2_b,how='outer',on='card_id')\
                                           .merge(test_df_wt_Outliers_3,how='outer',on='card_id').merge(test_df_wt_Outliers_3_b,how='outer',on='card_id')\

# test_df_wt_Outliers = test_df_wt_Outliers_1.merge(test_df_wt_Outliers_2,how='outer',on='card_id').merge(test_df_wt_Outliers_3,how='outer',on='card_id')

test_df_wt_Outliers = test_df_wt_Outliers.set_index('card_id')
test_df_wt_Outliers = test_df_wt_Outliers.apply(lambda x: x.mean(),axis=1).to_frame('test_df_wt_Outliers')
test_df_wt_Outliers.to_pickle('test_df_wt_Outliers.pkl')
