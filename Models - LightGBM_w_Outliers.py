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
agg_plus_2_dat = pd.read_pickle('agg_plus_2_dat_2.pkl')
agg_plus_3_dat = pd.read_pickle('agg_plus_3_dat.pkl')
months = pd.read_pickle('months.pkl')

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

def kfold_lightgbm(train_df, test_df, num_folds, params, stratified = False, debug= False,weighted=False,random_state=15):
    FEATS_EXCLUDED = ['first_active_month', 'target', 'card_id', 'outliers', 'index',
                  'hist_purchase_date_max', 'hist_purchase_date_min', 'hist_card_id_size',
                  'new_purchase_date_max', 'new_purchase_date_min', 'new_card_id_size',
                  'OOF_PRED', 'month_0','hist_latest_purchase_date','new_latest_purchase_date','new_earliest_purchase_date']

    if debug== False:
        # keeping=['agg11_hist_mean_1', 'hist_month_nunique', 'delta_purchase_days_2',
           # 'agg11_hist_max_1', 'new_amount_month_ratio_max',
           # 'agg11_hist_sum_1', 'agg10_hist_sum_1', 'hist_duration_min',
           # 'hist_purchase_date_diff', 'new_month_lag_mean',
           # 'hist_weekofyear_nunique', 'hist_authorized_flag_mean',
           # 'hist_month_diff_max',
           # 'new_agg_card_Recency_purchase_date_int_mean',
           # 'hist_month_diff_mean', 'agg14_hist_sum_0', 'agg14_new_max_1',
           # 'agg12_hist_sum_0', 'agg4_hist_mean', 'agg13_hist_sum_0',
           # 'hist_installments_sum', 'agg5_new_var', 'delta_purchase_days_11',
           # 'new_purchase_amount_max', 'new_day_mean', 'new_duration_max',
           # 'hist_category_1_mean', 'agg10_hist_min_1', 'category_1_mean',
           # 'agg11_hist_min_1', 'agg11_hist_mean_0', 'agg11_new_sum_1',
           # 'hist_agg_card_Recency_category_1_nunique',
           # 'hist_agg_card_Recency_purchase_amount_sum', 'agg11_hist_sum_0',
           # 'agg10_new_sum_1', 'agg13_new_var_1', 'agg12_new_sum_1',
           # 'hist_agg_card_Recency_card_id_count',
           # 'hist_agg_card_Recency_purchase_date_int_mean', 'hist_price_var',
           # 'hist_purchase_date_average',
           # 'new_agg_card_Recency_purchase_date_int_max',
           # 'hist_fathers_day_2017_mean',
           # 'hist_agg_card_Recency_installments_sum',
           # 'hist_agg_card_Recency_purchase_date_int_max', 'agg12_new_var_1',
           # 'hist_agg_card_Recency_purchase_amount_max', 'agg10_new_min_1',
           # 'agg11_hist_var_0', 'delta_purchase_days_8', 'agg14_hist_min_0',
           # 'hist_duration_mean', 'hist_hour_mean', 'agg12_hist_var_1',
           # 'agg12_hist_var_0', 'agg4_hist_var',
           # 'new_agg_card_Recency_purchase_amount_max', 'card_id_cnt_ratio',
           # 'hist_last_buy', 'agg11_hist_var_1', 'new_day_max',
           # 'hist_Children_day_2017_mean', 'new_purchase_date_uptonow',
           # 'agg14_hist_max_0', 'agg5_hist_var', 'agg10_hist_max_0',
           # 'hist_first_buy', 'agg10_hist_mean_1', 'days_feature2',
           # 'agg10_hist_var_1', 'hist_agg_card_Recency_authorized_flag_mean',
           # 'new_purchase_date_diff', 'hist_Valentine_Day_2017_mean',
           # 'hist_month_lag_var', 'new_agg_card_Recency_purchase_amount_std',
           # 'hist_merchant_id_nunique', 'agg14_hist_var_0', 'agg7_hist_var',
           # 'agg10_hist_max_1', 'agg12_hist_max_0', 'days_feature1',
           # 'month_diff_ratio', 'hist_weekend_mean', 'agg14_hist_max_1',
           # 'hist_agg_card_Recency_purchase_amount_median',
           # 'new_weekofyear_max', 'hist_weekday_mean', 'agg10_hist_min_0',
           # 'agg13_hist_var_1', 'hist_agg_card_Recency_category_1_sum',
           # 'new_weekofyear_mean', 'agg10_hist_sum_0', 'new_category_1_mean',
           # 'hist_month_lag_skew', 'new_month_lag_max', 'agg6_hist_mean',
           # 'agg13_hist_var_0', 'hist_agg_card_Recency_purchase_amount_mean',
           # 'agg14_new_var_1', 'new_agg_card_Recency_purchase_amount_sum',
           # 'hist_agg_card_Recency_merchant_id_nunique',
           # 'hist_subsector_id_nunique', 'new_category_3_mean_mean',
           # 'new_purchase_date_average', 'hist_month_lag_mean',
           # 'hist_agg_card_Recency_purchase_amount_min',
           # 'hist_agg_card_Recency_purchase_amount_nunique',
           # 'agg12_hist_min_0', 'agg12_hist_max_1', 'days_feature1_ratio',
           # 'hist_price_sum', 'new_month_mean', 'agg11_new_mean_1',
           # 'hist_Mothers_Day_2017_mean', 'hist_day_mean',
           # 'installments_ratio', 'new_duration_var', 'agg10_hist_mean_0',
           # 'agg12_hist_min_1', 'hist_merchant_category_id_nunique',
           # 'agg6_hist_var', 'agg10_new_max_0',
           # 'hist_agg_card_Recency_installments_mean', 'new_last_buy',
           # 'agg14_hist_mean_1', 'agg11_new_sum_0', 'agg10_hist_var_0',
           # 'agg14_hist_mean_0', 'hist_month_diff_var', 'new_hour_mean',
           # 'hist_purchase_amount_skew', 'agg10_new_mean_0',
           # 'hist_amount_month_ratio_skew', 'hist_amount_month_ratio_min',
           # 'hist_CLV', 'hist_agg_card_Recency_category_1_mean',
           # 'hist_price_mean', 'hist_purchase_amount_sum', 'month_lag_mean',
           # 'new_agg_card_Recency_purchase_amount_median', 'agg14_hist_min_1',
           # 'new_first_buy', 'new_amount_month_ratio_var',
           # 'hist_month_diff_skew', 'agg13_hist_sum_1',
           # 'hist_agg_card_Recency_12_mont_lag', 'hist_duration_skew',
           # 'hist_category_3_mean', 'new_agg_card_Recency_purchase_amount_min',
           # 'new_price_var', 'hist_agg_card_Recency_purchase_amount_std',
           # 'agg13_new_sum_1', 'agg5_hist_mean',
           # 'new_agg_card_Recency_purchase_amount_mean', 'agg12_hist_mean_0',
           # 'agg6_new_mean', 'hist_category_3_mean_mean', 'agg14_hist_var_1',
           # 'hist_purchase_date_uptonow', 'month_lag_max',
           # 'hist_installments_mean', 'agg8_hist_mean', 'agg6_new_var',
           # 'hist_Black_Friday_2017_mean', 'agg10_new_min_0',
           # 'agg12_hist_mean_1', 'new_day_min', 'hist_installments_var',
           # 'hist_installments_skew', 'new_Christmas_Day_2017_mean',
           # 'installments_total', 'new_month_max', 'CLV_ratio',
           # 'new_weekofyear_min', 'delta_purchase_days_7',
           # 'hist_purchase_date_uptomin', 'agg12_new_min_1', 'new_price_min',
           # 'agg15_hist_mean_0', 'agg15_hist_mean_1', 'agg15_hist_min_0',
           # 'agg15_hist_min_1', 'agg15_hist_max_0', 'agg15_hist_max_1',
           # 'agg15_hist_sum_0', 'agg15_hist_sum_1', 'agg15_hist_nunique_0',
           # 'agg15_hist_nunique_1', 'agg15_new_mean_0', 'agg15_new_mean_1',
           # 'agg15_new_min_0', 'agg15_new_min_1', 'agg15_new_max_0',
           # 'agg15_new_max_1', 'agg15_new_sum_0', 'agg15_new_sum_1',
           # 'agg15_new_nunique_0', 'agg15_new_nunique_1', 'agg16_hist_mean',
           # 'agg16_hist_var', 'agg16_new_mean', 'agg16_new_var',
           # 'agg17_hist_mean_0', 'agg17_hist_mean_1', 'agg17_hist_min_0',
           # 'agg17_hist_min_1', 'agg17_hist_max_0', 'agg17_hist_max_1',
           # 'agg17_hist_sum_0', 'agg17_hist_sum_1', 'agg17_hist_var_0',
           # 'agg17_hist_var_1', 'agg17_new_mean_0', 'agg17_new_mean_1',
           # 'agg17_new_min_0', 'agg17_new_min_1', 'agg17_new_max_0',
           # 'agg17_new_max_1', 'agg17_new_sum_0', 'agg17_new_sum_1',
           # 'agg17_new_var_0', 'agg17_new_var_1']

          keeping = ['CLV_ratio',
                     'agg10_hist_max_0',
                     'agg10_hist_max_1',
                     'agg10_hist_mean_1',
                     'agg10_hist_min_1',
                     'agg10_hist_sum_0',
                     'agg10_hist_sum_1',
                     'agg10_hist_var_0',
                     'agg10_hist_var_1',
                     'agg10_new_max_0',
                     'agg10_new_mean_0',
                     'agg10_new_min_1',
                     'agg10_new_sum_1',
                     'agg11_hist_max_1',
                     'agg11_hist_mean_0',
                     'agg11_hist_mean_1',
                     'agg11_hist_min_1',
                     'agg11_hist_sum_0',
                     'agg11_hist_sum_1',
                     'agg11_hist_var_0',
                     'agg11_hist_var_1',
                     'agg11_new_sum_0',
                     'agg11_new_sum_1',
                     'agg12_hist_max_0',
                     'agg12_hist_mean_0',
                     'agg12_hist_min_0',
                     'agg12_hist_min_1',
                     'agg12_hist_sum_0',
                     'agg12_hist_var_0',
                     'agg12_hist_var_1',
                     'agg12_new_min_1',
                     'agg12_new_sum_1',
                     'agg12_new_var_1',
                     'agg13_hist_sum_0',
                     'agg13_hist_var_0',
                     'agg13_hist_var_1',
                     'agg13_new_sum_1',
                     'agg13_new_var_1',
                     'agg14_hist_max_0',
                     'agg14_hist_max_1',
                     'agg14_hist_mean_1',
                     'agg14_hist_min_0',
                     'agg14_hist_sum_0',
                     'agg14_hist_var_0',
                     'agg14_hist_var_1',
                     'agg14_new_max_1',
                     'agg14_new_var_1',
                     'agg15_hist_mean_0',
                     'agg15_hist_nunique_1',
                     'agg15_hist_sum_1',
                     'agg16_hist_mean',
                     'agg16_hist_var',
                     'agg17_hist_max_0',
                     'agg17_hist_max_1',
                     'agg17_hist_mean_0',
                     'agg17_hist_min_0',
                     'agg17_hist_min_1',
                     'agg17_hist_sum_0',
                     'agg17_hist_sum_1',
                     'agg17_hist_var_0',
                     'agg17_hist_var_1',
                     'agg17_new_max_0',
                     'agg17_new_max_1',
                     'agg17_new_mean_0',
                     'agg17_new_mean_1',
                     'agg17_new_min_0',
                     'agg17_new_min_1',
                     'agg17_new_sum_0',
                     'agg17_new_sum_1',
                     'agg17_new_var_0',
                     'agg4_hist_mean',
                     'agg4_hist_var',
                     'agg5_hist_var',
                     'agg5_new_var',
                     'agg6_hist_mean',
                     'agg6_new_mean',
                     'agg6_new_var',
                     'agg7_hist_var',
                     'agg8_hist_mean',
                     'card_id_cnt_ratio',
                     'category_1_mean',
                     'days_feature1',
                     'days_feature1_ratio',
                     'days_feature2',
                     'delta_purchase_days_11',
                     'delta_purchase_days_2',
                     'delta_purchase_days_8',
                     'hist_CLV',
                     'hist_Children_day_2017_mean',
                     'hist_Mothers_Day_2017_mean',
                     'hist_Valentine_Day_2017_mean',
                     'hist_agg_card_Recency_authorized_flag_mean',
                     'hist_agg_card_Recency_card_id_count',
                     'hist_agg_card_Recency_category_1_nunique',
                     'hist_agg_card_Recency_category_1_sum',
                     'hist_agg_card_Recency_installments_mean',
                     'hist_agg_card_Recency_installments_sum',
                     'hist_agg_card_Recency_merchant_id_nunique',
                     'hist_agg_card_Recency_purchase_amount_max',
                     'hist_agg_card_Recency_purchase_amount_mean',
                     'hist_agg_card_Recency_purchase_amount_median',
                     'hist_agg_card_Recency_purchase_amount_nunique',
                     'hist_agg_card_Recency_purchase_amount_sum',
                     'hist_agg_card_Recency_purchase_date_int_max',
                     'hist_agg_card_Recency_purchase_date_int_mean',
                     'hist_amount_month_ratio_min',
                     'hist_amount_month_ratio_skew',
                     'hist_authorized_flag_mean',
                     'hist_category_1_mean',
                     'hist_duration_mean',
                     'hist_duration_min',
                     'hist_duration_skew',
                     'hist_fathers_day_2017_mean',
                     'hist_first_buy',
                     'hist_hour_mean',
                     'hist_installments_sum',
                     'hist_last_buy',
                     'hist_merchant_category_id_nunique',
                     'hist_merchant_id_nunique',
                     'hist_month_diff_max',
                     'hist_month_diff_mean',
                     'hist_month_lag_mean',
                     'hist_month_lag_var',
                     'hist_month_nunique',
                     'hist_price_mean',
                     'hist_price_sum',
                     'hist_price_var',
                     'hist_purchase_amount_sum',
                     'hist_purchase_date_average',
                     'hist_purchase_date_diff',
                     'hist_purchase_date_uptonow',
                     'hist_subsector_id_nunique',
                     'hist_weekend_mean',
                     'hist_weekofyear_nunique',
                     'installments_ratio',
                     'month_diff_ratio',
                     'month_lag_max',
                     'month_lag_mean',
                     'new_Christmas_Day_2017_mean',
                     'new_agg_card_Recency_purchase_amount_max',
                     'new_agg_card_Recency_purchase_amount_mean',
                     'new_agg_card_Recency_purchase_amount_min',
                     'new_agg_card_Recency_purchase_amount_std',
                     'new_agg_card_Recency_purchase_amount_sum',
                     'new_agg_card_Recency_purchase_date_int_max',
                     'new_agg_card_Recency_purchase_date_int_mean',
                     'new_amount_month_ratio_max',
                     'new_category_1_mean',
                     'new_category_3_mean_mean',
                     'new_day_max',
                     'new_day_mean',
                     'new_duration_max',
                     'new_duration_var',
                     'new_first_buy',
                     'new_last_buy',
                     'new_month_lag_max',
                     'new_month_lag_mean',
                     'new_month_max',
                     'new_month_mean',
                     'new_price_min',
                     'new_price_var',
                     'new_purchase_amount_max',
                     'new_purchase_date_average',
                     'new_purchase_date_diff',
                     'new_purchase_date_uptonow',
                     'new_weekofyear_max',
                     'new_weekofyear_mean',
                     'new_weekofyear_min',
                    #######################
                    'days_diff_hist_min', 'days_diff_hist_max', 'days_diff_hist_mean',
                           'days_diff_hist_median', 'days_diff_hist_var', 'days_diff_new_min',
                           'days_diff_new_max', 'days_diff_new_mean', 'days_diff_new_median',
                           'days_diff_new_var', 'MAX_hist_month_lag_0',
                    # ########################
                    'agg25_hist_nunique_Y','agg25_hist_nunique_N','agg26_hist_nunique_Y','agg26_hist_nunique_N','agg27_hist_nunique_Y',
                    'agg17_new_sum_Y','agg21_hist_mean','agg28_hist_sum_N','agg17_new_var_N','agg17_hist_sum_Y']

    # Cross validation model
    if stratified:
        folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=random_state)
    else:
        folds = KFold(n_splits= num_folds, shuffle=True, random_state=random_state)

    # Create arrays and dataframes to store results
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in train_df.columns if f not in FEATS_EXCLUDED]
    if debug== False:
        feats = [c for c in feats if c in keeping]

    print("Starting LightGBM. Train shape: {}, test shape: {}".format(len(feats), len(feats)))

    # k-fold
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['outliers'])):
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['target'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['target'].iloc[valid_idx]

        # weight_sub = np.array(weight)[train_idx]
        # set data structure
        if weighted == True:
            lgb_train = lgb.Dataset(train_x,
                                    label=train_y, weight=weight_sub,
                                    free_raw_data=False)
        else:
            lgb_train = lgb.Dataset(train_x,
                                    label=train_y,
                                    free_raw_data=False)
        lgb_test = lgb.Dataset(valid_x,
                               label=valid_y,
                               free_raw_data=False)


        # param_goss ={
        #         'task': 'train',
        #         'boosting': 'goss',
        #         'objective': 'regression',
        #         'metric': 'rmse',
        #         'learning_rate': 0.01,
        #         'subsample': 0.9855232997390695,
        #         'max_depth': 6,
        #         'top_rate': 0.9064148448434349,
        #         'num_leaves': 65,
        #         'min_child_weight': 41.9612869171337,
        #         'other_rate': 0.0721768246018207,
        #         'reg_alpha': 9.677537745007898,
        #         'colsample_bytree': 0.5665320670155495,
        #         'min_split_gain': 9.820197773625843,
        #         'reg_lambda': 8.2532317400459,
        #         'min_data_in_leaf': 21,
        #         'verbose': -1,
        #         'seed':int(2**n_fold),
        #         'bagging_seed':int(2**n_fold),
        #         'drop_seed':int(2**n_fold)
        #         }
        #
        # params_gbdt = {'num_leaves': 300,
        #  'min_data_in_leaf': 149,
        #  'objective':'regression',
        #  'max_depth': 9,
        #  'learning_rate': 0.005,
        #  "boosting": "gbdt",
        #  "feature_fraction": 0.7,
        #  "bagging_freq": 1,
        #  "bagging_fraction": 0.65 ,
        #  "bagging_seed": 11,
        #  "metric": 'rmse',
        #  "lambda_l1": 0.2634,
        #  "random_state": 133,
        #  "verbosity": -1}


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
        return test_df, feature_importance_df

# [ ""+str(k)+"" for k in keeping]
############################    Training    ##############################

df2 = create_dat_combination(dat_list=[df,delta_days_dat,recency_dat,aggs_dat,aggs_plus_dat,agg_plus_2_dat,months])

train_df = df2[df2['target'].notnull()]
test_df = df2[df2['target'].isnull()]
gc.collect()

_, feature_importance_df = kfold_lightgbm(train_df, test_df, num_folds=5, stratified=False, debug=True)


imp = feature_importance_df.groupby('feature')['importance'].mean().reset_index()
imp.importance.describe()

keeping = imp.loc[imp.importance>=10.8].feature
len(keeping)



_,feature_importance_df = kfold_lightgbm(train_df, test_df, num_folds=5, stratified=False, debug=False)

######################################################################################################################
test_df_w_Outliers_1 = kfold_lightgbm(train_df, test_df, num_folds=5, stratified=False, debug=False)

# test_df_w_Outliers_1[0].to_pickle('test_df_w_Outliers.pkl')
test_df_w_Outliers_1[0].to_csv('SUBMISSION.csv',index=False)


test_df_w_Outliers_1[1].loc[test_df_w_Outliers_1[1].fold==1]


Starting LightGBM. Train shape: 168, test shape: 168
Training until validation scores don't improve for 200 rounds.
[100]	train's rmse: 3.6551	test's rmse: 3.72848
[200]	train's rmse: 3.58401	test's rmse: 3.68728
[300]	train's rmse: 3.54391	test's rmse: 3.67134
[400]	train's rmse: 3.51599	test's rmse: 3.6652
[500]	train's rmse: 3.49565	test's rmse: 3.66215
[600]	train's rmse: 3.47814	test's rmse: 3.66031
[700]	train's rmse: 3.46119	test's rmse: 3.65942
[800]	train's rmse: 3.4462	test's rmse: 3.65914
[900]	train's rmse: 3.43331	test's rmse: 3.65942
Early stopping, best iteration is:
[740]	train's rmse: 3.45528	test's rmse: 3.65909
Fold  1 RMSE : 3.659095
Training until validation scores don't improve for 200 rounds.
[100]	train's rmse: 3.67037	test's rmse: 3.66458
[200]	train's rmse: 3.59703	test's rmse: 3.6304
[300]	train's rmse: 3.55499	test's rmse: 3.6192
[400]	train's rmse: 3.52747	test's rmse: 3.61426
[500]	train's rmse: 3.50621	test's rmse: 3.61183
[600]	train's rmse: 3.48762	test's rmse: 3.6105
[700]	train's rmse: 3.47085	test's rmse: 3.6099
[800]	train's rmse: 3.45527	test's rmse: 3.60917
[900]	train's rmse: 3.44138	test's rmse: 3.60875
[1000]	train's rmse: 3.42811	test's rmse: 3.60826
[1100]	train's rmse: 3.41501	test's rmse: 3.60803
[1200]	train's rmse: 3.40117	test's rmse: 3.60762
[1300]	train's rmse: 3.3892	test's rmse: 3.60804
Early stopping, best iteration is:
[1197]	train's rmse: 3.40149	test's rmse: 3.60758
Fold  2 RMSE : 3.607576
Training until validation scores don't improve for 200 rounds.
[100]	train's rmse: 3.67696	test's rmse: 3.63609
[200]	train's rmse: 3.6035	test's rmse: 3.60289
[300]	train's rmse: 3.56082	test's rmse: 3.59221
[400]	train's rmse: 3.53427	test's rmse: 3.58762
[500]	train's rmse: 3.51336	test's rmse: 3.58649
[600]	train's rmse: 3.49572	test's rmse: 3.58531
[700]	train's rmse: 3.47823	test's rmse: 3.58492
[800]	train's rmse: 3.46421	test's rmse: 3.58534
Early stopping, best iteration is:
[685]	train's rmse: 3.48076	test's rmse: 3.58466
Fold  3 RMSE : 3.584657
Training until validation scores don't improve for 200 rounds.
[100]	train's rmse: 3.628	test's rmse: 3.83322
[200]	train's rmse: 3.55305	test's rmse: 3.797
[300]	train's rmse: 3.51296	test's rmse: 3.78516
[400]	train's rmse: 3.48596	test's rmse: 3.78138
[500]	train's rmse: 3.46396	test's rmse: 3.77976
[600]	train's rmse: 3.44674	test's rmse: 3.77854
[700]	train's rmse: 3.43191	test's rmse: 3.77785
[800]	train's rmse: 3.4175	test's rmse: 3.77773
[900]	train's rmse: 3.402	test's rmse: 3.77709
[1000]	train's rmse: 3.38712	test's rmse: 3.77713
Early stopping, best iteration is:
[887]	train's rmse: 3.40412	test's rmse: 3.77705
Fold  4 RMSE : 3.777054
Training until validation scores don't improve for 200 rounds.
[100]	train's rmse: 3.67322	test's rmse: 3.65005
[200]	train's rmse: 3.6	test's rmse: 3.61723
[300]	train's rmse: 3.56185	test's rmse: 3.60821
[400]	train's rmse: 3.53516	test's rmse: 3.60456
[500]	train's rmse: 3.51336	test's rmse: 3.60212
[600]	train's rmse: 3.49478	test's rmse: 3.60126
[700]	train's rmse: 3.47818	test's rmse: 3.6008
[800]	train's rmse: 3.46235	test's rmse: 3.60051
[900]	train's rmse: 3.44878	test's rmse: 3.60078
Early stopping, best iteration is:
[791]	train's rmse: 3.46372	test's rmse: 3.60048
Fold  5 RMSE : 3.600484

Starting LightGBM. Train shape: 179, test shape: 179
Training until validation scores don't improve for 200 rounds.
[100]	train's rmse: 3.65473	test's rmse: 3.72768
[200]	train's rmse: 3.58381	test's rmse: 3.68784
[300]	train's rmse: 3.54219	test's rmse: 3.6724
[400]	train's rmse: 3.51372	test's rmse: 3.66556
[500]	train's rmse: 3.49286	test's rmse: 3.66258
[600]	train's rmse: 3.47527	test's rmse: 3.66078
[700]	train's rmse: 3.4599	test's rmse: 3.6597
[800]	train's rmse: 3.4454	test's rmse: 3.6599
Early stopping, best iteration is:
[697]	train's rmse: 3.46044	test's rmse: 3.65964
Fold  1 RMSE : 3.659638
Training until validation scores don't improve for 200 rounds.
[100]	train's rmse: 3.6706	test's rmse: 3.66652
[200]	train's rmse: 3.59622	test's rmse: 3.6312
[300]	train's rmse: 3.55367	test's rmse: 3.62094
[400]	train's rmse: 3.5268	test's rmse: 3.61624
[500]	train's rmse: 3.50569	test's rmse: 3.61424
[600]	train's rmse: 3.48649	test's rmse: 3.61314
[700]	train's rmse: 3.46939	test's rmse: 3.6126
[800]	train's rmse: 3.45434	test's rmse: 3.61193
[900]	train's rmse: 3.43988	test's rmse: 3.61138
[1000]	train's rmse: 3.4261	test's rmse: 3.61149
[1100]	train's rmse: 3.41288	test's rmse: 3.6112
[1200]	train's rmse: 3.39913	test's rmse: 3.61136
[1300]	train's rmse: 3.38683	test's rmse: 3.61138
Early stopping, best iteration is:
[1134]	train's rmse: 3.40843	test's rmse: 3.61116
Fold  2 RMSE : 3.611163
Training until validation scores don't improve for 200 rounds.
[100]	train's rmse: 3.67643	test's rmse: 3.63608
[200]	train's rmse: 3.60225	test's rmse: 3.60227
[300]	train's rmse: 3.55995	test's rmse: 3.59086
[400]	train's rmse: 3.5334	test's rmse: 3.58634
[500]	train's rmse: 3.51222	test's rmse: 3.58412
[600]	train's rmse: 3.49417	test's rmse: 3.58347
[700]	train's rmse: 3.47863	test's rmse: 3.58329
[800]	train's rmse: 3.46412	test's rmse: 3.58346
Early stopping, best iteration is:
[634]	train's rmse: 3.48884	test's rmse: 3.58305
Fold  3 RMSE : 3.583049
Training until validation scores don't improve for 200 rounds.
[100]	train's rmse: 3.62813	test's rmse: 3.83292
[200]	train's rmse: 3.55298	test's rmse: 3.79644
[300]	train's rmse: 3.51341	test's rmse: 3.78477
[400]	train's rmse: 3.48783	test's rmse: 3.78229
[500]	train's rmse: 3.46356	test's rmse: 3.77933
[600]	train's rmse: 3.44656	test's rmse: 3.77799
[700]	train's rmse: 3.42977	test's rmse: 3.77684
[800]	train's rmse: 3.41425	test's rmse: 3.77687
[900]	train's rmse: 3.40022	test's rmse: 3.77679
[1000]	train's rmse: 3.38671	test's rmse: 3.77678
[1100]	train's rmse: 3.37362	test's rmse: 3.77695
[1200]	train's rmse: 3.36104	test's rmse: 3.77684
Early stopping, best iteration is:
[1027]	train's rmse: 3.38344	test's rmse: 3.77662
Fold  4 RMSE : 3.776620
Training until validation scores don't improve for 200 rounds.
[100]	train's rmse: 3.67341	test's rmse: 3.65
[200]	train's rmse: 3.60011	test's rmse: 3.61729
[300]	train's rmse: 3.56167	test's rmse: 3.60797
[400]	train's rmse: 3.53527	test's rmse: 3.60394
[500]	train's rmse: 3.51459	test's rmse: 3.60449
[600]	train's rmse: 3.49973	test's rmse: 3.60392
Early stopping, best iteration is:
[492]	train's rmse: 3.51541	test's rmse: 3.6023
Fold  5 RMSE : 3.602303


Starting LightGBM. Train shape: 145, test shape: 145
Training until validation scores don't improve for 200 rounds.
[100]	train's rmse: 3.65667	test's rmse: 3.72668
[200]	train's rmse: 3.58652	test's rmse: 3.68513
[300]	train's rmse: 3.54732	test's rmse: 3.67013
[400]	train's rmse: 3.5192	test's rmse: 3.66416
[500]	train's rmse: 3.49934	test's rmse: 3.66141
[600]	train's rmse: 3.4822	test's rmse: 3.6603
[700]	train's rmse: 3.46578	test's rmse: 3.65929
[800]	train's rmse: 3.45219	test's rmse: 3.65893
[900]	train's rmse: 3.43929	test's rmse: 3.65868
[1000]	train's rmse: 3.42583	test's rmse: 3.65834
[1100]	train's rmse: 3.41225	test's rmse: 3.65789
[1200]	train's rmse: 3.39798	test's rmse: 3.65748
[1300]	train's rmse: 3.38523	test's rmse: 3.65741
[1400]	train's rmse: 3.37205	test's rmse: 3.65772
[1500]	train's rmse: 3.3595	test's rmse: 3.65776
Early stopping, best iteration is:
[1310]	train's rmse: 3.38382	test's rmse: 3.65739
Fold  1 RMSE : 3.657386
Training until validation scores don't improve for 200 rounds.
[100]	train's rmse: 3.6708	test's rmse: 3.66383
[200]	train's rmse: 3.59931	test's rmse: 3.62946
[300]	train's rmse: 3.56021	test's rmse: 3.61905
[400]	train's rmse: 3.53388	test's rmse: 3.615
[500]	train's rmse: 3.51297	test's rmse: 3.61304
[600]	train's rmse: 3.49433	test's rmse: 3.61187
[700]	train's rmse: 3.47794	test's rmse: 3.61137
[800]	train's rmse: 3.46341	test's rmse: 3.61068
[900]	train's rmse: 3.44941	test's rmse: 3.61023
[1000]	train's rmse: 3.43508	test's rmse: 3.60996
[1100]	train's rmse: 3.42058	test's rmse: 3.60992
[1200]	train's rmse: 3.40686	test's rmse: 3.60985
[1300]	train's rmse: 3.39281	test's rmse: 3.61023
Early stopping, best iteration is:
[1144]	train's rmse: 3.41515	test's rmse: 3.60952
Fold  2 RMSE : 3.609524
Training until validation scores don't improve for 200 rounds.
[100]	train's rmse: 3.67824	test's rmse: 3.63574
[200]	train's rmse: 3.60722	test's rmse: 3.60163
[300]	train's rmse: 3.5676	test's rmse: 3.59072
[400]	train's rmse: 3.5418	test's rmse: 3.58679
[500]	train's rmse: 3.52193	test's rmse: 3.58483
[600]	train's rmse: 3.50551	test's rmse: 3.58398
[700]	train's rmse: 3.48988	test's rmse: 3.58348
[800]	train's rmse: 3.47586	test's rmse: 3.58388
[900]	train's rmse: 3.46355	test's rmse: 3.5837
Early stopping, best iteration is:
[724]	train's rmse: 3.48668	test's rmse: 3.58341
Fold  3 RMSE : 3.583413
Training until validation scores don't improve for 200 rounds.
[100]	train's rmse: 3.62866	test's rmse: 3.83281
[200]	train's rmse: 3.55575	test's rmse: 3.79669
[300]	train's rmse: 3.51709	test's rmse: 3.78443
[400]	train's rmse: 3.49177	test's rmse: 3.78105
[500]	train's rmse: 3.47212	test's rmse: 3.78081
[600]	train's rmse: 3.45132	test's rmse: 3.77936
[700]	train's rmse: 3.43583	test's rmse: 3.77874
[800]	train's rmse: 3.42126	test's rmse: 3.77846
[900]	train's rmse: 3.40762	test's rmse: 3.77822
[1000]	train's rmse: 3.39275	test's rmse: 3.77858
[1100]	train's rmse: 3.37887	test's rmse: 3.77843
Early stopping, best iteration is:
[902]	train's rmse: 3.40735	test's rmse: 3.7782
Fold  4 RMSE : 3.778202
Training until validation scores don't improve for 200 rounds.
[100]	train's rmse: 3.67468	test's rmse: 3.65126
[200]	train's rmse: 3.61695	test's rmse: 3.63553
[300]	train's rmse: 3.61447	test's rmse: 3.64657
[400]	train's rmse: 3.55928	test's rmse: 3.61214
[500]	train's rmse: 3.53543	test's rmse: 3.60722
[600]	train's rmse: 3.51253	test's rmse: 3.6059
[700]	train's rmse: 3.48921	test's rmse: 3.60571
[800]	train's rmse: 3.46984	test's rmse: 3.60478
[900]	train's rmse: 3.45645	test's rmse: 3.6042
[1000]	train's rmse: 3.4418	test's rmse: 3.60451
[1100]	train's rmse: 3.42871	test's rmse: 3.60491
Early stopping, best iteration is:
[906]	train's rmse: 3.45559	test's rmse: 3.60419
Fold  5 RMSE : 3.604188


test_df_w_Outliers_1_b = kfold_lightgbm(train_df, test_df, num_folds=5, stratified=True, debug=False)
#####################################################################################################
test_df_w_Outliers_1_2 = kfold_lightgbm(train_df, test_df, num_folds=5, stratified=False, debug=False,random_state=8909023)
test_df_w_Outliers_1_3 = kfold_lightgbm(train_df, test_df, num_folds=5, stratified=False, debug=False,random_state=2728752)
test_df_w_Outliers_1_4 = kfold_lightgbm(train_df, test_df, num_folds=5, stratified=False, debug=False,random_state=147741)
test_df_w_Outliers_1_5 = kfold_lightgbm(train_df, test_df, num_folds=5, stratified=False, debug=False,random_state=852852)
test_df_w_Outliers_1_6 = kfold_lightgbm(train_df, test_df, num_folds=5, stratified=False, debug=False,random_state=39696396)
test_df_w_Outliers_1_7 = kfold_lightgbm(train_df, test_df, num_folds=5, stratified=False, debug=False,random_state=86354)
test_df_w_Outliers_1_8 = kfold_lightgbm(train_df, test_df, num_folds=5, stratified=False, debug=False,random_state=5)
test_df_w_Outliers_1_9 = kfold_lightgbm(train_df, test_df, num_folds=5, stratified=False, debug=False,random_state=742)
test_df_w_Outliers_1_10 = kfold_lightgbm(train_df, test_df, num_folds=5, stratified=False, debug=False,random_state=753)
test_df_w_Outliers_1_11 = kfold_lightgbm(train_df, test_df, num_folds=5, stratified=False, debug=False,random_state=4521456)

###########################
test_df_w_Outliers_1_0 = test_df_w_Outliers_1[0].target\
                        +test_df_w_Outliers_1_2[0].target\
                        +test_df_w_Outliers_1_3[0].target\
                        +test_df_w_Outliers_1_4[0].target\
                        +test_df_w_Outliers_1_5[0].target\
                        +test_df_w_Outliers_1_6[0].target\
                        +test_df_w_Outliers_1_7[0].target\
                        +test_df_w_Outliers_1_8[0].target\
                        +test_df_w_Outliers_1_9[0].target\
                        +test_df_w_Outliers_1_10[0].target\
                        +test_df_w_Outliers_1_11[0].target\

test_df_w_Outliers_1_0 = test_df_w_Outliers_1_11



#######################

df2 = create_dat_combination(dat_list=[df,delta_days_dat,aggs_dat,aggs_plus_dat])

train_df = df2[df2['target'].notnull()]
test_df = df2[df2['target'].isnull()]
gc.collect()

test_df_w_Outliers_2 = kfold_lightgbm(train_df, test_df, num_folds=5, stratified=False, debug=False)
test_df_w_Outliers_2_b = kfold_lightgbm(train_df, test_df, num_folds=5, stratified=True, debug=False)


#######################

df2 = create_dat_combination(dat_list=[df,recency_dat,aggs_dat,aggs_plus_dat,agg_plus_2_dat])


train_df = df2[df2['target'].notnull()]
test_df = df2[df2['target'].isnull()]
gc.collect()

test_df_w_Outliers_3 = kfold_lightgbm(train_df, test_df, num_folds=5, stratified=False, debug=False)
test_df_w_Outliers_3_b = kfold_lightgbm(train_df, test_df, num_folds=5, stratified=True, debug=False)

#########################################################################################
test_df_w_Outliers_3_random_2 = kfold_lightgbm(train_df, test_df, num_folds=5, stratified=False, debug=False,random_state=123)
test_df_w_Outliers_3_random_3 = kfold_lightgbm(train_df, test_df, num_folds=5, stratified=False, debug=False,random_state=546234)
test_df_w_Outliers_3_random_4 = kfold_lightgbm(train_df, test_df, num_folds=5, stratified=False, debug=False,random_state=0)
test_df_w_Outliers_3_random_5 = kfold_lightgbm(train_df, test_df, num_folds=5, stratified=False, debug=False,random_state=45612)
test_df_w_Outliers_3_random_6 = kfold_lightgbm(train_df, test_df, num_folds=5, stratified=False, debug=False,random_state=1111)
test_df_w_Outliers_3_random_7 = kfold_lightgbm(train_df, test_df, num_folds=5, stratified=False, debug=False,random_state=7894564)
test_df_w_Outliers_3_random_8 = kfold_lightgbm(train_df, test_df, num_folds=5, stratified=False, debug=False,random_state=234986)
test_df_w_Outliers_3_random_9 = kfold_lightgbm(train_df, test_df, num_folds=5, stratified=False, debug=False,random_state=33331)
test_df_w_Outliers_3_random_10 = kfold_lightgbm(train_df, test_df, num_folds=5, stratified=False, debug=False,random_state=789)

test_df_w_Outliers_3_random_10[0].to_pickle('test_df_w_Outliers_789.pkl')


test_df_w_Outliers_3_random_11 = kfold_lightgbm(train_df, test_df, num_folds=5, stratified=False, debug=False,random_state=565432)


test_df_w_Outliers_3_0 = test_df_w_Outliers_3[0].target\
                        +test_df_w_Outliers_3_random_2[0].target\
                        +test_df_w_Outliers_3_random_3[0].target\
                        +test_df_w_Outliers_3_random_4[0].target\
                        +test_df_w_Outliers_3_random_5[0].target\
                        +test_df_w_Outliers_3_random_6[0].target\
                        +test_df_w_Outliers_3_random_7[0].target\
                        +test_df_w_Outliers_3_random_8[0].target\
                        +test_df_w_Outliers_3_random_9[0].target\
                        +test_df_w_Outliers_3_random_10[0].target\
                        +test_df_w_Outliers_3_random_11[0].target\
test_df_w_Outliers_3_0 = test_df_w_Outliers_3_0/11
#################################################################
# test_df_w_Outliers = test_df_w_Outliers_1.merge(test_df_w_Outliers_1_b,how='outer',on='card_id')\
#                                            .merge(test_df_w_Outliers_2,how='outer',on='card_id').merge(test_df_w_Outliers_2_b,how='outer',on='card_id')\
#                                            .merge(test_df_w_Outliers_3,how='outer',on='card_id').merge(test_df_w_Outliers_3_b,how='outer',on='card_id')\
#
# test_df_w_Outliers = test_df_w_Outliers_1[0].merge(test_df_w_Outliers_2[0],how='outer',on='card_id').merge(test_df_w_Outliers_3[0],how='outer',on='card_id')



# test_df_w_Outliers = test_df_w_Outliers.set_index('card_id')
# test_df_w_Outliers = test_df_w_Outliers.apply(lambda x: x.mean(),axis=1).to_frame('test_df_w_Outliers')
# test_df_w_Outliers.reset_index().to_pickle('test_df_w_Outliers.pkl')

test_df_w_Outliers = test_df_w_Outliers_1[0]
test_df_w_Outliers['target'] = (test_df_w_Outliers_1_0 + test_df_w_Outliers_3_0)/2
test_df_w_Outliers.to_csv('test_df_w_Outliers_SUBMISSION.csv',index=False)
# test_df_w_Outliers.to_pickle('test_df_w_Outliers.pkl')

###################################################################################################################################

        params_goss ={
                'task': 'train',
                'boosting': 'goss',
                'objective': 'regression',
                'metric': 'rmse',
                'learning_rate': 0.01,
                'subsample': 0.9855232997390695,
                'max_depth': 6,
                'top_rate': 0.9064148448434349,
                'num_leaves': 65,
                'min_child_weight': 41.9612869171337,
                'other_rate': 0.0721768246018207,
                'reg_alpha': 9.677537745007898,
                'colsample_bytree': 0.5665320670155495,
                'min_split_gain': 9.820197773625843,
                'reg_lambda': 8.2532317400459,
                'min_data_in_leaf': 21,
                'verbose': -1,
                'seed':int(2**5),
                'bagging_seed':int(2**5),
                'drop_seed':int(2**5)
                }

        params_gbdt = {'num_leaves': 300,
         'min_data_in_leaf': 149,
         'objective':'regression',
         'max_depth': 9,
         'learning_rate': 0.005,
         "boosting": "gbdt",
         "feature_fraction": 0.7,
         "bagging_freq": 1,
         "bagging_fraction": 0.65 ,
         "bagging_seed": 11,
         "metric": 'rmse',
         "lambda_l1": 0.2634,
         "random_state": 133,
         "verbosity": -1}


###################################################################################################################################
df2 = create_dat_combination(dat_list=[df,delta_days_dat,recency_dat,aggs_dat,aggs_plus_dat,agg_plus_2_dat,months])

train_df = df2[df2['target'].notnull()]
test_df = df2[df2['target'].isnull()]

test_df_w_Outliers_1 = kfold_lightgbm(train_df, test_df, num_folds=5, params=params_goss, stratified=False, debug=False,random_state=0)
test_df_w_Outliers_2 = kfold_lightgbm(train_df, test_df, num_folds=5, params=params_gbdt, stratified=False, debug=False,random_state=160)
test_df_w_Outliers_3 = kfold_lightgbm(train_df, test_df, num_folds=5, params=params_gbdt, stratified=False, debug=False,random_state=2450)
test_df_w_Outliers_4 = kfold_lightgbm(train_df, test_df, num_folds=5, params=params_goss, stratified=False, debug=False,random_state=35670)

df2 = create_dat_combination(dat_list=[df,delta_days_dat,recency_dat,aggs_dat,aggs_plus_dat,months])

train_df = df2[df2['target'].notnull()]
test_df = df2[df2['target'].isnull()]

test_df_w_Outliers_5 = kfold_lightgbm(train_df, test_df, num_folds=5, params=params_goss, stratified=False, debug=False,random_state=460)
test_df_w_Outliers_6 = kfold_lightgbm(train_df, test_df, num_folds=5, params=params_gbdt, stratified=False, debug=False,random_state=52340)
test_df_w_Outliers_7 = kfold_lightgbm(train_df, test_df, num_folds=5, params=params_goss, stratified=False, debug=False,random_state=64320)


df2 = create_dat_combination(dat_list=[df,delta_days_dat,recency_dat,aggs_dat,aggs_plus_dat,agg_plus_2_dat])

train_df = df2[df2['target'].notnull()]
test_df = df2[df2['target'].isnull()]

test_df_w_Outliers_8 = kfold_lightgbm(train_df, test_df, num_folds=5, params=params_gbdt, stratified=False, debug=False,random_state=750)
test_df_w_Outliers_9 = kfold_lightgbm(train_df, test_df, num_folds=5, params=params_goss, stratified=False, debug=False,random_state=812340)
test_df_w_Outliers_11 = kfold_lightgbm(train_df, test_df, num_folds=5, params=params_goss, stratified=False, debug=False,random_state=110)

df2 = create_dat_combination(dat_list=[df,delta_days_dat,recency_dat,aggs_dat,aggs_plus_dat])

train_df = df2[df2['target'].notnull()]
test_df = df2[df2['target'].isnull()]

test_df_w_Outliers_12 = kfold_lightgbm(train_df, test_df, num_folds=5, params=params_gbdt, stratified=False, debug=False,random_state=8900)
test_df_w_Outliers_13 = kfold_lightgbm(train_df, test_df, num_folds=5, params=params_goss, stratified=False, debug=False,random_state=6785)
test_df_w_Outliers_14 = kfold_lightgbm(train_df, test_df, num_folds=5, params=params_gbdt, stratified=False, debug=False,random_state=12983)


df2 = create_dat_combination(dat_list=[df,delta_days_dat,recency_dat,aggs_plus_dat,months])

train_df = df2[df2['target'].notnull()]
test_df = df2[df2['target'].isnull()]

test_df_w_Outliers_15 = kfold_lightgbm(train_df, test_df, num_folds=5, params=params_goss, stratified=False, debug=False,random_state=901)
test_df_w_Outliers_16 = kfold_lightgbm(train_df, test_df, num_folds=5, params=params_gbdt, stratified=False, debug=False,random_state=432)
test_df_w_Outliers_17 = kfold_lightgbm(train_df, test_df, num_folds=5, params=params_gbdt, stratified=False, debug=False,random_state=523)

df2 = create_dat_combination(dat_list=[df,delta_days_dat,recency_dat])

train_df = df2[df2['target'].notnull()]
test_df = df2[df2['target'].isnull()]

test_df_w_Outliers_18 = kfold_lightgbm(train_df, test_df, num_folds=5, params=params_goss, stratified=False, debug=False,random_state=5423)
test_df_w_Outliers_19 = kfold_lightgbm(train_df, test_df, num_folds=5, params=params_gbdt, stratified=False, debug=False,random_state=75645)
test_df_w_Outliers_20 = kfold_lightgbm(train_df, test_df, num_folds=5, params=params_gbdt, stratified=False, debug=False,random_state=6352)

df2 = create_dat_combination(dat_list=[df,aggs_dat,aggs_plus_dat,agg_plus_2_dat,months])

train_df = df2[df2['target'].notnull()]
test_df = df2[df2['target'].isnull()]

test_df_w_Outliers_21 = kfold_lightgbm(train_df, test_df, num_folds=10, params=params_goss, stratified=False, debug=False,random_state=4321)
test_df_w_Outliers_22 = kfold_lightgbm(train_df, test_df, num_folds=10, params=params_goss, stratified=False, debug=False,random_state=4212)
test_df_w_Outliers_23 = kfold_lightgbm(train_df, test_df, num_folds=10, params=params_gbdt, stratified=False, debug=False,random_state=12)

df2 = create_dat_combination(dat_list=[df,delta_days_dat,aggs_plus_dat,agg_plus_2_dat,months])

train_df = df2[df2['target'].notnull()]
test_df = df2[df2['target'].isnull()]

test_df_w_Outliers_24 = kfold_lightgbm(train_df, test_df, num_folds=10, params=params_goss, stratified=False, debug=False,random_state=61)
test_df_w_Outliers_25 = kfold_lightgbm(train_df, test_df, num_folds=10, params=params_goss, stratified=False, debug=False,random_state=111)
test_df_w_Outliers_26 = kfold_lightgbm(train_df, test_df, num_folds=10, params=params_gbdt, stratified=False, debug=False,random_state=222)

df2 = create_dat_combination(dat_list=[df,recency_dat,aggs_dat,agg_plus_2_dat,months])
train_df = df2[df2['target'].notnull()]
test_df = df2[df2['target'].isnull()]

test_df_w_Outliers_27 = kfold_lightgbm(train_df, test_df, num_folds=10, params=params_goss, stratified=False, debug=False,random_state=15)
test_df_w_Outliers_28 = kfold_lightgbm(train_df, test_df, num_folds=10, params=params_goss, stratified=False, debug=False,random_state=7788)
test_df_w_Outliers_28 = kfold_lightgbm(train_df, test_df, num_folds=10, params=params_goss, stratified=False, debug=False,random_state=7788)
