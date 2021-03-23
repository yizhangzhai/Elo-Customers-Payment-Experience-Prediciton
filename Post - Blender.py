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

#############################################################################
test_df_w_Outliers = pd.read_pickle('test_df_w_Outliers_789.pkl')
test_df_wt_Outliers = pd.read_pickle('test_df_wt_Outliers_17815.pkl')
test_df_Outliers = pd.read_pickle('test_df_Outliers.pkl')

test_submission = test_df_w_Outliers.set_index('card_id').merge(test_df_wt_Outliers.set_index('card_id'),how='outer',left_index=True,right_index=True).merge(test_df_Outliers,how='outer',left_index=True,right_index=True)
test_submission.columns = ['test_df_w_Outliers','test_df_wt_Outliers','test_df_Outliers']

outlier_id = test_df_Outliers.sort_values(by='outlier',ascending=False).head(20000).index
test_submission['target'] = [a if id in outlier_id else b for id,a,b in test_submission.reset_index().iloc[:,:3].values]


test_submission.reset_index()[['card_id', 'target']].to_csv('SUBMISSION_1.csv', index=False)

#############################################################################

test_df_w_Outliers = pd.read_pickle('test_df_w_Outliers.pkl')
test_df_wt_Outliers = pd.read_pickle('test_df_wt_Outliers_17815.pkl')
test_df_Outliers = pd.read_pickle('test_df_Outliers.pkl')

test_submission = test_df_w_Outliers.set_index('card_id').merge(test_df_wt_Outliers.set_index('card_id'),how='outer',left_index=True,right_index=True).merge(test_df_Outliers,how='outer',left_index=True,right_index=True)
test_submission.columns = ['test_df_w_Outliers','test_df_wt_Outliers','test_df_Outliers']

outlier_id = test_df_Outliers.sort_values(by='outlier',ascending=False).head(20000).index
test_submission['target'] = [a if id in outlier_id else b for id,a,b in test_submission.reset_index().iloc[:,:3].values]


test_submission.reset_index()[['card_id', 'target']].to_csv('SUBMISSION_2.csv', index=False)

#############################################################################

test_df_w_Outliers = pd.read_pickle('test_df_w_Outliers_789.pkl')
test_df_wt_Outliers = pd.read_pickle('test_df_wt_Outliers.pkl')
test_df_Outliers = pd.read_pickle('test_df_Outliers.pkl')

test_submission = test_df_w_Outliers.set_index('card_id').merge(test_df_wt_Outliers.set_index('card_id'),how='outer',left_index=True,right_index=True).merge(test_df_Outliers,how='outer',left_index=True,right_index=True)
test_submission.columns = ['test_df_w_Outliers','test_df_wt_Outliers','test_df_Outliers']

outlier_id = test_df_Outliers.sort_values(by='outlier',ascending=False).head(20000).index
test_submission['target'] = [a if id in outlier_id else b for id,a,b in test_submission.reset_index().iloc[:,:3].values]


test_submission.reset_index()[['card_id', 'target']].to_csv('SUBMISSION_3.csv', index=False)

##############################################################################################
test_df_w_Outliers_1 = pd.read_pickle('test_df_w_Outliers_1.pkl')
test_df_w_Outliers_2 = pd.read_pickle('test_df_w_Outliers_2.pkl')
test_df_w_Outliers_3 = pd.read_pickle('test_df_w_Outliers_3.pkl')
test_df_w_Outliers_4 = pd.read_pickle('test_df_w_Outliers_4.pkl')

test_df_w_Outliers_5 = pd.read_pickle('test_df_w_Outliers_5.pkl')
test_df_w_Outliers_6 = pd.read_pickle('test_df_w_Outliers_6.pkl')
test_df_w_Outliers_7 = pd.read_pickle('test_df_w_Outliers_7.pkl')

test_df_w_Outliers_8 = pd.read_pickle('test_df_w_Outliers_8.pkl')
test_df_w_Outliers_9 = pd.read_pickle('test_df_w_Outliers_9.pkl')
test_df_w_Outliers_11 = pd.read_pickle('test_df_w_Outliers_11.pkl')

test_df_w_Outliers_12 = pd.read_pickle('test_df_w_Outliers_12.pkl')
test_df_w_Outliers_13 = pd.read_pickle('test_df_w_Outliers_13.pkl')
test_df_w_Outliers_14 = pd.read_pickle('test_df_w_Outliers_14.pkl')

test_df_w_Outliers_15 = pd.read_pickle('test_df_w_Outliers_15.pkl')
test_df_w_Outliers_16 = pd.read_pickle('test_df_w_Outliers_16.pkl')
test_df_w_Outliers_17 = pd.read_pickle('test_df_w_Outliers_17.pkl')

test_df_w_Outliers_18 = pd.read_pickle('test_df_w_Outliers_18.pkl')
test_df_w_Outliers_19 = pd.read_pickle('test_df_w_Outliers_19.pkl')
test_df_w_Outliers_20 = pd.read_pickle('test_df_w_Outliers_20.pkl')

test_df_w_Outliers_21 = pd.read_pickle('test_df_w_Outliers_21.pkl')
test_df_w_Outliers_22 = pd.read_pickle('test_df_w_Outliers_22.pkl')
test_df_w_Outliers_23 = pd.read_pickle('test_df_w_Outliers_23.pkl')

test_df_w_Outliers_24 = pd.read_pickle('test_df_w_Outliers_24.pkl')
test_df_w_Outliers_25 = pd.read_pickle('test_df_w_Outliers_25.pkl')
test_df_w_Outliers_26 = pd.read_pickle('test_df_w_Outliers_26.pkl')

test_df_w_Outliers_27 = pd.read_pickle('test_df_w_Outliers_27.pkl')
test_df_w_Outliers_28 = pd.read_pickle('test_df_w_Outliers_28.pkl')
test_df_w_Outliers_29 = pd.read_pickle('test_df_w_Outliers_29.pkl')

###
test_df_w_Outliers = [a*0.12+b*0.12+c*0.12+d*0.08+e*0.08+f*0.08+g*0.05+h*0.05+i*0.05+g*0.05+h*0.05+i*0.05+j*0.05+k*0.05\
                      for a,b,c,d,e,f,g,h,i,j,k in zip(test_df_w_Outliers_1.target.values,\
                                                    test_df_w_Outliers_2.target.values,\
                                                    test_df_w_Outliers_3.target.values,\
                                                    test_df_w_Outliers_6.target.values,\
                                                    test_df_w_Outliers_9.target.values,\
                                                    test_df_w_Outliers_13.target.values,\
                                                    test_df_w_Outliers_16.target.values,\
                                                    test_df_w_Outliers_19.target.values,\
                                                    test_df_w_Outliers_22.target.values,\
                                                    test_df_w_Outliers_25.target.values,\
                                                    test_df_w_Outliers_29.target.values)]

test_df_w_Outliers = pd.DataFrame(test_df_w_Outliers)
test_df_w_Outliers['card_id'] = test_df_w_Outliers_1.card_id

test_df_wt_Outliers = pd.read_pickle('test_df_wt_Outliers_17815.pkl')
test_df_Outliers = pd.read_pickle('test_df_Outliers.pkl')

test_submission = test_df_w_Outliers.set_index('card_id').merge(test_df_wt_Outliers.set_index('card_id'),how='outer',left_index=True,right_index=True).merge(test_df_Outliers,how='outer',left_index=True,right_index=True)
test_submission.columns = ['test_df_w_Outliers','test_df_wt_Outliers','test_df_Outliers']

outlier_id = test_df_Outliers.sort_values(by='outlier',ascending=False).head(20000).index
test_submission['target'] = [a if id in outlier_id else b for id,a,b in test_submission.reset_index().iloc[:,:3].values]


test_submission.reset_index()[['card_id', 'target']].to_csv('SUBMISSION_1.csv', index=False)
