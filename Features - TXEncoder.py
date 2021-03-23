import pandas as pd
import numpy as np
from keras.layers import *
from keras.models import *
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from sklearn.model_selection import KFold

historical_transactions = pd.read_csv('historical_transactions.csv', parse_dates=['purchase_date'])
new_transactions = pd.read_csv('new_merchant_transactions.csv',parse_dates=['purchase_date'])
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

historical_transactions = historical_transactions.sort_values(by=['card_id','purchase_date'])
new_transactions = new_transactions.sort_values(by=['card_id','purchase_date'])

historical_transactions['month'] = historical_transactions.purchase_date.dt.month
historical_transactions['day'] = historical_transactions.purchase_date.dt.day
historical_transactions['year'] = historical_transactions.purchase_date.dt.year
historical_transactions['weekofyear'] = historical_transactions.purchase_date.dt.weekofyear
historical_transactions['weekday'] = historical_transactions.purchase_date.dt.weekday

hist = historical_transactions.groupby('card_id')['card_id','purchase_amount','authorized_flag', 'city_id', 'category_1','category_3',\
                                                  'merchant_category_id','merchant_id','category_2','state_id','subsector_id',\
                                                  'month','day','year','weekofyear','weekday','month_lag'].tail(100)

order = hist.groupby('card_id').cumcount()+1
hist['order'] = order

authorized_flag = hist.pivot(index='card_id',columns='order',values='authorized_flag')
city_id = hist.pivot(index='card_id',columns='order',values='city_id')
category_1 = hist.pivot(index='card_id',columns='order',values='category_1')
category_3 = hist.pivot(index='card_id',columns='order',values='category_3')
merchant_category_id = hist.pivot(index='card_id',columns='order',values='merchant_category_id')
merchant_id = hist.pivot(index='card_id',columns='order',values='merchant_id')
category_2 = hist.pivot(index='card_id',columns='order',values='category_2')
state_id = hist.pivot(index='card_id',columns='order',values='state_id')
subsector_id = hist.pivot(index='card_id',columns='order',values='subsector_id')

month = hist.pivot(index='card_id',columns='order',values='month')
day = hist.pivot(index='card_id',columns='order',values='day')
year = hist.pivot(index='card_id',columns='order',values='year')
weekofyear = hist.pivot(index='card_id',columns='order',values='weekofyear')
weekday = hist.pivot(index='card_id',columns='order',values='weekday')
month_lag = hist.pivot(index='card_id',columns='order',values='month_lag')
purchase_amount = hist.pivot(index='card_id',columns='order',values='purchase_amount')


month.values




###########################################################################################################
Input_purchase_amount = Input(shape=(purchase_amount.shape[1],1))
Input_month_lag = Input(shape=(month_lag.shape[1],1))


Input_authorized_flag_0 = Input(shape=(authorized_flag.shape[1],1))
Embedding_authorized_flag = Embedding(input_dim=historical_transactions.authorized_flag.nunique(),output_dim=3)(Input_authorized_flag_0)
Embedding_authorized_flag = Reshape(target_shape=(100,3))(Embedding_authorized_flag)
Input_authorized_flag = BatchNormalization()(Embedding_authorized_flag)

Input_city_id_0 = Input(shape=(city_id.shape[1],1))
Embedding_city_id = Embedding(input_dim=historical_transactions.city_id.nunique(),output_dim=10)(Input_city_id_0)
Embedding_city_id = Reshape(target_shape=(100,10))(Embedding_city_id)
Input_city_id = BatchNormalization()(Embedding_city_id)

Input_category_1_0 = Input(shape=(category_1.shape[1],1))
Embedding_category_1 = Embedding(input_dim=historical_transactions.category_1.nunique(),output_dim=5)(Input_category_1_0)
Embedding_category_1 = Reshape(target_shape=(100,5))(Embedding_category_1)
Input_category_1 = BatchNormalization()(Embedding_category_1)

Input_category_2_0 = Input(shape=(category_2.shape[1],1))
Embedding_category_2 = Embedding(input_dim=historical_transactions.category_2.nunique(),output_dim=5)(Input_category_2_0)
Embedding_category_2 = Reshape(target_shape=(100,5))(Embedding_category_2)
Input_category_2 = BatchNormalization()(Embedding_category_2)

Input_category_3_0 = Input(shape=(category_3.shape[1],1))
Embedding_category_3 = Embedding(input_dim=historical_transactions.category_3.nunique(),output_dim=5)(Input_category_3_0)
Embedding_category_3 = Reshape(target_shape=(100,5))(Embedding_category_3)
Input_category_3 = BatchNormalization()(Embedding_category_3)

Input_merchant_category_id_0 = Input(shape=(merchant_category_id.shape[1],1))
Embedding_merchant_category_id = Embedding(input_dim=historical_transactions.merchant_category_id.nunique(),output_dim=30)(Input_merchant_category_id_0)
Embedding_merchant_category_id = Reshape(target_shape=(100,30))(Embedding_merchant_category_id)
Input_merchant_category_id = BatchNormalization()(Embedding_merchant_category_id)

Input_merchant_id_0 = Input(shape=(merchant_id.shape[1],1))
Embedding_merchant_id = Embedding(input_dim=historical_transactions.merchant_id.nunique(),output_dim=100)(Input_merchant_id_0)
Embedding_merchant_id = Reshape(target_shape=(100,100))(Embedding_merchant_id)
Input_merchant_id = BatchNormalization()(Embedding_merchant_id)

Input_state_id_0 = Input(shape=(state_id.shape[1],1))
Embedding_state_id = Embedding(input_dim=historical_transactions.state_id.nunique(),output_dim=8)(Input_state_id_0)
Embedding_state_id = Reshape(target_shape=(100,8))(Embedding_state_id)
Input_state_id = BatchNormalization()(Embedding_state_id)

Input_subsector_id_0 = Input(shape=(subsector_id.shape[1],1))
Embedding_subsector_id = Embedding(input_dim=historical_transactions.subsector_id.nunique(),output_dim=6)(Input_subsector_id_0)
Embedding_subsector_id = Reshape(target_shape=(100,6))(Embedding_subsector_id)
Input_subsector_id = BatchNormalization()(Embedding_subsector_id)

##############################################################################
merged = Concatenate(axis=2)([Input_purchase_amount,Input_month_lag,Input_authorized_flag,Input_city_id,\
                              Input_category_1,Input_category_2,Input_category_3,Input_merchant_category_id,\
                              Input_merchant_id,Input_state_id,Input_subsector_id])

model = Conv1D(32,5,padding='same')(merged)
model = BatchNormalization()(model)
model = Activation('relu')(model)

model = Conv1D(24,3,padding='same')(model)
model = BatchNormalization()(model)
model = Activation('relu')(model)

model = Flatten()(model)

model = Dense(40,activation='relu')(model)
model = BatchNormalization()(model)
model = Dropout(0.2)(model)

out = Dense(1)(model)

model = Model(inputs=[Input_purchase_amount,Input_month_lag,Input_authorized_flag_0,Input_city_id_0,\
                      Input_category_1_0,Input_category_2_0,Input_category_3_0,Input_merchant_category_id_0,\
                      Input_merchant_id_0,Input_state_id_0,Input_subsector_id_0],outputs=out)
model.summary()


model.compile(loss='mean_squared_error',optimizer='Adam')

models.fit([df['purchase_amount'].values,
            df['month_lag'].values,
            df['authorized_flag'].values,
            df['city_id'].values,
            df['category_1'].values,
            df['category_2'].values,
            df['category_3'].values,
            df['merchant_category_id'].values,
            df['merchant_id'].values,
            df['state_id'].values,
            df['subsector_id'].values],
            purchase_amount, batch_size=5120,epochs=1,verbose=1)


###########################################################################################################
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
