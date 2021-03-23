import pandas as pd
import matplotlib.pyplot as plt
import datetime
import numpy as np
from gensim.models import Word2Vec, KeyedVectors
from gensim.test.utils import get_tmpfile
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import LSTM, Bidirectional, GRU, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Activation, Reshape
from keras.layers.embeddings import Embedding
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import itertools
import gc

#Read data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
historical_transactions = pd.read_csv('historical_transactions.csv')
merchants = pd.read_csv('merchants.csv')
new_merchant_transactions = pd.read_csv('new_merchant_transactions.csv')
sample_submission = pd.read_csv('sample_submission.csv')

###         merchants_id
#####################################################################################
historical_transactions['merchants_records'] = [str(a)+'_'+str(b) for a,b in historical_transactions[['merchant_id','authorized_flag']].values]
historical_transactions = historical_transactions.sort_values(by=['card_id','purchase_date'])
merchants_seq = historical_transactions.groupby('card_id')['merchants_records'].apply(lambda x: list(x))
merchants_seq = merchants_seq.to_frame().reset_index()
temp = [len(x) for x in merchants_seq.merchants_records]

max_records = 120

path = get_tmpfile("hist_card_merchants_2Vec.kv")
w2v = Word2Vec(merchants_seq['merchants_records'], size=max_records, window=5, min_count=3, workers=8, sg=0)
w2v.wv.save(path)
# w2v = KeyedVectors.load("hist_card_merchants_2Vec.kv", mmap='r')

censor_until_months = 3
holder = historical_transactions.loc[abs(historical_transactions.month_lag)+1 <= censor_until_months][['card_id','merchants_records']]
keys = []
subs = []
for key, sub in holder.groupby('card_id'):
        v = sub.merchants_records.value_counts().reset_index().as_matrix().tolist()
        if len(v) > 0:
            subs.append(v)
            keys.append([key]*len(v))

del holder
temp = pd.DataFrame(np.concatenate(subs))
temp['card_id'] = list(itertools.chain(*keys))
temp.columns = ['merchants_visited','weights','card_id']

embed = pd.DataFrame(w2v.wv.syn0)
embed.columns = embed.columns.astype(str)
embed.columns = ['merchants_embedding_'+str(x) for x in embed.columns]
embed['merchants'] = w2v.wv.index2word

temp = temp.merge(embed, how='left', left_on='merchants_visited', right_on='merchants')
temp.drop(['merchants'],axis=1, inplace=True)
temp['weights'] = temp.weights.astype(int)

for i in range(120):
    temp['merchants_embedding_'+str(i)] = temp.weights * temp['merchants_embedding_'+str(i)]

temp = temp.groupby('card_id').apply(lambda x: x.iloc[:,3:].sum()/x.iloc[:,2].shape[0]).reset_index()

temp.to_pickle('hist_merchants2vec.pkl')

###########################################################################

new_merchant_transactions['merchants_records'] = [str(a)+'_'+str(b) for a,b in new_merchant_transactions[['merchant_id','authorized_flag']].values]
new_merchant_transactions = new_merchant_transactions.sort_values(by=['card_id','purchase_date'])
merchants_seq = new_merchant_transactions.groupby('card_id')['merchants_records'].apply(lambda x: list(x))
merchants_seq = merchants_seq.to_frame().reset_index()
temp_new = [len(x) for x in merchants_seq.merchants_records]
max_records = 80

path = get_tmpfile("new_card_merchants_2Vec.kv")
w2v_new = Word2Vec(merchants_seq['merchants_records'], size=max_records, window=5, min_count=3, workers=8, sg=0)
w2v_new.wv.save(path)
# w2v = KeyedVectors.load("new_card_merchants_2Vec.kv", mmap='r')

holder = new_merchant_transactions[['card_id','merchants_records']]
keys = []
subs = []
for key, sub in holder.groupby('card_id'):
        v = sub.merchants_records.value_counts().reset_index().as_matrix().tolist()
        if len(v) > 0:
            subs.append(v)
            keys.append([key]*len(v))

del holder
temp_new = pd.DataFrame(np.concatenate(subs))
temp_new['card_id'] = list(itertools.chain(*keys))
temp_new.columns = ['merchants_visited','weights','card_id']

embed_new = pd.DataFrame(w2v_new.wv.syn0)
embed_new.columns = embed_new.columns.astype(str)
embed_new.columns = ['merchants_embedding_'+str(x) for x in embed_new.columns]
embed_new['merchants'] = w2v_new.wv.index2word

temp_new = temp_new.merge(embed, how='left', left_on='merchants_visited', right_on='merchants')
temp_new.drop(['merchants'],axis=1, inplace=True)
temp_new['weights'] = temp_new.weights.astype(int)

for i in range(120):
    temp_new['merchants_embedding_'+str(i)] = temp_new.weights * temp_new['merchants_embedding_'+str(i)]

temp_new = temp_new.groupby('card_id').apply(lambda x: x.iloc[:,3:].sum()/x.iloc[:,2].shape[0]).reset_index()
temp_new.to_pickle('new_merchants2vec.pkl')

###         subsector_id
#####################################################################################
historical_transactions = historical_transactions.sort_values(by=['card_id','purchase_date'])
merchants_seq = historical_transactions.groupby('card_id')['subsector_id'].apply(lambda x: list(str(x)))
merchants_seq = merchants_seq.to_frame().reset_index()

max_records = 30

path = get_tmpfile("hist_card_subsector_2Vec.kv")
w2v = Word2Vec(merchants_seq['subsector_id'], size=max_records, window=5, min_count=3, workers=8, sg=0)
w2v.wv.save(path)
# w2v = KeyedVectors.load("hist_card_merchants_2Vec.kv", mmap='r')

censor_until_months = 3
holder = historical_transactions.loc[abs(historical_transactions.month_lag)+1 <= censor_until_months][['card_id','subsector_id']]
keys = []
subs = []
for key, sub in holder.groupby('card_id'):
        v = sub.subsector_id.value_counts().reset_index().as_matrix().tolist()
        if len(v) > 0:
            subs.append(v)
            keys.append([key]*len(v))

del holder
temp = pd.DataFrame(np.concatenate(subs))
temp['card_id'] = list(itertools.chain(*keys))
temp.columns = ['subsector_visited','weights','card_id']

embed = pd.DataFrame(w2v.wv.syn0)
embed.columns = embed.columns.astype(str)
embed.columns = ['subsector_embedding_'+str(x) for x in embed.columns]
embed['subsector_id'] = w2v.wv.index2word

temp = temp.merge(embed, how='left', left_on='subsector_visited', right_on='subsector_id')
temp.drop(['subsector_id'],axis=1, inplace=True)
temp['weights'] = temp.weights.astype(int)

for i in range(30):
    temp['subsector_embedding_'+str(i)] = temp.weights * temp['subsector_embedding_'+str(i)]

temp = temp.groupby('card_id').apply(lambda x: x.iloc[:,3:].sum()/x.iloc[:,2].shape[0]).reset_index()

temp.to_pickle('hist_subsector2vec.pkl')

###########################################################################
new_merchant_transactions = new_merchant_transactions.sort_values(by=['card_id','purchase_date'])
merchants_seq = new_merchant_transactions.groupby('card_id')['subsector_id'].apply(lambda x: list(str(x)))
merchants_seq = merchants_seq.to_frame().reset_index()

max_records = 30

path = get_tmpfile("new_card_subsector_2Vec.kv")
w2v = Word2Vec(merchants_seq['subsector_id'], size=max_records, window=5, min_count=3, workers=8, sg=0)
w2v.wv.save(path)
# w2v = KeyedVectors.load("hist_card_merchants_2Vec.kv", mmap='r')

holder = new_merchant_transactions[['card_id','subsector_id']]
keys = []
subs = []
for key, sub in holder.groupby('card_id'):
        v = sub.subsector_id.value_counts().reset_index().as_matrix().tolist()
        if len(v) > 0:
            subs.append(v)
            keys.append([key]*len(v))

del holder
temp = pd.DataFrame(np.concatenate(subs))
temp['card_id'] = list(itertools.chain(*keys))
temp.columns = ['subsector_visited','weights','card_id']

embed = pd.DataFrame(w2v.wv.syn0)
embed.columns = embed.columns.astype(str)
embed.columns = ['subsector_embedding_'+str(x) for x in embed.columns]
embed['subsector_id'] = w2v.wv.index2word

temp = temp.merge(embed, how='left', left_on='subsector_visited', right_on='subsector_id')
temp.drop(['subsector_id'],axis=1, inplace=True)
temp['weights'] = temp.weights.astype(int)

for i in range(30):
    temp['subsector_embedding_'+str(i)] = temp.weights * temp['subsector_embedding_'+str(i)]

temp = temp.groupby('card_id').apply(lambda x: x.iloc[:,3:].sum()/x.iloc[:,2].shape[0]).reset_index()

temp.to_pickle('new_subsector2vec.pkl')

####################################################################################################################
###         location
#####################################################################################
historical_transactions['location'] = [str(a)+'_'+str(b) for a,b in historical_transactions[['state_id','city_id']].values]
historical_transactions = historical_transactions.sort_values(by=['card_id','purchase_date'])
merchants_seq = historical_transactions.groupby('card_id')['location'].apply(lambda x: list(x))
merchants_seq = merchants_seq.to_frame().reset_index()

max_records = 50

path = get_tmpfile("hist_card_location_2Vec.kv")
w2v = Word2Vec(merchants_seq['location'], size=max_records, window=5, min_count=3, workers=8, sg=0)
w2v.wv.save(path)
# w2v = KeyedVectors.load("hist_card_merchants_2Vec.kv", mmap='r')

censor_until_months = 3
holder = historical_transactions.loc[abs(historical_transactions.month_lag)+1 <= censor_until_months][['card_id','location']]
keys = []
subs = []
for key, sub in holder.groupby('card_id'):
        v = sub.location.value_counts().reset_index().as_matrix().tolist()
        if len(v) > 0:
            subs.append(v)
            keys.append([key]*len(v))

del holder
temp = pd.DataFrame(np.concatenate(subs))
temp['card_id'] = list(itertools.chain(*keys))
temp.columns = ['location_visited','weights','card_id']

embed = pd.DataFrame(w2v.wv.syn0)
embed.columns = embed.columns.astype(str)
embed.columns = ['location_embedding_'+str(x) for x in embed.columns]
embed['location'] = w2v.wv.index2word

temp = temp.merge(embed, how='left', left_on='location_visited', right_on='location')
temp.drop(['location'],axis=1, inplace=True)
temp['weights'] = temp.weights.astype(int)

for i in range(50):
    temp['location_embedding_'+str(i)] = temp.weights * temp['location_embedding_'+str(i)]

temp = temp.groupby('card_id').apply(lambda x: x.iloc[:,3:].sum()/x.iloc[:,2].shape[0]).reset_index()

temp.to_pickle('hist_location2vec.pkl')

###########################################################################
new_merchant_transactions['location'] = [str(a)+'_'+str(b) for a,b in new_merchant_transactions[['state_id','city_id']].values]
new_merchant_transactions = new_merchant_transactions.sort_values(by=['card_id','purchase_date'])
new_merchant_transactions['location'] = new_merchant_transactions['location'].astype(str)
merchants_seq = new_merchant_transactions.groupby('card_id')['location'].apply(lambda x: list(x))
merchants_seq = merchants_seq.to_frame().reset_index()

max_records = 50

path = get_tmpfile("new_card_location_2Vec.kv")
w2v = Word2Vec(merchants_seq['location'], size=max_records, window=5, min_count=3, workers=8, sg=0)
w2v.wv.save(path)
# w2v = KeyedVectors.load("hist_card_merchants_2Vec.kv", mmap='r')

holder = new_merchant_transactions[['card_id','location']]
keys = []
subs = []
for key, sub in holder.groupby('card_id'):
        v = sub.location.value_counts().reset_index().as_matrix().tolist()
        if len(v) > 0:
            subs.append(v)
            keys.append([key]*len(v))

del holder
temp = pd.DataFrame(np.concatenate(subs))
temp['card_id'] = list(itertools.chain(*keys))
temp.columns = ['location_visited','weights','card_id']

embed = pd.DataFrame(w2v.wv.syn0)
embed.columns = embed.columns.astype(str)
embed.columns = ['location_embedding_'+str(x) for x in embed.columns]
embed['location'] = w2v.wv.index2word

temp = temp.merge(embed, how='left', left_on='location_visited', right_on='location')
temp.drop(['location'],axis=1, inplace=True)
temp['weights'] = temp.weights.astype(int)

for i in range(50):
    temp['location_embedding_'+str(i)] = temp.weights * temp['location_embedding_'+str(i)]

temp = temp.groupby('card_id').apply(lambda x: x.iloc[:,3:].sum()/x.iloc[:,2].shape[0]).reset_index()

temp.to_pickle('new_location2vec.pkl')
