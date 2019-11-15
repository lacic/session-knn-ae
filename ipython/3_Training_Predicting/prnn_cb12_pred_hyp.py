#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.layers import Input, Dense, concatenate
from keras.layers.recurrent import GRU
from keras.utils import plot_model
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint
import keras
import pandas as pd
import numpy as np
import keras.backend as K
from keras.utils import to_categorical
from keras.losses import categorical_crossentropy
from multiprocessing import Pool, cpu_count
import pickle
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'


# In[2]:


dataset = "cb12/"
path =  "../../data/"
interim_path = path + dataset + "interim/"
processed_path = path + dataset + "processed/"
model_path = "models/"
model_path_valid = "models/valid/"


# In[3]:


def TOP1(y_true, y_pred):
    y1 = y_pred * y_true
    y2 = K.sum(y1, axis=1)[:, np.newaxis]
    y3 = y_true - y1
    return (K.sum(K.sigmoid(y_pred - y2)) + y3 * y3) / tf.cast(tf.shape(y_true)[0], tf.float32)

loss = TOP1

def create_prnn_model(left_input_size, right_input_size, batch_size = 512, hidden_units = 100, o_activation='softmax', lr = 0.001):   
    emb_size = 50
    size = emb_size

    # left input - item vector
    input_left = Input(batch_shape=(batch_size, 1, left_input_size), name='input_left')
    gru_left, gru_left_states = GRU(hidden_units, stateful=True, return_state=True, name='gru_left')(input_left)

    # right input - feature vector
    input_right = Input(batch_shape=(batch_size, 1, right_input_size), name='input_right')
    gru_right, gru_right_states = GRU(hidden_units, stateful=True, return_state=True, name='gru_right')(input_right)
    
    # merging both layers and creating the model
    merged = concatenate([gru_left, gru_right])
    #change softmax per another activation funciton?
    output = Dense(left_input_size, activation=o_activation, name='output')(merged)
    model = Model(inputs=[input_left, input_right], outputs=output, name='gru4rec')
    
    encoder = Model(inputs=[input_left, input_right], outputs=merged)

    # define model's optimizer
    #optimizer = optim.Optimizer(optimizer=self.optimizer, lr=self.lr)
    #opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    opt = keras.optimizers.Adagrad(lr=lr)
    
    # define model's loss function --> implement here the top1 loss function
#     loss_function = loss.LossFunction(loss_type=self.loss_function)
    #model.compile(loss=loss_function, optimizer=opt, metrics=['accuracy'])
    
    model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])
    filepath = model_path_valid + 'prnn_cb12_checkpoint.h5'
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=2, save_best_only=True, mode='min')
    callbacks_list = []
    model.summary()
    #plot_model(model, show_shapes=True, to_file='rnn-structure.png')
    return model, encoder

def get_states(model):
    #return the actual states of the layers
    return [K.get_value(s) for s,_ in model.state_updates]


def freeze_layer(model, layer_name, lr):
    if layer_name == 'gru_left':
        # gru left layer will not be trained this mini batch
        model.get_layer(layer_name).trainable = False
        # but gru right will
        model.get_layer('gru_right').trainable = True
    elif layer_name == 'gru_right':
        # gru right layer will not be trained this mini batch
        model.get_layer(layer_name).trainable = False
        # but gru left will
        model.get_layer('gru_left').trainable = True
    else:
        raise NotImplementedError
        
    # opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    opt = keras.optimizers.Adagrad(lr=lr)
    model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])
    return model


# In[4]:


train_path = '../../data/' + dataset + 'processed/valid_train_14d.csv'
train = pd.read_csv(train_path, sep='\t')[['session_id', 'item_id', 'created_at']]

interactions = pd.read_csv('../../data/' + dataset + 'interim/interactions.csv', header=0, sep='\t')
items = pd.read_csv('../../data/' + dataset + 'interim/items.csv', header=0, sep='\t')
view_fields = ["item_id", "state", "ReqTopic", "DescTopic", "TitTopic"]
common_items = items.merge(interactions, on=['item_id'])[view_fields].drop_duplicates()
print(common_items.head(3))

## to test - delete after
#train = train.head(6)
#print(train.session_id)
#common_items = common_items.merge(train, on=['item_id'])[view_fields].drop_duplicates()

## delete 


item_count = len(train['item_id'].unique())
session_count = len(train['created_at'].unique())
print(len(common_items))
# common_items.head(10)


# In[5]:


# Studo items need to be converted to dummies

common = common_items

common["item_id"] = common["item_id"].astype('str')
common["DescTopic"] = common["DescTopic"].astype('str')
common["TitTopic"] = common["TitTopic"].astype('str')
common["ReqTopic"] = common["ReqTopic"].astype('str')

df2 = pd.DataFrame(index=common.index)
s1 = pd.get_dummies(common["state"].fillna("").str.split(",").apply(pd.Series).stack(), prefix="state").sum(level=0)
df2 = pd.concat([df2, s1], axis=1)
s1 = pd.get_dummies(common["ReqTopic"].fillna("").str.split(",").apply(pd.Series).stack(), prefix="ReqTopic").sum(level=0)
df2 = pd.concat([df2, s1], axis=1)
df2 = df2.drop(["state_", "ReqTopic_"], axis=1, errors="ignore")

s1 = pd.get_dummies(common["DescTopic"].fillna("").str.split(",").apply(pd.Series).stack(), prefix="DescTopic").sum(level=0)
df2 = pd.concat([df2, s1], axis=1)

s1 = pd.get_dummies(common["TitTopic"].fillna("").str.split(",").apply(pd.Series).stack(), prefix="TitTopic").sum(level=0)
df2 = pd.concat([df2, s1], axis=1)

df2 = df2.drop(["DescTopic_", "TitTopic_"], axis=1, errors="ignore")


common = common.drop(["state", "ReqTopic", "DescTopic", "TitTopic"], axis=1)
df2 = pd.concat([common, df2], axis=1)
print(df2.head(2))

one_hot = df2
print(one_hot.shape)
# number of content features per item
feature_size = one_hot.shape[1] - 1

item_encodings = {}
for index, row in one_hot.iterrows():
    item_id = str(row["item_id"])
    item_encodings[item_id] = row.values[1:]

print(len(item_encodings))


# In[6]:


print(feature_size)
print(item_count)

print(len(common.item_id.unique()))
for value in common.item_id.unique():
    print(value)
    print(item_encodings[value])
    print(type(item_encodings[value]))
    break
    
empty_feature_vec = np.zeros(feature_size, dtype=int)


# In[9]:


class SessionDataset:
    """Credit to yhs-968/pyGRU4REC."""    
    def __init__(self, data, sep='\t', session_key='session_id', item_key='item_id', time_key='created_at', n_samples=-1, itemmap=None, time_sort=False):
        """
        Args:
            path: path of the csv file
            sep: separator for the csv
            session_key, item_key, time_key: name of the fields corresponding to the sessions, items, time
            n_samples: the number of samples to use. If -1, use the whole dataset.
            itemmap: mapping between item IDs and item indices
            time_sort: whether to sort the sessions by time or not
        """
        self.df = data
        self.session_key = session_key
        self.item_key = item_key
        self.time_key = time_key
        self.time_sort = time_sort
        self.add_item_indices(itemmap=itemmap)
        self.df.sort_values([session_key, time_key], inplace=True)

        # Sort the df by time, and then by session ID. That is, df is sorted by session ID and
        # clicks within a session are next to each other, where the clicks within a session are time-ordered.

        self.click_offsets = self.get_click_offsets() 
        #array of the positions where there is a change of session. 
        #len = len(session_idx_arr) + 1
        
        self.session_idx_arr = self.order_session_idx() 
        #array of sessions [0 1 2 3 4 .... n-1]
        
    def get_click_offsets(self):
        """
        Return the offsets of the beginning clicks of each session IDs,
        where the offset is calculated against the first click of the first session ID.
        """
        offsets = np.zeros(self.df[self.session_key].nunique() + 1, dtype=np.int32)
        # group & sort the df by session_key and get the offset values
        offsets[1:] = self.df.groupby(self.session_key).size().cumsum()
        return offsets

    def order_session_idx(self):
        """ Order the session indices """
        if self.time_sort:
            # starting time for each sessions, sorted by session IDs
            sessions_start_time = self.df.groupby(self.session_key)[self.time_key].min().values
            # order the session indices by session starting times
            session_idx_arr = np.argsort(sessions_start_time)
        else:
            session_idx_arr = np.arange(self.df[self.session_key].nunique())
        return session_idx_arr
    
    def add_item_indices(self, itemmap=None):
        """ 
        Add item index column named "item_idx" to the df
        Args:
            itemmap (pd.DataFrame): mapping between the item Ids and indices
        """
        if itemmap is None:
            item_ids = self.df[self.item_key].unique()  # unique item ids
            item2idx = pd.Series(data=np.arange(len(item_ids)),
                                 index=item_ids)
            itemmap = pd.DataFrame({self.item_key:item_ids,
                                   'item_idx':item2idx[item_ids].values})
        
        self.itemmap = itemmap
        self.df = pd.merge(self.df, self.itemmap, on=self.item_key, how='inner')
        
    @property    
    def items(self):
        return self.itemmap.item_id.unique()


# In[10]:


class SessionDataLoader:
    """Credit to yhs-968/pyGRU4REC."""    
    def __init__(self, dataset, batch_size):
        """
        A class for creating session-parallel mini-batches.
        Args:
            dataset (SessionDataset): the session dataset to generate the batches from
            batch_size (int): size of the batch
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.done_sessions_counter = 0
        
    def __iter__(self):
        """ Returns the iterator for producing session-parallel training mini-batches.
        Yields:
            input (B,):  Item indices that will be encoded as one-hot vectors later.
            target (B,): a Variable that stores the target item indices
            masks: Numpy array indicating the positions of the sessions to be terminated
        """

        df = self.dataset.df
        
        session_key='session_id'
        item_key='item_id'
        time_key='created_at'
        self.n_items = df[item_key].nunique()
        click_offsets = self.dataset.click_offsets
        #print(click_offsets)
        session_idx_arr = self.dataset.session_idx_arr
        #print(session_idx_arr)
        
        iters = np.arange(self.batch_size)
        #iters = np.arange(1)

        maxiter = iters.max()
                
        start = click_offsets[session_idx_arr[iters]]
        end = click_offsets[session_idx_arr[iters] + 1]
        #print(start)
        #print(end)
        mask = [] # indicator for the sessions to be terminated
        finished = False        

        while not finished:
            #minimum lenght of all the sessions
            minlen = (end - start).min()
            # Item indices (for embedding) for clicks where the first sessions start
            idx_target = df.item_idx.values[start]
            for i in range(minlen - 1):
                # Build inputs & targets
                idx_input = idx_target
                idx_target = df.item_idx.values[start + i + 1]
                inp = idx_input
                target = idx_target
                yield inp, target, mask
                
            # click indices where a particular session meets second-to-last element
            start = start + (minlen - 1)
            # see if how many sessions should terminate
            mask = np.arange(len(iters))[(end - start) <= 1]
            self.done_sessions_counter = len(mask)
            for idx in mask:
                maxiter += 1
                if maxiter >= len(click_offsets) - 1:
                    finished = True
                    break
                # update the next starting/ending point
                iters[idx] = maxiter
                start[idx] = click_offsets[session_idx_arr[maxiter]]
                end[idx] = click_offsets[session_idx_arr[maxiter] + 1]
            


# # Hyperparameter definitions

# In[12]:


batch_size = 512
acts = ['softmax', 'tanh']
l_sizes = [100, 1000]
lrs = [0.001, 0.01]


# # Predict for hyperparameters

# In[ ]:


import keras.losses
keras.losses.TOP1 = TOP1
pd.set_option('display.max_colwidth', -1) 

train_dataset = SessionDataset(train)
loader = SessionDataLoader(train_dataset, batch_size=batch_size)
    
def predict_function(sid, test_session, pr, item_idx_map, idx_item_map, cut_off=20, 
                     session_key='session_id', item_key='item_id', time_key='created_at'):
    test_session.sort_values([time_key], inplace=True)
    # get first and only session_id (as we grouped it before calling this method)
    session_id = test_session[session_key].unique()[0]

    log_columns = ["session_id", "input_items", "input_count", "position", "remaining_items", "remaining_count", "predictions"]
    log_df = pd.DataFrame(columns = log_columns)

    session_length = len(test_session)
    il = a = np.zeros((batch_size, 1, len(item_idx_map)))
    ir = a = np.zeros((batch_size, 1, 115))
    
    for i in range(session_length -1):
        # use current item as reference point (rest is for testing)
        current_item_id = test_session[item_key].values[i]

        item_vec = np.zeros(len(item_idx_map), dtype=int)
        item_idx = item_idx_map[current_item_id]
        item_vec[item_idx] = 1
        # set vector in batch input
        il[i, 0] = item_vec
        
        #item_features = item_encodings[current_item_id]
        
        # use empty feature vec if missing
        item_features = empty_feature_vec
        if current_item_id in item_encodings.keys():
            item_features = item_encodings[result]
                    
        #item_features = item_features.reshape(1,1, len(item_features))
        ir[i, 0] = item_features
        
    # do batch prediction
    pred = model.predict([il, ir], batch_size=batch_size)
    
    # for every subsession prediction
    for i in range(session_length-1):
        preds = pred[i]
        topn_idx_preds = preds.argsort()[-cut_off:][::-1]
        
        predictions = []
        # for every recommended item index
        for item_idx in topn_idx_preds:
            pred_item = idx_item_map[item_idx]
            predictions.append(pred_item)
            
        current_input_set = test_session[item_key].values[:i+1]
        remaining_test_set = test_session[item_key].values[i+1:]
        
        position = "MID"
        if i == 0:
            position = "FIRST"
        if len(remaining_test_set) == 1:
            position = "LAST"
        
        log_df = log_df.append({
            "session_id": sid,
            "input_items":  ','.join(map(str, current_input_set)),
            "input_count":  len(current_input_set),
            "position": position,
            "remaining_items":  ','.join(map(str, remaining_test_set)),
            "remaining_count":  len(remaining_test_set),
            "predictions": ','.join(map(str, predictions))
        }, ignore_index=True) 
            
    
    log_df['input_count'] = log_df['input_count'].astype(int)
    log_df['remaining_count'] = log_df['remaining_count'].astype(int)
    
    return log_df

test_path = '../../data/' + dataset + 'processed/valid_test_14d.csv'
test = pd.read_csv(test_path, sep='\t')[['session_id', 'item_id', 'created_at']]
test_dataset = SessionDataset(test)
test_generator = SessionDataLoader(test_dataset, batch_size=batch_size)

session_groups = test.groupby("session_id")
mapitem = loader.dataset.itemmap

item_idx_map = {}
idx_item_map = {}
for index, row in mapitem.iterrows():
    item_id = row["item_id"]
    item_idx = row["item_idx"]
    item_idx_map[item_id] = item_idx
    idx_item_map[item_idx] = item_id

    
predict_path = "../../data/cb12/interim/predict/hyperparam/"

for act in acts:
    for ls in l_sizes:
        for lr in lrs:
            model_name = "cb12_prnn_a_" + act + "_ls_" + str(ls) + "_lr_" + str(lr) + ".model"
            model = pickle.load(open(model_path_valid + model_name, 'rb'))
            
            print("Loaded: " + model_name)
            res_list = []
            # predict
            report_freq = len(session_groups) // 5 
            count = 0
            for sid, session in session_groups:
                pred_df = predict_function(sid, session, model, item_idx_map, idx_item_map)
                res_list.append(pred_df)
                # reset states
                model.get_layer('gru_left').reset_states()
                model.get_layer('gru_right').reset_states()
                # print progress
                count += 1
                if count % report_freq == 0:
                    print("Predicted for " + str(count) + " sessions. " + str(len(session_groups) - count) + " sessions to go." )
            # concat results
            res = pd.concat(res_list)
            res = res.reindex(columns = ["session_id", "input_items", "input_count", "position", "remaining_items", "remaining_count", "predictions"])

            store_name = model_name.replace("cb12_", "").replace(".model", "")
            res.to_csv(predict_path + "test_14d_" + store_name + ".csv", sep='\t')
            
            print("Stored predictions: " + predict_path + "test_14d_" + store_name + ".csv")
