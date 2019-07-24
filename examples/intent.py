import pandas as pd
import numpy as np
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

import sys
from os import path
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from tensorflow.python.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

from deepctr.models import DSCN
import deepctr.models as dm
from deepctr.inputs import SparseFeat, VarLenSparseFeat,get_fixlen_feature_names,get_varlen_feature_names


def run():
    model = DSCN(task=6723)
    model.compile("adam", "sparse_categorical_crossentropy", metrics=['accuracy'])
    history = model.fit(train_model_input, data[target].values, batch_size=256, epochs=20, verbose=2, validation_data=(test_model_input, test_data[target].values))

def process_varfeature(data, f, max_len):

    key2index = {}

    def split(x):
        key_ans = x.split(',')
	for key in key_ans:
	    if key not in key2index:
		key2index[key] = len(key2index) + 1
	return list(map(lambda x: key2index[x], key_ans))

    f_list = list(map(split, data[f].values))
    f_list = pad_sequences(f_list, maxlen=max_len, padding='post', )
    varlen_feature_columns = [VarLenSparseFeat(f, len(key2index) + 1, max_len, 'mean')]
    return f_list, varlen_feature_columns


def set_session():
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    tf.keras.backend.set_session(sess)


if __name__ == "__main__":
    data = pd.read_csv('../datasets/intent_train.csv', sep=';')

    target = ['label']

    factor_list, factor_columns = process_varfeature(data, 'factor', 500)
    action_list, action_columns = process_varfeature(data, 'action', 100)
    service_list, service_columns = process_varfeature(data, 'service', 20)
    train_model_input = [factor_list] + [action_list] + [service_list]

    # linear_feature_columns = factor_columns + action_columns + service_columns
    # dnn_feature_columns = factor_columns + action_columns + service_columns

    test_data = pd.read_csv('../datasets/intent_test.csv', sep=';')
    factor_list, factor_columns = process_varfeature(test_data, 'factor', 500)
    action_list, action_columns = process_varfeature(test_data, 'action', 100)
    service_list, service_columns = process_varfeature(test_data, 'service', 20)
    test_model_input = [factor_list] + [action_list] + [service_list]

    model = DSCN(task=6723)
    model.compile("adam", "sparse_categorical_crossentropy", metrics=['accuracy'])

    print('start training...')
    # hist = model.fit(train_model_input, data[target].values, batch_size=256, epochs=20, verbose=2, validation_data=(test_model_input, test_data[target].values))
    history = model.fit(train_model_input, data[target].values, batch_size=256, epochs=20, verbose=2, validation_split=0.035)
    model.save('./dscn.h5')

    # print('start test...')

    # pred_ans = model.predict(test_model_input, batch_size=256)
    # pred_res = model.evaluate(test_model_input, test_data[target].values, batch_size=256, verbose=2)
    # print(pred_res)
    # print("test LogLoss", round(log_loss(test_data[target].values, pred_ans), 4))
    # print("test AUC", round(roc_auc_score(test_data[target].values, pred_ans), 4))
