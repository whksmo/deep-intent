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

from deepctr.models import DeepFM
import deepctr.models as dm
from deepctr.inputs import SparseFeat, VarLenSparseFeat,get_fixlen_feature_names,get_varlen_feature_names


def split(x):
    key_ans = x.split('|')
    for key in key_ans:
        if key not in key2index:
            # Notice : input value 0 is a special "padding",so we do not use 0 to encode valid feature for sequence input
            key2index[key] = len(key2index) + 1
    return list(map(lambda x: key2index[x], key_ans))


def process_varfeature(data, f):
    key2index = {}
    factor_list = list(map(split, data[f].values))
    factor_length = np.array(list(map(len, factor_list)))
    max_len = max(factor_length)
    factor_list = pad_sequences(factor_list, maxlen=max_len, padding='post', )
    varlen_feature_columns = [VarLenSparseFeat(f, len(key2index) + 1, max_len, 'mean')]
    return factor_list, varlen_feature_columns

if __name__ == "__main__":
    data = pd.read_csv('../datasets/intent_train.csv', sep=';')
    test_data = pd.read_csv('../datasets/intent_test.csv', sep=';')

    target = ['label']

    factor_list, factor_columns = process_varfeature(data, 'factor')
    action_list, action_columns = process_varfeature(data, 'action')
    service_list, service_columns = process_varfeature(data, 'service')

    linear_feature_columns = factor_columns + action_columns + service_columns
    dnn_feature_columns = factor_columns + action_columns + service_columns
    varlen_feature_names = get_varlen_feature_names(linear_feature_columns + dnn_feature_columns)

    train_model_input = [factor_list] + [action_list] + [service_list]
    model = dm.dcn(dnn_feature_columns, task=6723)
    model.compile("adam", "categorical_crossentropy", metrics=['accuracy'])

    history = model.fit(train_model_input, data[target].values,
                        batch_size=256, epochs=20, verbose=2, validation_split=0.1)

    factor_list, factor_columns = process_varfeature(test_data, 'factor')
    action_list, action_columns = process_varfeature(test_data, 'action')
    service_list, service_columns = process_varfeature(test_data, 'service')
    test_model_input = [factor_list] + [action_list] + [service_list]

    pred_ans = model.predict(test_model_input, batch_size=256)
    print("test LogLoss", round(log_loss(test_data[target].values, pred_ans), 4))
    # print("test AUC", round(roc_auc_score(test_data[target].values, pred_ans), 4))
