import tensorflow as tf

from ..inputs import input_from_feature_columns,build_input_features,combined_dnn_input
from ..layers.core import PredictionLayer, DNN
from ..layers.tcn import TCN
from ..layers.interaction import CrossNet, SeqEmbedding
from tensorflow.python.keras.layers import Activation, Embedding, Input, Flatten, Lambda, CuDNNLSTM, LSTM
from base_model import *


def DSCN(embedding_size='auto', cross_num=2, dnn_hidden_units=(1000, 512,), l2_reg_embedding=1e-5,
        l2_reg_cross=1e-5, l2_reg_dnn=0, init_std=0.0001, seed=1024, dropout=0.5, dnn_use_bn=False,
        dnn_activation='relu', task='binary'):

    num_class = 1 if task == 'binary' else task
    input_factor = Input(shape=(500,), dtype=tf.int32, name='factor')
    input_action = Input(shape=(100,), dtype=tf.int32, name='action')
    input_service = Input(shape=(20,), dtype=tf.int32, name='service')
    input_list = [input_factor, input_action, input_service]

    factor_num = 19084
    factor_embedding_dim = 141

    action_num = 3826
    action_embedding_dim = 62

    service_num = 1751
    service_embedding_dim = 51

    dense_layers = tf.keras.Sequential([tf.keras.layers.Dense(dim, activation='relu') for dim in dnn_hidden_units])

    with tf.name_scope('embedding'):
        # factor_embedding = init_embedding('factor_embedding', factor_num, factor_embedding_dim)
        input_factor_embedding = Embedding(factor_num, factor_embedding_dim, mask_zero=True)(input_factor)
        # input_factor_embedding = tf.nn.embedding_lookup(factor_embedding, input_factor)
        input_factor_feature = Lambda(lambda x: tf.reduce_mean(x, axis=1))(input_factor_embedding)
        # input_factor_feature = Lambda(get_MIL_att)(input_factor_embedding)

    input_action_embedding = Embedding(action_num, action_embedding_dim, mask_zero=True)(input_action)
    input_service_embedding = Embedding(service_num, service_embedding_dim, mask_zero=True)(input_service)
    # input_action_feature = TCN(nb_filters=64, kernel_size=6, dilations=[1, 2, 4, 8, 16, 32, 64])(input_action_embedding)
    # input_action_feature = LSTM(action_embedding_dim, unroll=True)(input_action_embedding)
    input_action_feature = Lambda(lambda x: tf.reduce_mean(x, axis=1))(input_action_embedding)
    # input_service_feature = TCN(nb_filters=64, kernel_size=6, dilations=[1, 2, 4, 8, 16, 32, 64])(input_service_embedding)
    # input_service_feature = LSTM(service_embedding_dim, unroll=True)(input_service_embedding)
    input_service_feature = Lambda(lambda x: tf.reduce_mean(x, axis=1))(input_service_embedding)
    print('hahaha')
    # input_action_feature = SeqEmbedding(action_num, action_embedding_dim, type='mean')(input_action)
    # input_service_feature = SeqEmbedding(service_num, service_embedding_dim, type='mean')(input_service)

    input_layer = tf.keras.layers.concatenate([input_factor_feature, input_action_feature, input_service_feature], axis=1)

    # cross_output = Lamda(get_cross_output)(input_layer)
    cross_output = CrossNet(cross_num, l2_reg=l2_reg_cross)(input_layer)

    dnn_output = dense_layers(input_layer)
    concat_output = tf.keras.layers.concatenate([cross_output, dnn_output, input_action_feature], axis=1)

    concat_output = tf.keras.layers.Dropout(dropout)(concat_output)

    logits_stu = Dense(num_class)(concat_output)
    # pred_stu = PredictionLayer(num_class)(logits_stu)
    predict_result = Activation('softmax')(logits_stu)
    # predict_result = pred_stu
    model = tf.keras.models.Model(inputs=input_list, outputs=predict_result)

    return model
