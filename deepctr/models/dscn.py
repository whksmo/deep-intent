import tensorflow as tf

from ..inputs import input_from_feature_columns,build_input_features,combined_dnn_input
from ..layers.core import PredictionLayer, DNN
from ..layers.interaction import CrossNet
from tensorflow.python.keras.layers import Embedding, Input, Flatten
from base_model import *


def DSCN(embedding_size='auto', cross_num=2, dnn_hidden_units=(1000, 512,), l2_reg_embedding=1e-5,
        l2_reg_cross=1e-5, l2_reg_dnn=0, init_std=0.0001, seed=1024, dnn_dropout=0, dnn_use_bn=False,
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
        factor_embedding = init_embedding('factor_embedding', factor_num, factor_embedding_dim)
        input_factor_embedding = tf.nn.embedding_lookup(factor_embedding, input_factor)
        # input_factor_feature = tf.reduce_mean(input_factor_embedding, axis=1)
        input_factor_feature = get_MIL_att(input_factor_embedding)

	action_embedding = init_embedding('action_embedding', action_num, action_embedding_dim)
	service_embedding = init_embedding('service_embedding', service_num, service_embedding_dim)
	input_action_embedding = tf.nn.embedding_lookup(action_embedding, input_action)
	input_service_embedding = tf.nn.embedding_lookup(service_embedding, input_service)
	input_action_feature = get_lstm_output('action', input_action, input_action_embedding, action_embedding_dim)
	input_service_feature = get_lstm_output('service', input_service, input_service_embedding, service_embedding_dim)

    input_layer = input_factor_feature

    input_layer = tf.concat([input_layer, input_action_feature, input_service_feature], axis=1)

    cross_output = get_cross_output(input_layer)

    dnn_output = dense_layers(input_layer)
    concat_output = tf.concat([cross_output, dnn_output, input_action_feature], axis=1)

    concat_output = get_dropout(concat_output)

    logits_stu = tf.layers.dense(inputs=concat_output, units=num_class, activation=None, name='latent')
    pred_stu = PredictionLayer(num_class)(logits_stu)

    predict_result = pred_stu
    model = tf.keras.models.Model(inputs=input_list, outputs=predict_result)

    return model
