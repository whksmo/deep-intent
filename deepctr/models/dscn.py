import tensorflow as tf

from ..inputs import input_from_feature_columns,build_input_features,combined_dnn_input
from ..layers.core import PredictionLayer, DNN
from ..layers.tcn import TCN
from ..layers.interaction import *
from tensorflow.keras.layers import Activation, Embedding, Input, Flatten, Lambda, CuDNNLSTM, Dense, Concatenate
from base_model import *


def DSCN(cross_num=2, dnn_hidden_units=(1024, 512,), l2_reg_embedding=1e-5, l2_reg_cross=1e-5,
	l2_reg_dnn=0, dropout=0.5, dnn_use_bn=False, factor_num=19084, factor_embedding_dim=141,
	action_num=3826, action_embedding_dim=62, service_num=1751, service_embedding_dim=51, num_class=1):

    input_factor = Input(shape=(500,), dtype=tf.int32, name='factor')
    input_action = Input(shape=(100,), dtype=tf.int32, name='action')
    input_service = Input(shape=(20,), dtype=tf.int32, name='service')
    input_list = [input_factor, input_action, input_service]

    with tf.name_scope('embedding'):
        input_factor_embedding = Embedding(factor_num, factor_embedding_dim, mask_zero=True)(input_factor)
        #input_factor_feature = MaskMean()(input_factor_embedding)
        input_factor_feature = MILAttention()(input_factor_embedding)

	# input_action_feature = TCN(kernel_size=6, dilations=[1, 2, 4, 8, 16, 32, 64])(input_action_embedding)
	# input_service_feature = TCN(kernel_size=6, dilations=[1, 2, 4, 8, 16, 32, 64])(input_service_embedding)
	input_action_feature = SeqEmbedding(action_num, action_embedding_dim, type='lstm')(input_action)
	input_service_feature = SeqEmbedding(service_num, service_embedding_dim, type='lstm')(input_service)

    input_layer = Concatenate(axis=1)([input_factor_feature, input_action_feature, input_service_feature])

    cross_output = CrossNet(cross_num, l2_reg=l2_reg_cross)(input_layer)
    dense_layers = DNN(dnn_hidden_units, l2_reg=l2_reg_dnn, use_bn=dnn_use_bn)
    dnn_output = dense_layers(input_layer)

    concat_output = Concatenate(axis=1)([cross_output, dnn_output, input_action_feature])

    concat_output = tf.keras.layers.Dropout(dropout)(concat_output)

    logits = Dense(num_class)(concat_output)
    predict_result = Activation('softmax')(logits)
    model = tf.keras.models.Model(inputs=input_list, outputs=predict_result)

    return model
