# -*- coding:utf-8 -*-
import tensorflow as tf
from tensorflow.python.training.training_util import get_or_create_global_step
from tensorflow.python.keras.initializers import (Zeros, glorot_normal, glorot_uniform)
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.layers import utils
from tensorflow.python.keras.layers import Layer, Input, Embedding, LSTM, Dense
import numpy as np


class MyBaseModel():
    @property
    def name(self):
        return 'MyBaseModel'

    def __init__(self, config):
        super(MyBaseModel, self).__init__(config)
        x = config.x
        self._factor = x[0].feature_name
        if len(x) > 2:
            self._action = x[1].feature_name
            self._service = x[2].feature_name
        if len(x) > 3:
            self._other = x[3].feature_name
        self._label_name = config.y[0].feature_name

        self.config = config
        self.update_op = None
        self.output_class_num = config.output_class_num
        self._batch_size = config.batch_size

        self._display_auc = config.get('display_auc', False)

        self._only_cross = config.get('only_cross', False)
        self._only_seq = config.get('only_seq', False)
        self._only_factor = config.get('only_factor', False)
        self._no_action = config.get('no_action', False)
        self._concat_autoint = config.get('concat_autoint', False)
        self._concat_dnn = config.get('concat_dnn', True)
        self._concat_retrieval = config.get('concat_retrieval', False)
        self._rank_only_topk = config.get('rank_only_topk', False)

        self._concat_att = config.get('concat_att', True)

        self._dropout_for_dnn = config.get('dropout_for_dnn', False)
        self._dropout_for_att = config.get('dropout_for_att', False)
        self._dropout_for_concat = config.get('dropout_for_concat', False)
        self._use_label_norm = config.get('use_label_norm', False)
        self._use_candidate_sample = config.get('use_candidate_sample', False)
        self._use_updated = config.get('use_updated', False)
        self._use_label_weight = config.get('use_label_weight', False)
        self._use_label_ratio = config.get('use_label_ratio', False)
        self._use_focal_loss = config.get('use_focal_loss', False)
        self._trim_minor = config.get('trim_minor', False)
        self._use_noise_regulation = config.get('use_noise_regulation', False)
        self._use_random_drop = config.get('use_random_drop', False)
        self._reg_all = config.get('reg_all', False)
        self._reg_cross = config.get('reg_cross', False)

        self._use_seq = config.get('use_seq', True)
        self._use_att_for_seq = config.get('use_att_for_seq', False) and self._use_seq
        self._use_transformer_for_seq = config.get('use_transformer_for_seq', False) and self._use_seq
        self._use_lstm_for_seq = config.get('use_lstm_for_seq', True) and self._use_seq
        self._use_seq_pooling = config.get('use_seq_pooling', False) and self._use_seq
        self._use_tcn_for_seq = config.get('use_tcn_for_seq', False) and self._use_seq
        self._use_lstm_att = config.get('use_lstm_att_for_seq', False) and self._use_seq
        self._use_struct_att = config.get('use_struct_att', False) and self._use_seq

        self._use_autoint_for_seq = config.get('use_autoint_for_seq', False)
        self._use_factor_pooling = config.get('use_factor_pooling', False)
        self._use_MIL_att = config.get('use_MIL_att', False)
        self._use_ten_encoding = config.get('use_ten_encoding', False)
        self._use_input_pooling = config.get('use_input_pooling', False)
        self._use_set_transformer = config.get('use_set_transformer', False)
        self._use_random_sample_input = config.get('use_random_sample_input', False)

        self._is_binary = False

def _conv_output_shape(input_shape, kernel_size, filters_list):
    # channels_last
    space = input_shape[1:-1]
    new_space = []
    for i in range(len(space)):
        new_dim = utils.conv_output_length(space[i], kernel_size[i], padding='same', stride=1, dilation=1)
        new_space.append(new_dim)
    return ([input_shape[0]] + new_space + [filters_list])

def _pooling_output_shape(input_shape, pool_size):
    # channels_last

    rows = input_shape[1]
    cols = input_shape[2]
    rows = utils.conv_output_length(rows, pool_size[0], 'valid', pool_size[0])
    cols = utils.conv_output_length(cols, pool_size[1], 'valid', pool_size[1])
    return [input_shape[0], rows, cols, input_shape[3]]


def unstack(input_tensor):
    input_ = tf.expand_dims(input_tensor, axis=2)
    return tf.unstack(input_, input_.shape[1], 1)

def get_fgcnn_output(inputs, filters=(14, 16,), kernel_width=(7, 7,), new_maps=(3, 3,), pooling_width=(2, 2)):
    print('get fgcnn output!!')
    input_shape = inputs.get_shape().as_list()
    filters_list = filters
    kernel_width_list = kernel_width
    map_list = new_maps
    pooling_list = pooling_width
    if len(input_shape) != 3:
        # raise ValueError("Unexpected inputs dimensions %d, expect to be 3 dimensions" % (len(input_shape)))
        inputs = tf.expand_dims(inputs, axis=-1)
        input_shape = inputs.get_shape().as_list()
    # ------ build -------
    conv_layers = []
    pooling_layers = []
    dense_layers = []
    pooling_shape = input_shape + [1, ]
    embedding_size = input_shape[-1]
    for i in range(1, len(filters_list) + 1):
        filters = filters_list[i - 1]
        width = kernel_width_list[i - 1]
        new_filters = map_list[i - 1]
        pooling_width = pooling_list[i - 1]
        conv_output_shape = _conv_output_shape(pooling_shape, (width, 1))
        pooling_shape = _pooling_output_shape(conv_output_shape, (pooling_width, 1))
        conv_layers.append(tf.keras.layers.Conv2D(filters=filters, kernel_size=(width, 1), strides=(1, 1), padding='same', activation='tanh', use_bias=True, ))
        pooling_layers.append(tf.keras.layers.MaxPooling2D(pool_size=(pooling_width, 1)))
        dense_layers.append(tf.keras.layers.Dense(pooling_shape[1] * embedding_size * new_filters, activation='tanh', use_bias=True))

    flatten = tf.keras.layers.Flatten()

    # ----- call -------
    embedding_size = input_shape[-1]
    pooling_result = tf.expand_dims(inputs, axis=3)

    new_feature_list = []

    for i in range(1, len(filters_list) + 1):
        new_filters = map_list[i - 1]

        conv_result = conv_layers[i - 1](pooling_result)

        pooling_result = pooling_layers[i - 1](conv_result)

        flatten_result = flatten(pooling_result)

        new_result = dense_layers[i - 1](flatten_result)

        new_feature_list.append(tf.reshape(new_result, (-1, pooling_result.shape[1].value * new_filters, embedding_size)))

    new_features = tf.concat(new_feature_list, axis=1)
    return new_features

# layer used in xDeepFM
def get_cin_output(inputs, layer_size=[128, 128], split_half=True):
    while len(inputs.shape) < 3:
        inputs = tf.expand_dims(inputs, -1)
    input_shape = inputs.get_shape().as_list()
    # --- weight initialization ---
    field_nums = [input_shape[1]]
    filters = []
    bias = []
    for i, size in enumerate(layer_size):
        filters.append(tf.get_variable(name='filter' + str(i), shape=[1, field_nums[-1] * field_nums[0], size], dtype=tf.float32, initializer=glorot_uniform(seed=i), ))

        bias.append(tf.get_variable(name='bias' + str(i), shape=[size], dtype=tf.float32,
                                         initializer=tf.initializers.zeros()))
        if split_half:
            if i != len(layer_size) - 1 and size % 2 > 0:
                raise ValueError(
                    "layer_size must be even number except for the last layer when split_half=True")

            field_nums.append(size // 2)
        else:
            field_nums.append(size)

    dim = input_shape[-1]
    hidden_nn_layers = [inputs]
    final_result = []
    split_tensor0 = tf.split(hidden_nn_layers[0], dim * [1], 2)
    for idx, size in enumerate(layer_size):
        split_tensor = tf.split(hidden_nn_layers[-1], dim * [1], 2)

        dot_result_m = tf.matmul(split_tensor0, split_tensor, transpose_b=True)

        dot_result_o = tf.reshape(dot_result_m, shape=[dim, -1, field_nums[0] * field_nums[idx]])

        dot_result = tf.transpose(dot_result_o, perm=[1, 0, 2])

        curr_out = tf.nn.conv1d(
            dot_result, filters=filters[idx], stride=1, padding='VALID')

        curr_out = tf.nn.bias_add(curr_out, bias[idx])

        curr_out = tf.nn.relu(curr_out)

        curr_out = tf.transpose(curr_out, perm=[0, 2, 1])

        if split_half:
            if idx != len(layer_size) - 1:
                next_hidden, direct_connect = tf.split(curr_out, 2 * [size // 2], 1)
            else:
                direct_connect = curr_out
                next_hidden = 0
        else:
            direct_connect = curr_out
            next_hidden = curr_out

        final_result.append(direct_connect)
        hidden_nn_layers.append(next_hidden)

    result = tf.concat(final_result, axis=1)
    result = tf.reduce_sum(result, -1, keep_dims=False)

    return result

def get_inner_product(inputs, reduce_sum=True):
    # input_shape = [ipt.get_shape().as_list() for ipt in inputs]

    embed_list = inputs
    row = []
    col = []
    num_inputs = len(embed_list)

    for i in range(num_inputs - 1):
        for j in range(i + 1, num_inputs):
            row.append(i)
            col.append(j)
    p = tf.concat([embed_list[idx] for idx in row], axis=1)  # batch num_pairs k
    q = tf.concat([embed_list[idx] for idx in col], axis=1)

    inner_product = p * q
    if reduce_sum:
        inner_product = tf.reduce_sum(inner_product, axis=2, keep_dims=True)
    return inner_product

def get_struct_att(cond_input, query_seq, name='jp', head_num=8, query_emb_dim=128, mask=None):
    seq_len = query_seq.get_shape()[1].value  # N

    embedding_seq = tf.tanh(Dense(query_emb_dim)(query_seq))  # b x N x q_emb
    A = Dense(head_num)(embedding_seq)  # b x N x r
    A = tf.transpose(A, [0, 2, 1])  # b x r x N
    if mask is not None:
        mask = tf.reshape(tf.squeeze(mask), [-1, seq_len])  # b x N
        A -= 1e9 * mask[:, tf.newaxis, :]
    A = tf.nn.softmax(A)
    print('shape of A!!:{0}'.format(A.shape.as_list()))

    conds = Dense(seq_len)(cond_input)  # b x N
    A *= conds[:, tf.newaxis, :]

    res = tf.matmul(A, query_seq)  # b x r x u
    return tf.layers.flatten(res)

def get_attention_seq(keys, querys, vals=None, use_ori=False, head_num=2, att_embedding_size=8, name='att_seq', mask=None, return_w=False):
    if len(querys.shape) < 3:
        querys = tf.expand_dims(querys, 1)  # b x 1 x d

    ori_inputs = keys
    query_dim = querys.get_shape()[-1].value  # d
    key_dim = keys.get_shape()[-1].value
    if vals is None:
        val_dim = key_dim
        vals = keys
    w_dim = att_embedding_size * head_num

    with tf.variable_scope(name):
        w_query = tf.get_variable('w_query', shape=[query_dim, w_dim], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=1.0 / w_dim), trainable=True)
        w_key = tf.get_variable('w_key', shape=[key_dim, w_dim], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=1.0 / w_dim), trainable=True)
        w_val = tf.get_variable('w_val', shape=[val_dim, w_dim], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=1.0 / w_dim), trainable=True)

    querys = tf.tensordot(querys, w_query, axes=(-1, 0))
    querys = tf.stack(tf.split(querys, head_num, axis=-1))  # h x b x M' x d

    keys = tf.tensordot(keys, w_key, axes=(-1, 0))  # b x M x d
    keys = tf.stack(tf.split(keys, head_num, axis=-1))  # h x b x M x d

    vals = tf.tensordot(vals, w_val, axes=(-1, 0))  # b x M x d
    vals = tf.stack(tf.split(vals, head_num, axis=-1))  # h x b x M x d

    inner_product = tf.matmul(querys, keys, transpose_b=True)  # h x b x M' x M
    if mask is not None:
        inner_product -= 1e9 * mask
    normalized_att_scores = tf.nn.softmax(inner_product)
    # normalized_att_scores = tf.transpose(normalized_att_scores, [1, 2, 3, 0])  # b x M' x M x h
    # normalized_att_scores = tf.stack(tf.split(normalized_att_scores, head_num), axis=-1)
    # normalized_att_scores = tf.squeeze(normalized_att_scores, axis=0)

    if use_ori:
        att_val = tf.matmul(normalized_att_scores, tf.tile(ori_inputs[tf.newaxis, ...], (head_num, 1, 1, 1)))  # h x b x M' x ori_dim
    else:
        att_val = tf.matmul(normalized_att_scores, vals)  # h x b x M' x d
    att_val = tf.transpose(att_val, [1, 2, 3, 0])  # b x M' x d x h
    # att_val_shape = att_val.get_shape().as_list()
    # print(att_val_shape)
    # att_val = tf.reshape(att_val, [-1, att_val_shape[1] * att_val_shape[2]])
    print(keys.get_shape().as_list())
    res_shape = att_val.shape.as_list()
    if res_shape[1] == 1:
        att_val = tf.layers.flatten(att_val)  # b x (d * h)
    else:
        att_val = tf.reshape(att_val, [-1, res_shape[1], res_shape[2] * res_shape[3]])  # b x M' x (d * h)
    if return_w:
        return att_val, tf.squeeze(normalized_att_scores, axis=0)
    else:
        return att_val

def get_autoint_output(inputs, att_embedding_size=8, head_num=2, use_res=True, layer_num=3, concat=True, reduce_mean=True, use_dropout=False, name='autoint', reuse=False):
    w_dim = att_embedding_size * head_num
    for l in range(layer_num):
        while len(inputs.shape) < 3:
            inputs = tf.expand_dims(inputs, axis=-1)
        layer_input = inputs
        embed_dim = layer_input.get_shape().as_list()[-1]

        with tf.variable_scope('{0}_attention_{1}'.format(name, l), reuse=reuse):
            w_query = tf.get_variable('w_query', shape=[embed_dim, w_dim], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=1.0 / w_dim), trainable=True)
            w_key = tf.get_variable('w_key', shape=[embed_dim, w_dim], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=1.0 / w_dim), trainable=True)
            w_value = tf.get_variable('w_value', shape=[embed_dim, w_dim], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=1.0 / w_dim), trainable=True)
            w_res = tf.get_variable('w_res', shape=[embed_dim, w_dim],
                dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=1.0/w_dim), trainable=True)

        querys = tf.tensordot(inputs, w_query, axes=(-1, 0))
        keys = tf.tensordot(inputs, w_key, axes=(-1, 0))
        values = tf.tensordot(inputs, w_value, axes=(-1, 0))

        querys = tf.stack(tf.split(querys, head_num, axis=2))  # head x batch x m x (d/head)
        keys = tf.stack(tf.split(keys, head_num, axis=2))
        values = tf.stack(tf.split(values, head_num, axis=2))

        inner_product = tf.matmul(querys, keys, transpose_b=True)  # head x batch x m x m
        normalized_att_scores = tf.nn.softmax(inner_product)
        result = tf.matmul(normalized_att_scores, values)  # h x b x m x d/h
        result = tf.concat(tf.split(result, head_num, ), axis=-1)
        result = tf.squeeze(result, axis=0)  # (batch_size,field_size,att_embedding_size*head_num)

        if use_res:
            result += tf.tensordot(inputs, w_res, axes=(-1, 0))
        if use_dropout:
            result = get_dropout(result)

        result = tf.nn.relu(result)  # (batch_size,field_size,att_embedding_size*head_num)

        layer_input = result

    if reduce_mean:
        att_output = tf.reduce_mean(result, axis=1)
    elif concat:
        att_output = tf.keras.layers.Flatten()(result)
    return att_output

def get_random_drop(inputs, max_drop=30):
    batch_size = tf.shape(inputs)[0]
    input_length = inputs.shape.as_list()[1]
    drop_num = tf.random_uniform((batch_size,), 0, max_drop, dtype=tf.float32)
    drop_gate = tf.cast(drop_num / input_length, tf.float32)
    rand_mask = tf.random_uniform(tf.shape(inputs), dtype=tf.float32)
    drop_mask = tf.greater(rand_mask, drop_gate[:, tf.newaxis])
    drop_mask_int = tf.cast(drop_mask, tf.int32)
    return inputs * drop_mask_int

def add_noise_regulation(inputs):
    rand_mask = tf.random_uniform(tf.shape(inputs), 1, self.config.factor_num, dtype=tf.int32)
    zero_mask = tf.equal(inputs, 0)
    zero_hist = tf.reduce_sum(tf.cast(zero_mask, tf.float32), axis=-1)
    tf.summary.histogram('zero_hist', zero_hist)
    return tf.where(zero_mask, rand_mask, inputs)

def get_MIL_att(inputs, D=128, num_seeds=1, return_w=False):
    # attention = Dense(D, activation='relu')(inputs)  # b x M x D
    attention = Dense(D, activation='tanh')(inputs) * Dense(D, activation='sigmoid')(inputs)  # b x M x D
    A = Dense(num_seeds)(attention)  # b x M x 1(n)
    A -= tf.cast(tf.equal(A, 0), tf.float32) * 1e9
    A = tf.nn.softmax(tf.transpose(A, [0, 2, 1]))  # b x 1 x M
    attented = tf.matmul(A, inputs)  # b x N(1) x d
    if num_seeds == 1:
        attented = tf.squeeze(attented, axis=1)
    if return_w:
        return attented, A
    else:
        return attented

def get_ten_encoding(inputs, K=32, mask=None):
    dim = inputs.shape.as_list()[-1]
    C = tf.get_variable('codewords', shape=[K, dim], initializer=glorot_uniform())
    S = tf.get_variable('scale', shape=[K], initializer=tf.initializers.random_uniform(-1, 0))
    residual = tf.tile(inputs[:, :, tf.newaxis, :], (1, 1, K, 1)) - C[tf.newaxis, tf.newaxis, ...]  # b x N x K x D
    scaled_l2 = S[tf.newaxis, tf.newaxis, :] * tf.reduce_sum(tf.pow(residual, 2), -1)  # b x N x K
    if mask is not None:
        mask = tf.squeeze(mask)  # b x N
        scaled_l2 -= 1e9 * mask[..., tf.newaxis]

    A = tf.keras.activations.softmax(scaled_l2, axis=1)  # b x N x K
    aggregate = tf.reduce_sum(A[..., tf.newaxis] * residual, axis=1)  # b x K x D
    return aggregate

def get_factor_pooling(factor, just_sort=True, filter_num=None, layer_num=1, use_activation=False, name='pooling'):
    inputs = factor
    if filter_num is None:
        # use M filters for default
        filter_num = inputs.shape[1].value

    for l in range(layer_num):
        if l > 0:
            filter_num = inputs.shape[1].value
        while len(inputs.shape) < 3:
            inputs = tf.expand_dims(inputs, axis=-1)
        embedding_dim = inputs.shape[-1].value
        with tf.variable_scope(name):
            w_pool = tf.get_variable('w%d_pooling' % l, shape=(embedding_dim, filter_num), initializer=glorot_uniform(), trainable=False)
            # b = tf.get_variable('b%d_pooling' % l, shape=(filter_num), initializer=Zeros())
        conv_res = tf.tensordot(inputs, w_pool, axes=(-1, 0))
        # conv_res += b
        if use_activation:
            conv_res = tf.nn.relu(conv_res)
        if just_sort:
            max_id = tf.argmax(conv_res, axis=1)
            pooling_res = tf.gather(inputs, max_id, axis=1)
            inputs = pooling_res
        else:
            pooling_res = tf.reduce_max(conv_res, axis=1, keepdims=False)
            inputs = pooling_res
    return pooling_res

def get_regularization(variables=None, l2_lambda=0.0001):
    if variables is None:
        variables = tf.trainable_variables()
    regularization_cost = l2_lambda * tf.reduce_sum([tf.nn.l2_loss(v) for v in variables])
    return regularization_cost

def get_dropout(inputs, keep_prob=1):
    output = tf.nn.dropout(inputs, keep_prob)
    return output

def get_dnn_output(input_layer, keep_prob=1, use_bn=False, layers_dim=[1000, 800]):
    h = [input_layer]
    for i, dim in enumerate(layers_dim):
        hidden = tf.layers.dense(inputs=h[i], units=dim, activation=tf.nn.relu)
        if use_bn:
            hidden = tf.layers.batch_normalization(hidden)
        h.append(hidden)
    out = h[-1]
    if keep_prob != 1:
        out = tf.nn.dropout(out, keep_prob)
    return out

def get_wide_output(input_layer, activation=None):
    dim = input_layer.get_shape()[1]
    out = tf.layers.dense(inputs=input_layer, units=dim, activation=activation)
    return out

def init_embedding(name, num, dim, trainable=True, add_padding=True):
    v_embedding = tf.get_variable(name, shape=[num, dim], dtype=tf.float32,
        initializer=tf.truncated_normal_initializer(stddev=1.0 / dim), trainable=trainable)
    if add_padding:
        padding = tf.constant(0.0, shape=[1, dim], dtype=tf.float32)
        v_embedding = tf.concat(axis=0, values=[padding, v_embedding])
    return v_embedding

def cross_layer(x_0, x_l, w, b):
    w_num = w.shape[1].value
    # batch_size = tf.shape(x_0)[0]
    input_dim = x_0.get_shape().as_list()[1]
    if w_num > 1:
        x_b = tf.tensordot(tf.reshape(x_l, [-1, 1, input_dim]), w, 1)  # b x 1 x w_num
        x_b = tf.transpose(x_b, [0, 2, 1])  # b x w_num x 1
        x_0 = tf.expand_dims(x_0, axis=-1)  # b x D x 1
        x_0 = tf.tile(x_0, (1, 1, w.shape[1].value))  # b x D x w_num
        return tf.squeeze(tf.matmul(x_0, x_b), axis=-1) + b  # b x D
    else:
        w = tf.squeeze(w, axis=-1)  # dim
        x_b = tf.tensordot(tf.reshape(x_l, [-1, 1, input_dim]), w, 1)  # b x 1
        return x_0 * x_b + b

def get_cross_output(input_layer, layer_num=2, activation=None, reg=None, w_num=1, name='cross', reuse=False, concat=False):
    dim = input_layer.get_shape()[1].value
    x = [input_layer]
    with tf.variable_scope('cross' + name, reuse=reuse):
        for i in range(layer_num):
            if reg is None:
                w = tf.get_variable('w{0}'.format(i), shape=(dim, w_num), initializer=tf.truncated_normal_initializer(stddev=1.0/dim))
            else:
                w = tf.get_variable('w{0}'.format(i), shape=(dim, w_num), initializer=tf.truncated_normal_initializer(stddev=1.0/dim), regularizer=reg)
            b = tf.get_variable('b{0}'.format(i), shape=(dim), initializer=tf.constant_initializer(1.0/dim))
            _x = cross_layer(x[0], x[i], w, b)
            if not concat:
                _x += x[i]
            x.append(_x)

    out = x[-1]
    if activation is not None:
        out = activation(out)
    if not concat:
        return out
    else:
        return tf.layers.flatten(tf.stack(x, axis=-1));

def get_last_state(outputs, length):
    batch_size = tf.shape(outputs)[0]
    max_length = tf.shape(outputs)[1]
    out_size = outputs.shape.as_list()[2]
    index = tf.range(0, batch_size) * max_length + tf.abs((tf.cast(length, tf.int32) - 1))
    flat = tf.reshape(outputs, [-1, out_size])
    relevant = tf.gather(flat, index)
    return relevant

def extract_axis_1d(outputs, ind):
    batch_range = tf.range(tf.shape(outputs)[0])
    indices = tf.stack([batch_range, ind], axis=1)
    res = tf.gather_nd(outputs, indices)
    return res

def get_lstm_output(name, inputs, embedding, dim, only_last=True, use_bidirection=False):
    rnn_cell = tf.contrib.rnn.BasicLSTMCell(dim, state_is_tuple=True)
    input_length = tf.reduce_sum(tf.cast(tf.not_equal(inputs, 0), tf.int32), 1, keep_dims=False)

    if use_bidirection is False:
        rnn_cell_bw = tf.contrib.rnn.BasicLSTMCell(dim, state_is_tuple=True)
        outputs, last_states = tf.nn.bidirectional_dynamic_rnn(rnn_cell,
                                                            rnn_cell_bw,
                                                            embedding,
                                                            sequence_length=input_length,
                                                            dtype=tf.float32,
                                                            scope=name)
        last_output = tf.concat([get_last_state(output, input_length) for output in outputs], axis=1)
    else:
        outputs, last_states = tf.nn.dynamic_rnn(rnn_cell,
                                                 embedding,
                                                 sequence_length=input_length,
                                                 dtype=tf.float32,
                                                 scope=name)
        last_output = get_last_state(outputs, input_length)
    if only_last:
        return last_output
    else:
        return last_output, outputs

def get_lstm_att(name, inputs, query, embedding, dim):
    rnn_cell = tf.contrib.rnn.BasicLSTMCell(dim, state_is_tuple=True)
    input_length = tf.reduce_sum(tf.cast(tf.not_equal(inputs, 0), tf.int32), 1, keep_dims=False)

    outputs, last_states = tf.nn.dynamic_rnn(rnn_cell,
                                             embedding,
                                             sequence_length=input_length,
                                             dtype=tf.float32,
                                             scope=name)
    # output = self.get_last_state(outputs, input_length)
    output = get_attention_seq(inputs, outputs, query, name=name)
    return output


def get_random_sample(inputs, sample_num=None, sample_times=1, axis=1):
    input_dim = inputs.shape[axis].value
    if sample_num is None:
        sample_num = input_dim / 10
    assert input_dim >= sample_num
    res_list = []
    for _ in range(sample_times):
        sample_id = tf.Variable(tf.random_uniform([input_dim]))
        _, indices = tf.nn.top_k(sample_id, sample_num)
        res_list.append(tf.gather(inputs, indices, axis=axis))
    if len(res_list) > 1:
        res = tf.stack(res_list, axis=-1)
    else:
        res = res_list[0]
    return res

def get_candidate_sample(label, sample_num=300):
    print('get candidate sample!!!!')
    batch_size = tf.shape(label)[0]
    label_weight = get_label_weight(label)

    rand_tb = tf.random_normal((batch_size, self.output_class_num))
    rand_tb += 100 * tf.one_hot(label, self.output_class_num) + label_weight
    _, sampled_class = tf.nn.top_k(rand_tb, sample_num)
    class_mask = tf.reduce_sum(tf.one_hot(sampled_class, depth=self.output_class_num), axis=1)
    # print('mask shape:' + str(class_mask.shape.as_list()))
    # sample_class = tf.nn.learned_unigram_candidate_sampler(tf.cast(label, tf.int64), 1, sample_num, unique=True, range_max=self.output_class_num)
    class_mask = tf.equal(class_mask, 1)
    return class_mask, sampled_class  # b x num_class, b x num_sample

def get_label_normalization(pred, epislon=0.1):
    print('get label normalization!!!!')
    sampled_class_num = tf.reduce_sum(tf.not_equal(pred, 0), axis=1)
    pred_ = (1 - epislon) * pred + epislon * 1 / sampled_class_num
    return pred_

def get_label_weight(lb, beta=0.8, batch=True):
    lb = tf.squeeze(lb)
    print('get label weight!!!!')
    hist = tf.histogram_fixed_width(lb, [0, self.output_class_num], self.output_class_num)
    if batch:
        return tf.cast(hist, tf.float32) / self._batch_size
    else:
        w_lb = tf.get_variable('label_weight', shape=(self.output_class_num), initializer=tf.initializers.constant(1/self.output_class_num), dtype=tf.float32, trainable=False)
        total_samples = tf.get_variable('total_sample', shape=(1), initializer=tf.initializers.constant(0), dtype=tf.float32, trainable=False)
        total_samples = tf.assign(total_samples, tf.cast(tf.shape(lb)[0], tf.float32) + total_samples)
        w_lb = tf.assign(w_lb, w_lb + tf.cast(hist, tf.float32))
        # tf.summary.histogram('label hist', w_lb)
        # tf.summary.scalar('total sample', tf.squeeze(total_samples, axis=-1))
        return w_lb / total_samples
        # res = beta * w_lb + (1 - beta) * tf.cast(hist, tf.float32) / self._batch_size
        # w_lb = tf.assign(w_lb, res)
        # return res

def print_cmd(cmd):
    import os
    process = os.popen(cmd)  # return file
    output = process.read()
    process.close()
    res = cmd + output
    print(res)
    return output

def get_available_gpus():
    """
    code from http://stackoverflow.com/questions/38559755/how-to-get-current-available-gpus-in-tensorflow
    """
    from tensorflow.python.client import device_lib as _device_lib
    local_device_protos = _device_lib.list_local_devices()
    return [x.name for x in local_device_protos]

def display_label_distribution(lb, label_weight=None, multiply=1):
    if label_weight is None:
        lbw = get_label_weight(lb, beta=0.5, batch=False)
    else:
        lbw = tf.squeeze(label_weight)
    hist_height = 200
    img_base = tf.tile(tf.expand_dims(tf.range(hist_height, 0, -1, dtype=tf.float32), axis=-1), [1, self.output_class_num])
    label_hist = lbw * hist_height * multiply
    img_true_hist = tf.cast(tf.greater(img_base, label_hist), tf.float32)
    tf.summary.image('label_hist', img_true_hist[tf.newaxis, ..., tf.newaxis])

def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)

    # apply sin to even indices in the array; 2i
    sines = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    cosines = np.cos(angle_rads[:, 1::2])

    pos_encoding = np.concatenate([sines, cosines], axis=-1)

    pos_encoding = pos_encoding[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


def create_padding_mask(seq):
    seq = tf.cast(tf.equal(seq, 0), tf.float32)
    return seq[tf.newaxis, :, tf.newaxis, :]  # (1, batch_size,  1, seq_len)


def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable
          to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
    output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


def MultiHeadAttention(inputs, mask, d_model, num_heads):
    depth = d_model // num_heads

    def split_heads(x, batch_size):
        x = tf.reshape(x, (batch_size, -1, num_heads, depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    batch_size = tf.shape(inputs)[0]

    q = tf.keras.layers.Dense(d_model)(inputs)  # (batch_size, seq_len, d_model)
    k = tf.keras.layers.Dense(d_model)(inputs)  # (batch_size, seq_len, d_model)
    v = tf.keras.layers.Dense(d_model)(inputs)  # (batch_size, seq_len, d_model)

    q = split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
    k = split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
    v = split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

    # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
    # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
    scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)

    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

    concat_attention = tf.reshape(scaled_attention, (batch_size, -1, d_model))  # (batch_size, seq_len_q, d_model)

    output = tf.keras.layers.Dense(d_model)(concat_attention)  # (batch_size, seq_len_q, d_model)

    return output, attention_weights


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
                              tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
                          tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
                            ])


def EncoderLayer(inputs, mask, training, d_model, num_heads, dff, rate=0.1):
    ffn = point_wise_feed_forward_network(d_model, dff)

    attn_output, _ = MultiHeadAttention(inputs, mask, d_model, num_heads)  # (batch_size, input_seq_len, d_model)
    attn_output = tf.keras.layers.Dropout(rate)(attn_output, training=training)
    out1 = tf.contrib.layers.layer_norm(inputs + attn_output)  # (batch_size, input_seq_len, d_model)

    ffn_output = ffn(out1)  # (batch_size, input_seq_len, d_model)
    ffn_output = tf.keras.layers.Dropout(rate)(ffn_output, training=training)
    out2 = tf.contrib.layers.layer_norm(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

    return out2


def transformer_encoder(x, mask, d_model, dnn_size, seq_len, training=True, num_layers=2, num_heads=8, rate=0.1):
    # seq_len = tf.shape(x)[1]
    pos_encoding = positional_encoding(seq_len, d_model)

    x *= tf.sqrt(tf.cast(d_model, tf.float32))
    x += pos_encoding[:, :seq_len, :]

    x = tf.keras.layers.Dropout(rate)(x, training=training)

    for i in range(num_layers):
        x = EncoderLayer(x, mask, training, d_model, num_heads, dnn_size, rate)

    return x  # (batch_size, input_seq_len, d_model)


################## set transformer ###################
def MAB(Q, K, dim_V, num_heads, ln=False, k=None, mask=None):
    fc_q = Dense(dim_V)
    fc_k = Dense(dim_V)
    fc_v = Dense(dim_V)
    if ln:
        ln0 = tf.contrib.layers.layer_norm
        ln1 = tf.contrib.layers.layer_norm
    fc_o = Dense(dim_V)

    Q = fc_q(Q)
    K, V = fc_k(K), fc_v(K)

    # emb_dim = dim_V // num_heads
    Q_ = tf.stack(tf.split(Q, num_heads, axis=-1))  # h x b x m x d
    K_ = tf.stack(tf.split(K, num_heads, axis=-1))  # h x b x n x d
    V_ = tf.stack(tf.split(V, num_heads, axis=-1))

    score = tf.matmul(Q_, K_, transpose_b=True)  # h x b x n x m
    # zero_mask = tf.cast(tf.equal(score, 0), tf.float32)
    if mask is not None:
        score -= mask * 1e9
    if k is not None:
        top_min = tf.reduce_min(tf.nn.top_k(score, k).values, axis=-1, keepdims=True)
        not_in_topk_mask = tf.cast(tf.less(score, top_min), tf.float32)
        score -= not_in_topk_mask * 1e9

    A = tf.nn.softmax(score / tf.sqrt(tf.cast(dim_V, tf.float32)))
    output = tf.concat(tf.split(Q_ + tf.matmul(A, V_), num_heads, axis=0), axis=-1)  # 1 x b x m x D
    output = tf.squeeze(output, axis=0)
    output = ln0(output) if 'ln0' in dir() else output
    output += tf.nn.relu(fc_o(output))
    # output = tf.nn.relu(output + fc_o(output))
    output = ln1(output) if 'ln0' in dir() else output
    return output

#  只有当X作为key时才需要mask
def SAB(X, dim_out, num_heads, ln=False, name='sab', k=None, mask=None):
    return MAB(X, X, dim_out, num_heads, ln, k, mask)


def ISAB(X, dim_out, num_heads=4, num_inds=16, ln=False, name="isab", k=None, mask=None):
    induce = tf.get_variable(name + 'induce', shape=[1, num_inds, dim_out], dtype=tf.float32, initializer=tf.glorot_uniform_initializer())
    H = MAB(tf.tile(induce, (tf.shape(X)[0], 1, 1)), X, dim_out, num_heads, ln, k, mask)
    return MAB(X, H, dim_out, num_heads, ln, k)


def PMA(X, dim, num_heads=4, num_seeds=48, ln=False, k=None, mask=None):
    S = tf.get_variable('S', shape=[1, num_seeds, dim], dtype=tf.float32, initializer=tf.glorot_uniform_initializer())
    return MAB(tf.tile(S, (tf.shape(X)[0], 1, 1)), X, dim, num_heads, ln, k, mask)
