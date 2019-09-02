# -*- coding: utf-8 -*-

"""
Created on Thu Aug 14 13:28:09 2018

@author: HareeshRavi
"""

import keras
from keras.models import Model
from keras.layers import TimeDistributed, Dense, GRU, Dropout, Embedding
from keras.layers import Input, concatenate
from keras import optimizers
from keras import backend as K
import tensorflow as tf
import utils_vist
from keras.initializers import Constant

# network architecture for baseline experiment
def baseline(modconfig, num_words, embedding_matrix):
      
    # read config
    MAX_SEQUENCE_LENGTH = modconfig['MAX_SEQUENCE_LENGTH']
    word_embd_dim = modconfig['wd_embd_dim']
    sent_feat_dim = modconfig['sent_fea_dim']
    img_feat_dim = modconfig['img_fea_dim']
    embedding_layer = Embedding(num_words,
                                word_embd_dim,
                                embeddings_initializer=Constant(embedding_matrix),
                                input_length=MAX_SEQUENCE_LENGTH,
                                name='wd_embedding_layer',
                                trainable=False)

    input_sent = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32', 
                       name='input1')
    input_img = Input(shape=(img_feat_dim,), dtype='float32', name='input2')

    # Encode sentence
    # This embedding layer will encode the input sequence
    # into a sequence of dense 512-dimensional vectors.
    x1 = embedding_layer(input_sent)

    # using GRU instead of LSTM
    Encode_sent = keras.layers.recurrent.GRU(sent_feat_dim, name='gru')(x1)
    
    Encode_sent_normed = keras.layers.Lambda(lambda x: K.abs(K.l2_normalize(
            x, axis=1)), name='sentFeaNorm')(Encode_sent)
    
    # encoding image feat
    Encode_img = Dense(sent_feat_dim, activation='linear', 
                       name='imgEncode')(input_img)
    Encode_img_normed = keras.layers.Lambda(lambda x: K.abs(K.l2_normalize(
            x, axis=1)), name='imgFeaNorm')(Encode_img)
    
    # define a Lambda merge layer
    lambda1 = keras.layers.Lambda(utils_vist.contrastive_loss, 
                                  output_shape=utils_vist.edis_outputshape, 
                                  name = 'orderEmbd')
    main_output = lambda1([Encode_sent_normed, Encode_img_normed])
    
    lambda2 = keras.layers.Lambda(utils_vist.retriv_acc, 
                                  output_shape=utils_vist.edis_outputshape, 
                                  name = 'Recall_1')
    acc_output = lambda2([Encode_sent_normed, Encode_img_normed])
    
#    main_output = keras.layers.merge([Encode_sent_normed, Encode_img_normed], 
#                                     mode=utils_vist.contrastive_loss,
#                                     output_shape=utils_vist.edis_outputshape, 
#                                     name='orderEmbd')
#
#    acc_output = keras.layers.merge([Encode_sent_normed, Encode_img_normed],
#                                    mode=utils_vist.retriv_acc,
#                                    output_shape=utils_vist.edis_outputshape, 
#                                    name='Recall_1')

    baselinemodel = Model(inputs=[input_sent, input_img], 
                          outputs=[main_output, acc_output])
    baselinemodel.compile(loss=['mean_absolute_error', 
                                 utils_vist.MyCustomLoss], 
                           optimizer='adam', 
                           loss_weights=[1,0])
    baselinemodel.summary()
    return baselinemodel

# sequential order embedding loss function
def orderEmb_loss(y_true, y_pred):
    y_true = K.l2_normalize(K.abs(y_true), axis=2)
    y_pred = K.l2_normalize(K.abs(y_pred), axis=2)
    y_truemod = K.expand_dims(y_true, axis=0)
    y_predmod = K.expand_dims(y_pred, axis=1)
    order_viola = K.permute_dimensions(K.sum(K.pow(K.maximum(
            K.cast(0, 'float32'), y_predmod - y_truemod), 2), axis=3),
            (2, 0, 1))
    diagErr_im = K.expand_dims(tf.matrix_diag_part(order_viola), axis=2)
    diagErr_s = K.expand_dims(tf.matrix_diag_part(order_viola), axis=1)
    margin = K.cast(0.05, 'float32')

    # negative samples shuffling code
    def body_func(idx, bool_in):
        idx = K.variable([0, 1, 2, 3, 4], dtype='int32')
        shuff_idx = tf.random_shuffle(idx)
        bool_in = tf.reduce_any(tf.equal(shuff_idx, idx))
        return shuff_idx, bool_in

    cond_func = lambda idx, bool_in: bool_in

    idx = K.variable([0, 1, 2, 3, 4], dtype='int32')
    bool_in = True

    def shuffle_tensoridx(arr_inp):
        result = tf.while_loop(cond_func, body_func, [idx, bool_in])
        arr_b = tf.gather(arr_inp, result[0])
        return arr_b

    ypred_shuffle = tf.map_fn(shuffle_tensoridx, y_pred)
    ypredmod_shuffle = K.expand_dims(ypred_shuffle, axis=1)

    # loss calculation according to objective function
    order_violb = K.permute_dimensions(K.sum(K.pow(K.maximum(
            K.cast(0, 'float32'), ypredmod_shuffle - y_truemod), 2),
            axis=3), (2, 0, 1))

    cost_im = K.maximum(K.cast(0, 'float32'),
                        diagErr_im - order_violb + margin)
    cost_s = K.maximum(K.cast(0, 'float32'),
                       diagErr_s - order_violb + margin)
    #    mask = tf.reverse(K.eye(32),[0])
    temp = tf.ones([32, 32], tf.float32)
    mask = tf.matrix_set_diag(temp, tf.zeros([32], tf.float32))
    tot_cost = K.sum(K.sum(K.sum(tf.multiply(cost_im + cost_s, mask),
                                 axis=2), axis=1), axis=0)
    # print(tot_cost)
    return tot_cost / 32


# stage 1 of proposed model
def stage1(config, num_words, embedding_matrix):
    
    modconfig = config['stage1']
    # read config
    MAX_SEQUENCE_LENGTH = modconfig['MAX_SEQUENCE_LENGTH']
    word_embed_dim = modconfig['wd_embd_dim']
    sent_feat_dim = modconfig['sent_fea_dim']
    img_feat_dim = modconfig['img_fea_dim']

    embedding_layer = Embedding(num_words,
                                word_embed_dim,
                                embeddings_initializer=Constant(embedding_matrix),
                                input_length=MAX_SEQUENCE_LENGTH,
                                name='vist_wd_embedding_layer',
                                trainable=False)

    input_sent = Input(shape=(MAX_SEQUENCE_LENGTH,),
                       dtype='int32', name='input1')

    x1 = embedding_layer(input_sent)

    # encoding each sentence GRU over words
    Encode_sent = keras.layers.recurrent.GRU(sent_feat_dim, name='gru')(x1)
    Encode_sent_normed = keras.layers.Lambda(lambda x: K.abs(
            K.l2_normalize(x, axis=1)), name='sentFeaNorm')(Encode_sent)

    input_img = Input(shape=(img_feat_dim,), dtype='float32', name='input2')

    # encoding image feat
    Encode_img = Dense(sent_feat_dim, activation='linear',
                       name='imgEncode')(input_img)
    Encode_img_normed = keras.layers.Lambda(lambda x: K.abs(
            K.l2_normalize(x, axis=1)), name='imgFeaNorm')(Encode_img)

    sentence_model = Model(inputs=[input_sent, input_img], outputs=[
            Encode_sent_normed, Encode_img_normed])
        
    sentence_model.get_layer('vist_wd_embedding_layer').trainable = False
    
    sentence_model.compile(loss=['mean_absolute_error', 
                                 utils_vist.MyCustomLoss], 
                           optimizer='adam', 
                           loss_weights=[1,0])
    sentence_model.summary()
    return sentence_model

# stage 2 of proposed model
def stage2(config, num_words, embedding_matrix):
    
    # read config
    modconfig = config['stage2']
    hidden_size1 = modconfig['hidden_size1']
    hidden_size2 = modconfig['hidden_size2']
    hidden_size3 = modconfig['hidden_size3']
    learningrate = modconfig['learningrate']
    x_len = modconfig['x_len']
    x_dim = modconfig['x_dim']
    y_dim = modconfig['y_dim']
    cohfeat_dim = modconfig['cohfeat_dim']

    opt = optimizers.adam(lr=learningrate)

    # input from sentence encoder and image encoder
    i11 = Input(shape=(x_len, x_dim), name='txt_input')
    d11 = Dropout(0.2, input_shape=(x_len, x_dim), name='layer_1_drop')(i11)
    
    # story encoder
    g11 = GRU(hidden_size1, return_sequences=True, name='layer_1_gru')(d11)
    g12 = GRU(hidden_size2, return_sequences=True, name='layer_2_gru')(g11)
    g13 = GRU(hidden_size3, return_sequences=True, name='layer_3_gru')(g12)
    
    if cohfeat_dim:
        # concatenate coherence vector as input
        i21 = Input(shape=(x_len, cohfeat_dim), name='coh_input')
        m1 = concatenate([g13, i21], axis=2, name='concatlayer1')
        # final dense layer
        td11 = TimeDistributed(Dense(y_dim), name='layer_4_timedist')(m1)
        # final model
        story_model = Model(inputs=[i11, i21], outputs=td11)
    else:
        # final dense layer
        td11 = TimeDistributed(Dense(y_dim), name='layer_4_timedist')(g13)
        # final model
        story_model = Model(inputs=i11, outputs=td11)
    

    # order embedding loss
    story_model.compile(loss=orderEmb_loss, optimizer=opt, 
                        metrics=['accuracy'])

    story_model.summary()

    return story_model

