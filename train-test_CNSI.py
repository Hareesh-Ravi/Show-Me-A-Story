# -*- coding: utf-8 -*-
"""
Created on Thu Aug 14 13:28:09 2018

@author: HareeshRavi
"""

import numpy as np
import csv
import keras
from keras import backend as K
import tensorflow as tf
from load_VIST_data_seq import loadData
import sys
import math
import pickle
import configparser
import config_all
import modelArch


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


def csvnumread(filename):
    out_list = []
    with open(filename, "r") as readtextfile:
        reader = csv.reader(readtextfile)
        for row in reader:
            if (len(row) > 1):
                out_list.append([int(i) for i in row])
    return out_list


def get_sent_img_feats(config, data, num_words, embedding_matrix):

    # Load Sentence encoder and initialize it's weights
    model_path = sys.argv[1]
    print('model:{}'.format(model_path))
    SentEncoder = modelArch.Model_Sent_Img(num_words, embedding_matrix)
    SentEncoder.load_weights(model_path, by_name=True)

    # Extract Sent and Img features using sent encoder
    [encoded_sents, encoded_imgs] = SentEncoder.predict(data, batch_size=128)
    
    # get stories and corresponding groud truth images in story x seq format
    num_sents = 5
    feat_temp = []
    labels_temp = []
    feat = []
    labels = []
    for i in range(0, len(train_data[0]), num_sents):
        for j in range(0, num_sents):
            ind = i + j
            sent = encoded_sents[ind]
            feat_temp.append(sent)
            img = encoded_imgs[ind]
            labels_temp.append(img)
        feat.append(feat_temp)
        labels.append(labels_temp)
        feat_temp = []
        labels_temp = []

    return feat, labels


def train(config, train_data, num_words, embedding_matrix):

    CohVecTrainFileName = config['DATALOADER']['CohVecTrain']
    trainImgFileName = config['DATALOADER']['trainImgId']
    epochs = config['MODEL_Story_ImgSeq_PARAMS']['epochs']
    BS = config['MODEL_Story_ImgSeq_PARAMS']['batchsize']
    CNSImodelname = config['OUTPUTFILENAMES']['CNSImodelname']

    coh_sent_train_full = np.expand_dims(np.load(CohVecTrainFileName), axis=1)
    coh_sent_train = np.delete(coh_sent_train_full,
                               [9925, 9926, 9927, 9928, 9929, 40154], axis=0)
    
    # load train image IDs
    train_image = csvnumread(trainImgFileName)

    # Load Story Encoder model architecture
    StoryEncoder = modelArch.Model_Story_ImgSeq_CNSI()

    x_train, y_train = get_sent_img_feats(train_data, num_words,
                                          embedding_matrix)

    NumTrain = math.floor(np.size(x_train, 0)/BS) * BS
    x_train = np.array(x_train[0:NumTrain][:][:])
    y_train = np.array(y_train[0:NumTrain][:][:])

    # Custom shuffle and train
    for i in range(0, epochs):
        tempIDS = np.array(list(train_image))
        tempIDS = tempIDS[0:NumTrain, :]
        tempCoh = coh_sent_train[0:NumTrain, :]
        np.repeat(tempCoh, 5, axis=1)

        iterations = math.floor(np.size(tempIDS, 0) / BS)

        p = np.zeros(np.size(tempIDS, 0), dtype=float)
        temptrain = np.zeros_like(tempIDS)
        coh_train = np.zeros((NumTrain, 5, 64))
        permsamp, TotIdx, Totinv, TotCount = np.unique(tempIDS, axis=0,
                                                       return_counts=True,
                                                       return_index=True,
                                                       return_inverse=True)
        x_train1 = np.zeros_like(x_train)
        y_train1 = np.zeros_like(y_train)
        ToNotDel = set(list(TotIdx[list(TotCount == 1)]))
        for j in range(0, iterations):
            samp, sampIdx, count = np.unique(tempIDS, axis=0,
                                             return_index=True,
                                             return_counts=True)
            samptemp = list(samp[:, 0])
            sampIdx = sampIdx[samptemp != -1]
            p[p != 0] = 0
            p[sampIdx] = 1 / len(sampIdx)
            try:
                sampled_arr = list(np.random.choice(np.size(tempIDS, 0), BS,
                                                    replace=False, p=p))
            except ValueError:
                p[:] = 1 / np.size(p, 0)
                sampled_arr = list(np.random.choice(np.size(tempIDS, 0), BS,
                                                    replace=False, p=p))

            temptrain[BS * j:BS * (j + 1), :] = tempIDS[sampled_arr, :]
            x_train1[BS * j:BS * (j + 1), :, :] = x_train[sampled_arr, :, :]
            y_train1[BS * j:BS * (j + 1), :, :] = y_train[sampled_arr, :, :]
            coh_train[BS * j:BS * (j + 1), :, :] = tempCoh[sampled_arr, :, :]
            if j > (iterations - 100):
                sampled_arr = list(set(sampled_arr) - ToNotDel)
            tempIDS[sampled_arr, 0] = -1
            p[sampled_arr] = -1

        StoryEncoder.fit([x_train1, coh_train], y_train1, epochs=1,
                         batch_size=BS, verbose=1)
        print("epoch: ", i, " completed")

    StoryEncoder.save(CNSImodelname)


def test(config, test_data, num_words, embedding_matrix):

    CohVecTestFileName = config['DATALOADER']['CohVecTest']
    testinsamplename = config['DATALOADER']['testSamplesFileName']
    CNSImodelname = config['OUTPUTFILENAMES']['CNSImodelname']
    testoutfeatname = config['OUTPUTFILENAMES']['testImgOutFeatSaveName']

    x_test, y_test = get_sent_img_feats(test_data, num_words,
                                        embedding_matrix)
    test200_lines = [line.rstrip('\n') for line in open(testinsamplename)]
    coh_sent_test = np.expand_dims(np.load(CohVecTestFileName), axis=1)

    sent_test200 = []
    for ind in test200_lines:
        ind = int(ind)
        sent_test200.append(x_test[0][ind])
    sent_test200 = np.array(sent_test200)
    sub_test200_coh = coh_sent_test[test200_lines, :, :]
    sub_test200_coh = np.repeat(sub_test200_coh, 5, axis=1)

    model_lstm = keras.models.load_model(
            CNSImodelname,
            custom_objects={'orderEmb_loss': orderEmb_loss})

    out_fea = model_lstm.predict([sent_test200, sub_test200_coh])
    with open(testoutfeatname, 'wb') as fp:
        pickle.dump(out_fea, fp)


if __name__ == '__main__':

    config = config_all.create_config()

    num_words, embedding_matrix, train_data, valid_data, test_data = loadData(
            config)

    booltrain = config['DEFAULT']['booltrain']
    booltest = config['DEFAULT']['booltest']

    if booltrain:
        train(config, train_data, num_words, embedding_matrix)
    if booltest:
        test(config, test_data, num_words, embedding_matrix)
