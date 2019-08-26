# -*- coding: utf-8 -*-
"""
Created on Thu Aug 14 13:28:09 2018

@author: HareeshRavi
"""

import numpy as np
import keras
from load_VIST_data_seq import loadData
import sys
import math
import pickle
import json
from config_all import create_config
import model
import utils_vist
import time


def get_sent_img_feats(config, data, num_words, embedding_matrix):

    # Load Sentence encoder and initialize it's weights
    model_path = sys.argv[1]
    print('model:{}'.format(model_path))
    SentEncoder = model.Model_Sent_Img(num_words, embedding_matrix)
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


def pretrain(config, train_data, valid_data, num_words, embedding_matrix):
    
    return True

def trainstage1(config, train_data, valid_data, num_words, embedding_matrix):
    
    batchsize = config['stage1']['batchsize']
    epochs = config['stage1']['epochs']
    MAX_SEQUENCE_LENGTH = config['stage1']['MAX_SEQUENCE_LENGTH']
    img_fea_dim = config['stage1']['img_fea_dim']
    
    # load all data
    print('Loading data')
    num_words, embedding_matrix, train_data, valid_data, test_data = loadData()
    

    print('no of total train samples: {0:d}'.format(len(train_data[0])))
    print('no of total val samples: {0:d}'.format(len(valid_data[0])))
    
    train_num = math.floor(np.size(train_data[0], 0)/batchsize) * batchsize
    valid_num = math.floor(np.size(valid_data[0], 0)/batchsize) * batchsize

    print('no of train samples: {0:d}'.format(train_num))
    print('no of val samples: {0:d}'.format(valid_num))

    # create fake labels for mycustomloss function
    fake_label = np.zeros(train_num)
    train_label = [fake_label, fake_label]
    fake_label_v = np.zeros(valid_num)
    valid_label = [fake_label_v, fake_label_v]
    
    # consider only valid_num samples for easily looping through each epoch
    valid_data[0] = valid_data[0][0:valid_num]
    valid_data[1] = valid_data[1][0:valid_num]
    
    # load model architecture
    model_base = model.baseline(num_words, embedding_matrix)
    model_base.compile(loss=['mean_absolute_error', model.MyCustomLoss], 
                       optimizer='adam', 
                       loss_weights=[1,0])
    model_base.summary()

    filepath = "./tmp/weights-improvement-{epoch:02d}.h5"
    checkpointer = keras.callbacks.ModelCheckpoint(filepath, 
                                                   monitor='val_loss', 
                                                   verbose=0,
                                                   save_best_only=False, 
                                                   save_weights_only=False, 
                                                   mode='auto', 
                                                   period=1)
    
    start_time = time.time()
    results = []
    
    # get input data
    sent_data = train_data[0]
    img_data = train_data[1]
    ids_data = train_data[2]
    
    #text_data_idx = np.arange(200704)
    iterations = math.floor(np.size(img_data, 0)/batchsize)
    
    # get only as many samples as train_num
    sent_data = sent_data[0:train_num, :]
    img_data = img_data[0:train_num, :]
    ids_data = ids_data[0:train_num]
    # Perform training by manually shuffling to remove duplicates
    for i in range(0, epochs):
        xtrain = np.zeros((batchsize*iterations, MAX_SEQUENCE_LENGTH), 
                          dtype=float)
        ytrain = np.zeros((batchsize*iterations, img_fea_dim), dtype=float)
        tempIDS = np.array(ids_data)
        textIDS = np.array(sent_data)
        p = np.zeros(np.size(tempIDS), dtype=float)
        permsamp, TotIdx, TotCount = np.unique(tempIDS, return_counts=True, 
                                               return_index=True)
        ToNotDel = set(list(TotIdx[list(TotCount==1)]))
        for j in range(0, iterations): 
            samp,sampIdx,count = np.unique(tempIDS, return_index=True, 
                                           return_counts=True)
            sampIdx = sampIdx[samp!=-1]           
            
            p[p!=0] = 0
            p[sampIdx] = 1/len(sampIdx)
            
            try:
                sampled_arr = list(np.random.choice(np.size(tempIDS, 0), 
                                                    batchsize, 
                                                    replace=False, p=p))
            except ValueError:
                p[:] = 1/np.size(p,0)
                sampled_arr = list(np.random.choice(np.size(tempIDS, 0), 
                                                    batchsize, 
                                                    replace=False, p=p))
            
            ytrain[batchsize*j:batchsize*(j+1), :] = img_data[sampled_arr, :]
            xtrain[batchsize*j:batchsize*(j+1), :] = sent_data[
                    sampled_arr, :]            
            if j > (iterations - 100):
                sampled_arr = list(set(sampled_arr)-ToNotDel)
            
            tempIDS[sampled_arr] = -1
            textIDS[sampled_arr] = -1
            p[sampled_arr] = -1
            
        train_input = [xtrain, ytrain]
        history = model_base.fit(train_input, train_label,
                                 validation_data = (valid_data, valid_label), 
                                 batch_size=batchsize, 
                                 verbose=1, 
                                 callbacks=[checkpointer])

    average_time_per_epoch = (time.time() - start_time) / epochs

    results.append((history, average_time_per_epoch))
    model_base.save('baseline_' + params['general']['date'] + '.h5')
    return model_base

def trainstage2(config, train_data, num_words, embedding_matrix):

    traindir = config['general']['datadir'] + 'train/'
    epochs = config['stage2']['epochs']
    BS = config['stage2']['batchsize']
    CNSImodelname = config['savemodel']

    coh_sent_train = np.expand_dims(np.load(traindir + 
                                                 'cohvec_train.npy'), axis=1)
    
    # load train image IDs
    train_image = utils_vist.getImgIds(traindir + 'train_image.csv')

    # Load Story Encoder model architecture
    StoryEncoder = model.Model_Story_ImgSeq()

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
            custom_objects={'orderEmb_loss': model.orderEmb_loss})

    out_fea = model_lstm.predict([sent_test200, sub_test200_coh])
    with open(testoutfeatname, 'wb') as fp:
        pickle.dump(out_fea, fp)


if __name__ == '__main__':

    try:
        params = json.load(open('config.json'))
    except FileNotFoundError:
        params = create_config()
    print('model name:')
    print(params['model'])
    if (params['model'] == 'nsi') and (
            params['stage2']['cohfeat_dim'] is not None):
        raise ValueError('For NSI model, make cohfeat_dim as None')
    print('general config:')
    print (json.dumps(params['general'], indent=2))
    print('stage 1 model parameters:')
    print (json.dumps(params['stage1'], indent=2))
    print('stage 2 model parameters:')
    print (json.dumps(params['stage2'], indent=2))

    num_words, embedding_matrix, train_data, valid_data, test_data = loadData(
            params)
    
    if params['pretrain']:
        premodel = pretrain(params['stage1'], train_data, num_words, 
                            embedding_matrix)
    if params['train']:
        
        stage1_model = trainstage1(params['stage1'], train_data, num_words, 
                                   embedding_matrix)

        stage2_model = trainstage2(params['stage2'], train_data, num_words, 
                                   embedding_matrix)
    if params['test']:
        test(params, test_data, num_words, embedding_matrix)
