# -*- coding: utf-8 -*-
"""
Created on Thu Aug 14 13:28:09 2018

@author: HareeshRavi
"""

import numpy as np
import keras
from load_data import loadData
import math
import pickle
import json
from configAll import create_config
import modelArch
import utils_vist
import time
from keras.preprocessing import sequence
import os
from data_provider_mscoco import getDataProvider
import pdb

# get sentence vectors using stage 1 of trained network
def get_sent_img_feats_stage1(config, data, num_words, embedding_matrix):

    # Load Sentence encoder and initialize it's weights
    model_path = (config['savemodel'] + 'stage1_' + 
                  config['date'] + '.h5')
    print('model:{}'.format(model_path))
    SentEncoder = modelArch.stage1(config, num_words, embedding_matrix)
    SentEncoder.load_weights(model_path, by_name=True)

    # Extract Sent and Img features using sent encoder
    [encoded_sents, encoded_imgs] = SentEncoder.predict([data[0], data[1]], 
                                                        batch_size=128)
    
    # get stories and corresponding groud truth images in story x seq format
    num_sents = 5
    feat_temp = []
    labels_temp = []
    ids_temp = []
    feat = []
    labels = []
    ids = []
    tot = len(data[0]) / num_sents
    k = 1
    for i in range(0, len(data[0]), num_sents):
        for j in range(0, num_sents):
            ind = i + j
            sent = encoded_sents[ind]
            feat_temp.append(sent)
            img = encoded_imgs[ind]
            labels_temp.append(img)
            ids_temp.append(data[2][ind])
        feat.append(feat_temp)
        labels.append(labels_temp)
        ids.append(ids_temp)
        feat_temp = []
        labels_temp = []
        ids_temp = []
        print('obtained stage1 feats for {}/{} stories'.format(k, tot), 
              end='\r')
        k += 1
    return feat, labels, ids

# preprocessing for pretraining model
def preProBuildWordVocab(config, sentence_iterator, word_count_threshold):
    max_features = config['pretrain']['MAX_NB_WORDS']
    wd_embd_dim = config['pretrain']['wd_embd_dim']
    print('Indexing word vectors.')
    embeddings_index = {}
    f = open(os.path.join(config['datadir'], 'glove.6B.300d.txt'), 'r', 
             encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    # count up all word counts so that we can threshold
    # this shouldnt be too expensive of an operation
    print ("preprocessing & creating vocab based on word count thresh %d" % (
            word_count_threshold ))
    t0 = time.time()
    word_counts = {}
    nsents = 0
    for sent in sentence_iterator:
        nsents += 1
        for w in sent['tokens']:
            word_counts[w] = word_counts.get(w, 0) + 1
    vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
    vocab.sort()
    print ('filtered words from %d to %d in %.2fs' % (len(word_counts), 
                                                      len(vocab), 
                                                      time.time() - t0))

    ixtoword = {}
    # period at the end of the sentence. make first dimension be end token
    ixtoword[0] = '.'  
    wordtoix = {}
    wordtoix['#START#'] = 0 # make first vector be the start token
    print('Preparing embedding matrix.')
    num_words = min(max_features, len(vocab) + 1)
    embedding_matrix = np.zeros((num_words, wd_embd_dim))
    ix = 1
    for w in vocab:
        if ix < num_words:
            wordtoix[w] = ix
            ixtoword[ix] = w
            embedding_vector = embeddings_index.get(w)
            if embedding_vector is not None:
              # words not found in embedding index will be all-zeros.
              embedding_matrix[ix] = embedding_vector
            ix += 1
        else:
            continue
      # prepare embedding matrix

    word_counts['.'] = nsents

    return wordtoix, embedding_matrix, num_words

# pretraining model on coco dataset. This is mostly a keras implementation of 
# ivan vendrov's order embedding based joint image caption embedding learning
def pretrain(config, dataset):
    
    epochs = config['pretrain']['epochs']
    max_length = config['pretrain']['MAX_SEQUENCE_LENGTH']
    img_feat_dim = config['pretrain']['img_fea_dim']
    word_count_threshold = config['pretrain']['word_count_threshold']
    modelname = (config['savemodel'] + 'stage1_pretrain_' + 
                 config['date'] + '.h5')
    print('Loading image-sentence pair')
    dp = getDataProvider(dataset)
    # stores various misc items that need to be passed around the framework
    misc = {}  
    # calculate how many iterations we need
    num_sentences_total = (dp.getSplitSize('train', ofwhat='sentences') + 
                           dp.getSplitSize('restval', ofwhat='sentences'))
    data_train = dp.split['train'] + dp.split['restval']  
    # we use ~110K data (80k + 30k)for trainining
    data_valid = dp.split['test'][0:1024]
    print('num of unique image:{} for train and {} for validation'.format(
            len(data_train), len(data_valid)))

    global batch_size
    batch_size = config['pretrain']['batchsize']
    num_sents_ori = len(data_train) * 5
    train_num = math.floor(num_sents_ori/batch_size) * batch_size
    #train_num = 1024
    print(train_num)
    #train_num = batch_size

    # go over all training sentences and find the vocabulary we want to use, 
    # i.e. the words that occur at least word_count_threshold number of times
    misc['wordtoix'], embedding_matrix, num_words = preProBuildWordVocab(
            config, dp.iterSentences_train_val(), word_count_threshold)

    # generate training pairs of image feat and sentence
    print('generate image_feat and sent idx pair...')
    imgs_feat = []  # np.zeros((train_num,img_feat_dim),dtype='float32')
    sent_widx = []  # np.zeros((train_num,max_length),dtype='int32')
    ctx_sents = 0
    wordtoix = misc['wordtoix']
    
    #one image with 5 sentence
    for i in range(5):
	#if train_num >= 0 and ctx_sents >= train_num: break
        for j, img in enumerate(data_train):
            if train_num >= 0 and ctx_sents >= train_num: break
            sent = data_train[j]['sentences'][i]
            image = dp._getImage(img)
            image_feat = image['feat']
            sentence = dp._getSentence(sent)
            ix = [0] + [
                    wordtoix[w] for w in sentence['tokens'] if w in wordtoix]
            imgs_feat.append(image_feat)
            sent_widx.append(ix)
            ctx_sents = ctx_sents + 1
            #if train_num >= 0 and ctx_sents >= train_num: break

    imgs_feat = sequence.pad_sequences(imgs_feat, img_feat_dim, 
                                       dtype='float32')
    sent_widx = sequence.pad_sequences(sent_widx, max_length)
    fake_label = np.zeros(len(sent_widx))
    print('Done...we have {} imgs and {} sents.'.format(np.shape(imgs_feat), 
                                                        np.shape(sent_widx)))

    #load valid data
    print('generate validation image_feat and sent idx pair...')
    imgs_feat_v = []  
    sent_widx_v = []  
    for i, img in enumerate(data_valid):
        image = dp._getImage(img)
        image_feat = image['feat']
        imgs_feat_v.append(image_feat)
        sent = img['sentences'][0]
        sentence = dp._getSentence(sent)
        ix = [0] + [wordtoix[w] for w in sentence['tokens'] if w in wordtoix]
        sent_widx_v.append(ix)
    imgs_feat_v = sequence.pad_sequences(imgs_feat_v, img_feat_dim, 
                                         dtype='float32')
    sent_widx_v = sequence.pad_sequences(sent_widx_v, max_length)
    fake_label_v = np.zeros(len(sent_widx_v))
    print('Done...we have {} imgs and {} sents.'.format(np.shape(imgs_feat_v), 
                                                        np.shape(sent_widx_v)))
    # Compile and train different models while measuring performance.
    results = []

    pretrain_model = modelArch.baseline(config['pretrain'], num_words, 
                                        embedding_matrix)
    for l in pretrain_model.layers:
        print('layer name: {}, trainable: {}'.format(l.name, str(l.trainable)))
    filepath = config['savemodel'] + "tmp/stage1_pretrain_coco-{epoch:02d}.h5"
    checkpointer = keras.callbacks.ModelCheckpoint(filepath, 
                                                   monitor='val_loss', 
                                                   verbose=0,
                                                   save_best_only=False, 
                                                   save_weights_only=False, 
                                                   mode='auto', period=1)
    
    start_time = time.time()

    train_input = [sent_widx, imgs_feat]
    train_label = [fake_label, fake_label]
    valid_input = [sent_widx_v, imgs_feat_v]
    valid_label = [fake_label_v, fake_label_v]

    history = pretrain_model.fit(train_input, train_label,
                                 batch_size=batch_size,
                                 epochs=epochs, 
                                 validation_data=(valid_input, valid_label), 
                                 shuffle='true', callbacks=[checkpointer])

    average_time_per_epoch = (time.time() - start_time) / epochs

    results.append((history, average_time_per_epoch))
    pretrain_model.save(modelname)
    # test to know the accuracy of pretraining
    batch_size = 1024
    model_test = modelArch.baseline(config['pretrain'], num_words, 
                                    embedding_matrix)
    model_test.load_weights(modelname, by_name=True)
    [loss1, rec1] = model_test.predict(valid_input,  batch_size=batch_size)
    print('predict res: loss:{} recall@1:{}'.format(np.mean(loss1), 
                                                    np.mean(rec1)))
    return pretrain_model

# stage 1 of the proposed network training
def trainstage1(config, train_data, valid_data, num_words, embedding_matrix):
    
    batchsize = config['stage1']['batchsize']
    epochs = config['stage1']['epochs']
    MAX_SEQUENCE_LENGTH = config['stage1']['MAX_SEQUENCE_LENGTH']
    img_fea_dim = config['stage1']['img_fea_dim']
    modelname = (config['savemodel'] + 'stage1_' +  
                 config['date'] + '.h5')

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
    x_val = np.array(valid_data[0][0:valid_num])
    y_val = np.array(valid_data[1][0:valid_num])
    
    # load model architecture
    SentenceEncoder = modelArch.baseline(config['stage1'], num_words, 
                                         embedding_matrix)
    try:
        SentenceEncoder.load_weights((config['savemodel'] + 
                                      'stage1_pretrain_' + 
                                      config['date'] + '.h5'), 
                                     by_name=True)
        print('stage 1 pretrained model loaded..')
    except FileNotFoundError:
        # continue training even without pretrained model
        print('stage 1 pretrained model does not exist. check!!! continuing..')
    
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
        val_input = [x_val, y_val]
        filepath = (config['savemodel'] + "tmp/stage1_vist_weights-" + 
                    "{03d}".format(i) + ".h5")
        checkpointer = keras.callbacks.ModelCheckpoint(filepath, 
                                                       monitor='val_loss', 
                                                       verbose=0,
                                                       save_best_only=False, 
                                                       save_weights_only=False, 
                                                       mode='auto', 
                                                       period=1)
        history = SentenceEncoder.fit(train_input, train_label,
                                      validation_data = (val_input, 
                                                         valid_label), 
                                      batch_size=batchsize, 
                                      verbose=1, 
                                      callbacks=[checkpointer])
        print("epoch: {} completed".format(i))

    average_time_per_epoch = (time.time() - start_time) / epochs
    print("avg time per epoch: {}".format(average_time_per_epoch))
    results.append((history, average_time_per_epoch))
    SentenceEncoder.save(modelname)
    return SentenceEncoder

# stage 2 of the proposed network training
def trainstage2(config, train_data, valid_data, num_words, embedding_matrix, 
                modeltype='cnsi'):

    traindir = config['datadir'] + 'train/'
    valdir = config['datadir'] + 'val/'
    epochs = config['stage2']['epochs']
    BS = config['stage2']['batchsize']
    cohfeatdim = config['stage2']['cohfeat_dim']
    modelname = (config['savemodel'] + 'stage2_' + modeltype + '_' +  
                 config['date'] + '.h5')

    if modeltype == 'cnsi':
        coh_sent_train = np.expand_dims(np.load(traindir + 
                                                'cohvec_train.npy'), axis=1)
#        coh_sent_val = np.expand_dims(np.load(valdir + 
#                                                'cohvec_val.npy'), axis=1)
    
    # load train image IDs
    train_image = utils_vist.getImgIds(traindir + 'train_image.csv')

    # Load Story Encoder model architecture
    StoryEncoder = modelArch.stage2(config, num_words, embedding_matrix)
    StoryEncoder.summary()
    
    # get sentence vectors for training and validation data
    x_train, y_train, imgids = get_sent_img_feats_stage1(config, 
                                                         train_data, num_words,
                                                         embedding_matrix)
#    
#    x_val, y_val = get_sent_img_feats_stage1(config['stage1'], 
#                                             valid_data, num_words,
#                                             embedding_matrix)

    NumTrain = math.floor(np.size(x_train, 0)/BS) * BS
    x_train = np.array(x_train[0:NumTrain][:][:])
    y_train = np.array(y_train[0:NumTrain][:][:])
    
#    NumVal = math.floor(np.size(x_val, 0)/BS) * BS
#    x_val = np.array(x_val[0:NumVal][:][:])
#    y_val = np.array(y_val[0:NumVal][:][:])
    
    start_time = time.time()
    results = []
    # Custom shuffle and train
    for i in range(0, epochs):
        tempIDS = np.array(list(train_image))
        tempIDS = tempIDS[0:NumTrain, :]
        if modeltype == 'cnsi':
#            tempvalcoh = coh_sent_val[0:NumVal, :]
#            tempvalcoh = np.repeat(tempvalcoh, 5, axis=1)
            tempCoh = coh_sent_train[0:NumTrain, :]
            coh_train = np.zeros((NumTrain, 5, cohfeatdim))
            tempCoh = np.repeat(tempCoh, 5, axis=1)

        iterations = math.floor(np.size(tempIDS, 0) / BS)

        p = np.zeros(np.size(tempIDS, 0), dtype=float)
        temptrain = np.zeros_like(tempIDS)
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
            if modeltype == 'cnsi':
                coh_train[BS * j:BS * (j + 1),
                          :, :] = tempCoh[sampled_arr, :, :]
            if j > (iterations - 100):
                sampled_arr = list(set(sampled_arr) - ToNotDel)
            tempIDS[sampled_arr, 0] = -1
            p[sampled_arr] = -1
        
        filepath = (config['savemodel'] + "tmp/stage2_vist_weights-" + 
                    "{03d}".format(i) + ".h5")
        checkpointer = keras.callbacks.ModelCheckpoint(filepath, 
                                                       monitor='val_loss', 
                                                       verbose=0,
                                                       save_best_only=False, 
                                                       save_weights_only=False, 
                                                       mode='auto', 
                                                       period=1)
        if modeltype == 'cnsi':
            history = StoryEncoder.fit([x_train1, coh_train], y_train1, 
#                                       validation_data = ([x_val, tempvalcoh],  
#                                                          y_val), 
                                       epochs=1,
                                       batch_size=BS, verbose=1, 
                                       callbacks=[checkpointer])
        else:
            history = StoryEncoder.fit(x_train1, y_train1, 
#                                       validation_data = (x_val, y_val), 
                                       epochs=1,
                                       batch_size=BS, verbose=1, 
                                       callbacks=[checkpointer])
        print("epoch: {} completed".format(i))
    
    average_time_per_epoch = (time.time() - start_time) / epochs
    print("avg time per epoch: {}".format(average_time_per_epoch))
    results.append((history, average_time_per_epoch))

    StoryEncoder.save(modelname)
    return StoryEncoder

# code to predict and then retrieve images for input stories vist testing data
def test(config, modelname, test_data, num_words, embedding_matrix, 
         modeltype='cnsi'):
    
    results = dict()
    testdir = config['datadir'] + 'test/'
    testinsamplename = config['testsamples']
    predictions = config['savepred']
    test_sents = utils_vist.getSent(testdir + 'test_text.csv')

    # get sentence and image vectors from stage 1
    print('obtaining sent and image vectors from stage 1...')
    x_test, y_test, id_test = get_sent_img_feats_stage1(config, test_data, 
                                                        num_words, 
                                                        embedding_matrix)

    test_lines = [line.rstrip('\n') for line in open(testinsamplename, 'r', 
                                                     encoding='utf-8')]
    if modeltype == 'cnsi':
        coh_sent_test = np.expand_dims(np.load(testdir + 'cohvec_test.npy'), 
                                       axis=1)
    # get input and gt ready
    print('gettin input and coherence vectors ready..')
    test_sent = []
    test_imgids = []
    test_stories = []
    for ind in test_lines:
        ind = int(ind)
        test_sent.append(x_test[ind])
        test_imgids.append(id_test[ind][:])
        test_stories.append(test_sents[ind])
    test_sent = np.array(test_sent)
    test_imgids = np.array(test_imgids)
    if modeltype == 'cnsi':
        coh_sent_test = coh_sent_test[np.array(test_lines).astype(int), :, :]
        coh_sent_test = np.repeat(coh_sent_test, 5, axis=1)

    # load model and predict
    print('predicting using stage 2...')
    trained_model = keras.models.load_model(
            modelname,
            custom_objects={'orderEmb_loss': modelArch.orderEmb_loss})
    if modeltype == 'cnsi':
        out_fea = trained_model.predict([test_sent, coh_sent_test])
    else:
        out_fea = trained_model.predict(test_sent)
        
    # save predictions
    with open(predictions, 'wb') as fp:
        pickle.dump(out_fea, fp)
    
    # retrieving images for input stories
    finalpreds = utils_vist.retrieve_images(np.array(y_test), out_fea, 
                                            np.array(id_test))
    
    # saving result dictionary
    
    results['input_stories'] = test_stories
    results['test_samples'] = test_lines
    results['test_gt_imageids'] = test_imgids
    results['test_pred_imageids'] = finalpreds
    
    pickle.dump(results, open('results_' + modeltype + '_' + config['date'] + 
                              '.pickle', 'wb'))
    return results


def main(config, process, modeltype='cnsi', model2test=None):
    
    print('model name..')
    print(modeltype)
    
       
    if process == 'pretrain':
        print('pretraining config..:')
        print (json.dumps(config['pretrain'], indent=2))
        pretrain(config, 'coco')
    
    print('loading data..')
    starttime = time.time()
    num_words, embedding_matrix, train_data, valid_data, test_data = loadData(
            config)
    print('data loaded in {} seconds'.format(time.time() - starttime))
    
    if process == 'trainstage1':
        print('stage 1 model parameters..:')
        print (json.dumps(config['stage1'], indent=2))
        trainstage1(config, train_data, valid_data, num_words, 
                    embedding_matrix)
        
    if process == 'trainstage2':
        print('stage 2 model parameters..:')
        print (json.dumps(config['stage2'], indent=2))
        trainstage2(config, train_data, valid_data, num_words, 
                    embedding_matrix, modeltype)
        
    if process == 'test':
        test(config, model2test, test_data, num_words, embedding_matrix, 
             modeltype)
    
    return True

if __name__ == '__main__':

    try:
        params = json.load(open('config.json'))
    except FileNotFoundError:
        params = create_config()
    
    main(params, 'train')