# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 12:16:15 2017

@author: HareeshRavi
"""

import json
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import config_all
import utils_vist

# get list of all sentences and corresponding image features 
def get_sents_imgs(imgID_set, imgFea_set, sents_set):
    sents = []
    imgs = []
    for i, imgIDs in enumerate(imgID_set):
        for ind in range(5):
            imgID = str(imgIDs[ind])
            img_fea = imgFea_set[imgID]
            sents.append(sents_set[i][ind])
            imgs.append(img_fea)
    return sents, imgs


# load train, val and test data of stories, images and embedding matrix
def loadData(config):
    
    dataconfig = config['DATALOADER']
    
    trainimgidfile = dataconfig['trainImgId']
    testimgidfile = dataconfig['testImgId']
    valimgidfile = dataconfig['valImgId']
    trainimgFeatfile = dataconfig['trainImgFeat']
    testimgFeatfile = dataconfig['testImgFeat']
    valimgFeatfile = dataconfig['valImgFeat']
    trainTextSeqfile = dataconfig['trainTextSeq']
    testTextSeqfile = dataconfig['testTextSeq']
    valTextSeqfile = dataconfig['valTextSeq']
    glovetext = dataconfig['glovetext']
    
    MAX_SEQUENCE_LENGTH = config['MODEL_Sent_Img_PARAMS'][
            'MAX_SEQUENCE_LENGTH']
    img_fea_dim = config['MODEL_Sent_Img_PARAMS']['img_fea_dim']
    EMBEDDING_DIM = config['MODEL_Sent_Img_PARAMS']['wd_embd_dim']
    MAX_NB_WORDS = config['MODEL_Sent_Img_PARAMS']['MAX_NB_WORDS']

    # load img feat files
    img_fea_train = json.loads(open(trainimgFeatfile).read())
    img_fea_valid = json.loads(open(valimgFeatfile).read())
    img_fea_test = json.loads(open(testimgFeatfile).read())

    # get img IDs
    train_imgID = utils_vist.getImgIds(trainimgidfile)
    valid_imgID = utils_vist.getImgIds(valimgidfile)
    test_imgID = utils_vist.getImgIds(testimgidfile)

    # get stories
    train_sents = utils_vist.getSent(trainTextSeqfile)
    valid_sents = utils_vist.getSent(valTextSeqfile)
    test_sents = utils_vist.getSent(testTextSeqfile)
    
    print('loaded all files..')
    # get word vectors from glove
    embeddings_index = {}
    f = open(glovetext)
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Indexed word vectors.')

    # get num of samples
    trainNum = len(train_imgID)*5
    validNum = len(valid_imgID)*5
    testNum = len(test_imgID)*5

    # get image features and text sentences in a single list
    train_sents, train_img_feats = get_sents_imgs(train_imgID,
                                                  img_fea_train, train_sents)

    valid_sents, valid_img_feats = get_sents_imgs(valid_imgID, img_fea_valid,
                                                  valid_sents)

    test_sents, test_img_feats = get_sents_imgs(test_imgID, img_fea_test,
                                                test_sents)
    
    # get all text in single list to process them together
    sents = train_sents + valid_sents + test_sents
    
    # tokenize and convert each sentence to sequences
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(sents)
    sequences = tokenizer.texts_to_sequences(sents)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    data_sents = pad_sequences(sequences, MAX_SEQUENCE_LENGTH)
    
    # get train data
    train_sents = data_sents[0:trainNum]
    train_img_feats = pad_sequences(train_img_feats, img_fea_dim)
    train_data = [train_sents, train_img_feats]
    
    # check some samples
    train_text = train_data[0]
    print(len(train_data[0]))
    train_imgs = train_data[1]
    print(np.shape(train_imgs[0]))
    
    # get val data
    valid_sents = data_sents[trainNum:(trainNum+validNum)]
    valid_img_feats = pad_sequences(valid_img_feats, img_fea_dim)
    valid_data = [valid_sents, valid_img_feats]

    # get test data
    test_sents = data_sents[(trainNum+validNum):(trainNum+validNum+testNum)]
    test_img_feats = pad_sequences(test_img_feats, img_fea_dim)
    test_data = [test_sents, test_img_feats]

    print('Preparing embedding matrix.')
    # prepare embedding matrix
    num_words = min(MAX_NB_WORDS, len(word_index)+1)
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i >= MAX_NB_WORDS:
            print('{}: {}'.format(word, i))
            continue
        if i == 0:
            print('{}: {}'.format(word, i))

        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    return num_words, embedding_matrix, train_data, valid_data, test_data


if __name__ == '__main__':
    try:
        config = json.load(open('config.json'))
    except FileNotFoundError:
        config = config_all.create_config()
    loadData(config)
