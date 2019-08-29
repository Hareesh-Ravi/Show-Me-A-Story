# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 12:16:15 2017

@author: HareeshRavi
"""

import json
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import configAll
import utils_vist
import time
import pickle

# load train, val and test data of stories, images and embedding matrix
def loadData(config):
    
    try:
        num_words = pickle.load(open(config['datadir'] + 
                                     'num_words.pickle', 'rb'))
        embedding_matrix = pickle.load(open(config['datadir'] + 
                                            'embedding_matrix.pickle', 'rb'))
        train_data = pickle.load(open(config['datadir'] + 
                                      'train_data.pickle', 'rb'))
        valid_data = pickle.load(open(config['datadir'] + 
                                      'valid_data.pickle', 'rb'))
        test_data = pickle.load(open(config['datadir'] + 
                                     'test_data.pickle', 'rb'))
        print('loaded existing data files')
    except:
        print('processed file(s) do not exist! Re-extracting all data')
        
        traindir = config['datadir'] + 'train/'
        testdir = config['datadir'] + 'test/'
        valdir = config['datadir'] + 'val/'
        glovetext = config['glovetext']
        
        MAX_SEQUENCE_LENGTH = config['stage1']['MAX_SEQUENCE_LENGTH']
        img_fea_dim = config['stage1']['img_fea_dim']
        EMBEDDING_DIM = config['stage1']['wd_embd_dim']
        MAX_NB_WORDS = config['stage1']['MAX_NB_WORDS']
        
        starttime = time.time()
        # load img feat files
        img_fea_train = json.loads(open(traindir + 
                                        'train_imgfeat.json').read())
        img_fea_valid = json.loads(open(valdir + 'val_imgfeat.json').read())
        img_fea_test = json.loads(open(testdir + 'test_imgfeat.json').read())
    
        # get img IDs
        train_imgID = utils_vist.getImgIds(traindir + 'train_image.csv')
        valid_imgID = utils_vist.getImgIds(valdir + 'val_image.csv')
        test_imgID = utils_vist.getImgIds(testdir + 'test_image.csv')
    
        # get stories
        train_sents = utils_vist.getSent(traindir + 'train_text.csv')
        valid_sents = utils_vist.getSent(valdir + 'val_text.csv')
        test_sents = utils_vist.getSent(testdir + 'test_text.csv')
        
        print('loaded all files in {} secs'.format(time.time() - starttime))
        # get word vectors from glove
        embeddings_index = {}
        f = open(glovetext, 'r', encoding='utf-8')
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
        train_sents, train_img_feats, trainids = utils_vist.flatten_all(
                train_imgID, img_fea_train, train_sents)
    
        valid_sents, valid_img_feats, valids = utils_vist.flatten_all(
                valid_imgID, img_fea_valid, valid_sents)
    
        test_sents, test_img_feats, testids = utils_vist.flatten_all(
                test_imgID, img_fea_test, test_sents)
        
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
        train_data = [train_sents, train_img_feats, trainids]
        
        # check some samples
        train_text = train_data[0]
        print(len(train_data[0]))
        train_imgs = train_data[1]
        print(np.shape(train_imgs[0]))
        
        # get val data
        valid_sents = data_sents[trainNum:(trainNum + validNum)]
        valid_img_feats = pad_sequences(valid_img_feats, img_fea_dim)
        valid_data = [valid_sents, valid_img_feats, valids]
    
        # get test data
        test_sents = data_sents[(trainNum + validNum):(trainNum + 
                                validNum + testNum)]
        test_img_feats = pad_sequences(test_img_feats, img_fea_dim)
        test_data = [test_sents, test_img_feats, testids]
    
        print('Preparing embedding matrix.')
        # prepare embedding matrix
        num_words = min(MAX_NB_WORDS, len(word_index) + 1)
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
        
        pickle.dump(num_words, open(config['datadir'] + 'num_words.pickle', 
                                    'wb', pickle.HIGHEST_PROTOCOL))
        pickle.dump(embedding_matrix, open(
                config['datadir'] + 'embedding_matrix.pickle', 
                'wb', pickle.HIGHEST_PROTOCOL))
        pickle.dump(train_data, open(config['datadir'] + 'train_data.pickle', 
                                     'wb', pickle.HIGHEST_PROTOCOL))
        pickle.dump(valid_data, open(config['datadir'] + 'valid_data.pickle', 
                                     'wb', pickle.HIGHEST_PROTOCOL))
        pickle.dump(test_data, open(config['datadir'] + 'test_data.pickle', 
                                     'wb', pickle.HIGHEST_PROTOCOL))
        
    return num_words, embedding_matrix, train_data, valid_data, test_data


if __name__ == '__main__':
    try:
        config = json.load(open('config.json'))
    except FileNotFoundError:
        config = configAll.create_config()
    loadData(config)
