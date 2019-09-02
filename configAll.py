# -*- coding: utf-8 -*-
"""
Created on Wed Aug 7 17:25:25 2018

@author: HareeshRavi
"""

import time
import json

def create_config():
    config = dict()
    
    # model can be 'nsi', 'cnsi' or 'baseline'
    config['model'] = 'cnsi'
    # replace the below date to match trained models for evaluation
    config['date'] = time.strftime("%Y-%m-%d")
    config['datadir'] = './data/'

    # filenames
    config['glovetext'] = './data/glove.6B.300d.txt'
    config['testsamples'] = ('./data/test_samples.txt')
    config['savepred'] = ('./data/test_samples_prediction.pickle')
    config['savemodel'] = './TrainedModels/'
    
    # Pretraining stage 1 (or baseline) parameters
    config['pretrain'] = dict()
    config['pretrain']['MAX_SEQUENCE_LENGTH'] = 100
    config['pretrain']['img_fea_dim'] = 4096
    config['pretrain']['MAX_NB_WORDS'] = 26000
    config['pretrain']['wd_embd_dim'] = 300
    config['pretrain']['sent_fea_dim'] = 1024
    config['pretrain']['batchsize'] = 128
    config['pretrain']['epochs'] = 15
    config['pretrain']['word_count_threshold'] = 1
    
    # stage 1 (or baseline) parameters
    config['stage1'] = dict()
    config['stage1']['MAX_SEQUENCE_LENGTH'] = 100
    config['stage1']['img_fea_dim'] = 4096
    config['stage1']['MAX_NB_WORDS'] = 26000
    config['stage1']['wd_embd_dim'] = 300
    config['stage1']['sent_fea_dim'] = 1024
    config['stage1']['batchsize'] = 128
    config['stage1']['epochs'] = 20
    
    # stage 2 parameters
    config['stage2'] = dict()
    config['stage2']['epochs'] = 150
    config['stage2']['batchsize'] = 32
    config['stage2']['hidden_size1'] = 512
    config['stage2']['hidden_size2'] = 512
    config['stage2']['hidden_size3'] = 1024
    config['stage2']['y_dim'] = 1024
    config['stage2']['x_len'] = 5
    config['stage2']['y_len'] = 5
    config['stage2']['x_dim'] = 1024
    config['stage2']['alpha'] = 0.05
    config['stage2']['learningrate'] = 0.001
    # 'cohfeat_dim' can be None or 64 for NSI and CNSI models respectively
    config['stage2']['cohfeat_dim'] = 64
    
    with open('config.json', 'w') as configfile:
        json.dump(config, configfile)
    print('config created and saved...')
    return config


if __name__ == '__main__':
    create_config()
