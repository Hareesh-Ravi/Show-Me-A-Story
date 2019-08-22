# -*- coding: utf-8 -*-
"""
Created on Wed Aug 7 17:25:25 2018

@author: HareeshRavi
"""

import time
import json

def create_config():
    config = dict()

    config['DEFAULT'] = dict()
    config['DEFAULT']['train'] = True
    config['DEFAULT']['test'] = False
    config['DEFAULT']['CreatedOn'] = time.strftime("%y-%m-%d")

    # Generalt data and path config
    config['DATALOADER'] = dict()
    config['DATALOADER']['trainImgId'] = './data/train/train_image.csv'
    config['DATALOADER']['testImgId'] = './data/test/test_image.csv'
    config['DATALOADER']['valImgId'] = './data/train/val_image.csv'
    config['DATALOADER']['trainImgFeat'] = ('./data/train/'
                                            + 'VIST_ImageFeattrain.json')
    config['DATALOADER']['testImgFeat'] = './data/test/VIST_ImageFeattest.json'
    config['DATALOADER']['valImgFeat'] = ('./data/train/'
                                          + 'VIST_ImageFeatval.json')
    config['DATALOADER']['trainTextSeq'] = ('./data/train/'
                                            + 'VISTTextTrainData_Seq.csv')
    config['DATALOADER']['testTextSeq'] = ('./data/test/'
                                           + 'VISTTextTestData_Seq.csv')
    config['DATALOADER']['valTextSeq'] = ('./data/train/'
                                          + 'VISTTextvalData_Seq.csv')
    config['DATALOADER']['CohVecTrain'] = './data/train/sent_coh_train.npy'
    config['DATALOADER']['CohVecTest'] = './data/test/sent_coh_test.npy'
    config['DATALOADER']['glovetext'] = './data/glove.6B.300d.txt'
    config['DATALOADER']['testSamplesFileName'] = ('./data/testing/'
                                                   + 'test200_row_id.txt')
    # baseline
    config['MODEL_Sent_Img_PARAMS'] = dict()
    config['MODEL_Sent_Img_PARAMS']['MAX_SEQUENCE_LENGTH'] = 100
    config['MODEL_Sent_Img_PARAMS']['img_fea_dim'] = 4096
    config['MODEL_Sent_Img_PARAMS']['MAX_NB_WORDS'] = 30000
    config['MODEL_Sent_Img_PARAMS']['wd_embd_dim'] = 300
    config['MODEL_Sent_Img_PARAMS']['sent_fea_dim'] = 1024

    # NSI model (without coherence)
    config['MODEL_Story_ImgSeq_NSI_PARAMS'] = dict()
    config['MODEL_Story_ImgSeq_NSI_PARAMS']['epochs'] = 150
    config['MODEL_Story_ImgSeq_NSI_PARAMS']['batchsize'] = 32
    config['MODEL_Story_ImgSeq_NSI_PARAMS']['hidden_size1'] = 512
    config['MODEL_Story_ImgSeq_NSI_PARAMS']['hidden_size2'] = 512
    config['MODEL_Story_ImgSeq_NSI_PARAMS']['hidden_size3'] = 1024
    config['MODEL_Story_ImgSeq_NSI_PARAMS']['y_dim'] = 1024
    config['MODEL_Story_ImgSeq_NSI_PARAMS']['x_len'] = 5
    config['MODEL_Story_ImgSeq_NSI_PARAMS']['y_len'] = 5
    config['MODEL_Story_ImgSeq_NSI_PARAMS']['x_dim'] = 1024
    config['MODEL_Story_ImgSeq_NSI_PARAMS']['alpha'] = 0.05
    config['MODEL_Story_ImgSeq_NSI_PARAMS']['learningrate'] = 0.001
    
    # CNSI model (with coherence)
    config['MODEL_Story_ImgSeq_CNSI_PARAMS'] = dict()
    config['MODEL_Story_ImgSeq_CNSI_PARAMS']['epochs'] = 150
    config['MODEL_Story_ImgSeq_CNSI_PARAMS']['batchsize'] = 32
    config['MODEL_Story_ImgSeq_CNSI_PARAMS']['hidden_size1'] = 512
    config['MODEL_Story_ImgSeq_CNSI_PARAMS']['hidden_size2'] = 512
    config['MODEL_Story_ImgSeq_CNSI_PARAMS']['hidden_size3'] = 1024
    config['MODEL_Story_ImgSeq_CNSI_PARAMS']['y_dim'] = 1024
    config['MODEL_Story_ImgSeq_CNSI_PARAMS']['x_len'] = 5
    config['MODEL_Story_ImgSeq_CNSI_PARAMS']['y_len'] = 5
    config['MODEL_Story_ImgSeq_CNSI_PARAMS']['x_dim'] = 1024
    config['MODEL_Story_ImgSeq_CNSI_PARAMS']['alpha'] = 0.05
    config['MODEL_Story_ImgSeq_CNSI_PARAMS']['learningrate'] = 0.001
    config['MODEL_Story_ImgSeq_CNSI_PARAMS']['cohfeat_dim'] = 64
    
    config['OUTPUTFILENAMES'] = {}
    config['FILENAMES']['testImgOutFeatSaveName'] = ('./data/testing/'
                                                     + 'test200_sents_fea_lstm.pickle')
    config['FILENAMES']['NSImodelname'] = ('./TrainedModels/'
                                           + time.strftime("%y-%m-%d")
                                           + '_nsi.h5')
    config['FILENAMES']['CNSImodelname'] = ('./TrainedModels/'
                                            + time.strftime("%y-%m-%d")
                                            + '_cnsi.h5')
    config['FILENAMES']['baseline'] = ('./TrainedModels/'
                                            + time.strftime("%y-%m-%d")
                                            + '_baseline.h5')

    with open('config' + config['DEFAULT']['CreatedOn'] + '.json', 
              'w') as configfile:
        json.dump(configfile, config)
    
    return config


if __name__ == '__main__':
    create_config()

# del config
# %%
# config = configparser.ConfigParser()
# config.read('simulator_config.ini')