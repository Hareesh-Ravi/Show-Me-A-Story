# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 17:09:01 2019

@author: HareeshRavi
"""
import config_all
import cnsi
import baseline
import process_vist
import vggfeat_vist
import get_cohvec_vist

if __name__ == '__main__':
    
    # get config
    configs = config_all.create_config()
    '''
    To preprocess
    '''
    # process vist data jsons and put it according to usage
    process_vist.main(configs)
    # extract vgg feats for all images. also remove images (and stories)
    # for where images are not present
    vggfeat_vist.main(configs)
    # get coherence vector for all stories
    get_cohvec_vist.main(configs)
    
    '''
    Get stage 1 ready for nsi or cnsi
    '''
    # to pretrain model on mscoco dataset
    cnsi.main(configs, 'pretrain')
    # to train model stage 1 on vist dataset
    cnsi.main(configs, 'trainstage1')
    
    '''
    To Train CNSI model
    '''
    # to train cnsi model stage 2 on vist dataset
    cnsi.main(configs, 'trainstage2', 'cnsi')
    
    '''
    To Train NSI model
    '''
    # to train nsi model stage 2 on vist dataset
    cnsi.main(configs, 'trainstage2', 'nsi')
    
    '''
    To Train baseline model
    '''
    configs['model'] = 'baseline'
    # to train baseline on vist dataset
    baseline.main(configs, 'train')
    
    '''
    To evaluate 'model' on VIST test set
    '''
    # get predictions for stories from testsamples for cnsi model
    model2test =  configs['savemodel'] + 'stage2_cnsi_' + configs['date'] + '.h5'
    cnsi.main(configs, 'test', 'cnsi', model2test)
    
    # get predictions for stories from testsamples for nsi model
    model2test =  configs['savemodel'] + 'stage2_nsi_' + configs['date'] + '.h5'
    cnsi.main(configs, 'test', 'nsi', model2test)
    
    # get predictions for stories from testsamples for baseline model
    baseline.main(configs, 'test')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    