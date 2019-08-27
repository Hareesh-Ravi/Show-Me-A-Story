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
import argparse


if __name__ == '__main__':
    
    #argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--preprocess', action='store_true', default=False, 
                        help='use this argument to preprocess VIST data')
    parser.add_argument('--pretrain', action='store_true', default=False, 
                        help='use this to pre-trainstage 1 of the network')
    parser.add_argument('--train', type=str, default='stage1', 
                        help ='train stage1, cnsi, nsi or baseline')
    parser.add_argument('--eval', type=str, default='stage1', 
                        help ='evaluate cnsi, nsi or baseline')
    parser.add_argument('--show', type=str, default='stage1', 
                        help ='show the story for cnsi, nsi or baseline')
    args = parser.parse_args()
    
    # get config
    configs = config_all.create_config()
    '''
    To preprocess
    '''
    if args.preprocess:
        # process vist data jsons and put it according to usage
        process_vist.main(configs)
        # extract vgg feats for all images. also remove images (and stories)
        # for where images are not present
        vggfeat_vist.main(configs)
        # get coherence vector for all stories
        get_cohvec_vist.main(configs)
    
    
    if args.pretrain:
        '''
        Pretrain stage 1 on MSCOCO dataset
        '''
        cnsi.main(configs, 'pretrain')
        
    if args.train == 'stage1':
        '''
        train stage 1 on VIST dataset
        '''
        cnsi.main(configs, 'trainstage1')
        
    elif args.train == 'cnsi':
        '''
        To Train CNSI model stage 2
        '''
        cnsi.main(configs, 'trainstage2', 'cnsi')
        
    elif args.train == 'nsi':
        '''
        To Train NSI model stage 2
        '''
        cnsi.main(configs, 'trainstage2', 'nsi')
        
    elif args.train == 'baseline':
        '''
        To Train baseline model
        '''
        configs['model'] = 'baseline'
        baseline.main(configs, 'train')
    else:
        raise ValueError('args for train can be stage1, cnsi, ' + 
                         'nsi or baseline only')
    
    '''
    To evaluate 'model' on VIST test set. Thiw will save predictions in file
    for further use by metrics. Will not print or show any results.
    '''
    if args.eval == 'cnsi':
        # get predictions for stories from testsamples for cnsi model
        model2test =  (configs['savemodel'] + 'stage2_cnsi_' + 
                       configs['date'] + '.h5')
        cnsi.main(configs, 'test', 'cnsi', model2test)
    elif args.eval == 'nsi':
        # get predictions for stories from testsamples for nsi model
        model2test =  (configs['savemodel'] + 'stage2_nsi_' + 
                       configs['date'] + '.h5')
        cnsi.main(configs, 'test', 'nsi', model2test)
    elif args.eval == 'baseline':
        # get predictions for stories from testsamples for baseline model
        baseline.main(configs, 'test')
    
    
    '''
    To evaluate 'model' on VIST test set. Thiw will save predictions in file
    for further use by metrics. Will not print or show any results.
    '''
    if args.show == 'cnsi':
        # get predictions for stories from testsamples for cnsi model
        model2test =  (configs['savemodel'] + 'stage2_cnsi_' + 
                       configs['date'] + '.h5')
        cnsi.main(configs, 'test', 'cnsi', model2test)
    elif args.show == 'nsi':
        # get predictions for stories from testsamples for nsi model
        model2test =  (configs['savemodel'] + 'stage2_nsi_' + 
                       configs['date'] + '.h5')
        cnsi.main(configs, 'test', 'nsi', model2test)
    elif args.show == 'baseline':
        # get predictions for stories from testsamples for baseline model
        baseline.main(configs, 'test')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    