import time
import json
import keras
import numpy as np
import math
from load_data import loadData
from configAll import create_config
import modelArch
import utils_vist
import pickle

#  make params global for use
def init(param):
    global batchsize
    batchsize = param['batch_size']
    global img_feat_dim
    img_feat_dim = param['img_fea_dim']
    global sent_feat_dim
    sent_feat_dim = param['sent_fea_dim']
    global word_embd_dim
    word_embd_dim = param['wd_embd_dim']
    global epochs
    epochs = param['epochs']  
    global MAX_SEQUENCE_LENGTH
    MAX_SEQUENCE_LENGTH = param['MAX_SEQUENCE_LENGTH']

# get sentence vectors using stage 1 of trained network
def get_sent_img_feats_baseline(config, data, num_words, embedding_matrix):

    # Load Sentence encoder and initialize it's weights
    model_path = (config['savemodel'] + 'baseline_' + 
                  config['date'] + '.h5')
    print('model:{}'.format(model_path))
    baselinemodel = modelArch.stage1(config, num_words, embedding_matrix)
    baselinemodel.load_weights(model_path, by_name=True)

    # Extract Sent and Img features using sent encoder
    [encoded_sents, encoded_imgs] = baselinemodel.predict([data[0], data[1]], 
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

# train baseline on VIST dataset
def train(params, num_words, embedding_matrix, train_data, valid_data):   

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
    model_base = modelArch.baseline(num_words, embedding_matrix)
    model_base.compile(loss=['mean_absolute_error', utils_vist.MyCustomLoss], 
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
        ytrain = np.zeros((batchsize*iterations, img_feat_dim), dtype=float)
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
    model_base.save(params['savemodel'] + 'baseline_' + 
                    params['date'] + '.h5')
    return model_base
 
def test(config, num_words, embedding_matrix, test_data):
    
    # load all data  
    results = dict()
    testdir = config['datadir'] + 'test/'
    testinsamplename = config['testsamples']
    predictions = config['savepred']
    test_sents = utils_vist.getSent(testdir + 'test_text.csv')

    # get sentence and image vectors from stage 1
    print('obtaining sent and image vectors from stage 1...')
    x_test, y_test, id_test = get_sent_img_feats_baseline(config, test_data, 
                                                          num_words, 
                                                          embedding_matrix)

    test_lines = [line.rstrip('\n') for line in open(testinsamplename, 'r', 
                                                     encoding='utf-8')]
#    test_batch = len(test_data[0])
#    model_test = modelArch.baseline(num_words, embedding_matrix)
#    model_test.load_weights('baseline_' + params['date'] + 
#                            '.h5', by_name=True)
#    [loss1, rec1] = model_test.predict(test_data,  batch_size=test_batch)
#    print('predict res: loss:{} recall@1:{}'.format(np.mean(loss1), 
#                                                    np.mean(rec1)))
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
    
    # save predictions
    with open(predictions, 'wb') as fp:
        pickle.dump(test_sent, fp)
        
    # retrieving images for input stories
    finalpreds = utils_vist.retrieve_images(np.array(y_test), 
                                            np.array(test_sent), 
                                            np.array(id_test))
    
    # saving result dictionary
    
    results['input_stories'] = test_stories
    results['test_samples'] = test_lines
    results['test_gt_imageids'] = test_imgids
    results['test_pred_imageids'] = finalpreds
    
    pickle.dump(results, open('results_baseline_' + config['date'] + 
                              '.pickle', 'wb'))
    return results

def show():
    
    return True

def main(config, process):
    
    # load all data
    print('Loading data')
    num_words, embedding_matrix, train_data, valid_data, test_data = loadData(
            config)
    
    assert config['model'] == 'baseline'
    print('config:')
    print (json.dumps(config, indent=2))

    init(config['stage1'])
    if process == 'train':
        train(config, num_words, embedding_matrix, train_data, valid_data)
    if process == 'test':
        test(config, num_words, embedding_matrix, test_data)
    if process == 'show':
        show(config)
        
    return True


if __name__ == "__main__":
    
    try:
        params = json.load(open('config.json'))
    except FileNotFoundError:
        params = create_config()
        
    main(params)
    
