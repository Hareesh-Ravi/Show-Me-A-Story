import time
import json
import keras
import numpy as np
import math
from load_VIST_data import loadData
from config_all import create_config
import model

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

def train(params): 
    
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
                    params['general']['date'] + '.h5')
    return model_base
 
def test(params):
    
    # load all data
    print('Loading data')
    num_words, embedding_matrix, train_data, valid_data, test_data = loadData()
    
    test_batch = len(test_data[0])
    model_test = model.baseline(num_words, embedding_matrix)
    model_test.load_weights('baseline_' + params['general']['date'] + 
                            '.h5', by_name=True)
    [loss1, rec1] = model_test.predict(test_data,  batch_size=test_batch)
    print('predict res: loss:{} recall@1:{}'.format(np.mean(loss1), 
                                                    np.mean(rec1)))
    
    return True

def evaluate(params):
    
    
    return True


if __name__ == "__main__":
    
    try:
        params = json.load(open('config.json'))
    except FileNotFoundError:
        params = create_config()
    assert params['model'] == 'baseline'
    print('general config:')
    print (json.dumps(params['general'], indent=2))
    print('model parameters:')
    print (json.dumps(params['stage1'], indent=2))
    init(params['stage1'])
    if params['general']['train']:
        train(params)
    if params['general']['test']:
        test(params)
    if params['general']['eval']:
        evaluate(params)
    
