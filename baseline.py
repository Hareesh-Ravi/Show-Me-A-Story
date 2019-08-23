import time
import json
from keras.layers import Input,Embedding, Dense
import keras
from keras.models import Model
import keras.backend as K
import tensorflow as tf
import numpy as np
import math
from load_VIST_data import loadData
from config_all import create_config

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
    

def order_violations(s, im):
    
    # Computes the order violations (Equation 2 in the paper)
    return K.pow(K.maximum(0.0, s - im), 2)

def contrastive_loss(s_im):
    
    margin = 0.005
    
    # For a minibatch of sentence and image embeddings, 
    # compute the pairwise contrastive loss
    s = s_im[0]
    im = s_im[1]

    # create two tensor 1xnumxdim  numx1xdim
    s2 = K.expand_dims(s,1)
    im2 = K.expand_dims(im,0)

    errors = K.sum(K.pow(K.maximum(0.0, s2 - im2), 2), axis=2)
    	
    diagonal = tf.diag_part(errors)
    
    # all constrastive image for each sentence
    cost_s = K.maximum(0.0, margin - errors + diagonal) 
    
    # all contrastive sentences for each image
    cost_im = K.maximum(0.0, margin - errors + K.reshape(diagonal,[-1, 1]))  

    cost_tot = cost_s + cost_im

    cost_tot = tf.matrix_set_diag(cost_tot, tf.zeros(batchsize))

    return K.sum(cost_tot)


def retriv_acc(s_im):
    
    # For a minibatch of sentence and image embeddings, 
    # compute the retrieval accuracy
    s = s_im[0]
    im = s_im[1]
    print(np.shape(s),np.shape(im))
    s2 = K.expand_dims(s, 1)
    im2 = K.expand_dims(im, 0)
    errors = K.sum(order_violations(s2, im2), axis=2)


    inds = K.argmin(errors,axis=1)
    inds = tf.cast(inds,tf.int32)
    inds_true = tf.range(batchsize)
    elements_equal_to_value = tf.equal(inds, inds_true)
    as_ints = tf.cast(elements_equal_to_value, tf.int32)
    results = tf.reduce_sum(as_ints)
    results = tf.cast(results,tf.float32)

    return results

def edis_outputshape(input_shape):
    shape = list(input_shape)
    assert len(shape)==2
    outshape = (shape[0][0],1)
    return tuple(outshape)

def MyCustomLoss(yTure, yPred):
    return yPred


def net_arch(num_words, embedding_matrix):
    
    embedding_layer = Embedding(num_words,
                                word_embd_dim,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)

    input_sent = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32', 
                       name='input1')
    input_img = Input(shape=(img_feat_dim,), dtype='float32', name='input2')

    # Encode sentence
    # This embedding layer will encode the input sequence
    # into a sequence of dense 512-dimensional vectors.
    x1 = embedding_layer(input_sent)

    # using GRU instead of LSTM
    Encode_sent = keras.layers.recurrent.GRU(sent_feat_dim, name='gru')(x1)
    
    Encode_sent_normed = keras.layers.Lambda(lambda x: K.abs(K.l2_normalize(
            x, axis=1)), name='sentFeaNorm')(Encode_sent)
    
    # encoding image feat
    Encode_img = Dense(sent_feat_dim, activation='linear', 
                       name='imgEncode')(input_img)
    Encode_img_normed = keras.layers.Lambda(lambda x: K.abs(K.l2_normalize(
            x, axis=1)), name='imgFeaNorm')(Encode_img)
    
    # define a Lambda merge layer
    main_output = keras.layers.merge([Encode_sent_normed, Encode_img_normed], 
                                     mode=contrastive_loss,
                                     output_shape=edis_outputshape, 
                                     name='orderEmbd')

    acc_output = keras.layers.merge([Encode_sent_normed, Encode_img_normed],
                                    mode=retriv_acc,
                                    output_shape=edis_outputshape, 
                                    name='Recall_1')

    model = Model(inputs=[input_sent, input_img], outputs=[main_output, 
                  acc_output])
    return model

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
    model = net_arch(num_words, embedding_matrix)
    model.compile(loss=['mean_absolute_error', MyCustomLoss], optimizer='adam', 
                  loss_weights=[1,0])
    model.summary()

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
        xtrain = np.zeros((batchsize*iterations, 100), dtype=float)
        ytrain = np.zeros((batchsize*iterations, 4096), dtype=float)
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
            if j>(iterations-100):
                sampled_arr = list(set(sampled_arr)-ToNotDel)
            
            tempIDS[sampled_arr] = -1
            textIDS[sampled_arr] = -1
            p[sampled_arr] = -1
            
        train_input = [xtrain, ytrain]
        history = model.fit(train_input, train_label,
                            validation_data = (valid_data, valid_label), 
                            batch_size=batchsize, 
                            verbose=1, 
                            callbacks=[checkpointer])

    average_time_per_epoch = (time.time() - start_time) / epochs

    results.append((history, average_time_per_epoch))
    model.save('baseline_' + params['general']['date'] + '.h5')
    return model
 
def test(params):
    
    # load all data
    print('Loading data')
    num_words, embedding_matrix, train_data, valid_data, test_data = loadData()
    
    test_batch = len(test_data[0])
    model_test = net_arch(num_words, embedding_matrix)
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
    
