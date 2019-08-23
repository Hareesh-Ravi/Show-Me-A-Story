import time
import argparse
import json
from keras.layers import Input,Embedding, Dense
import keras
from keras.models import Model
import keras.backend as K
import tensorflow as tf
import numpy as np
import math
from load_VIST_data import loadData
import csv

#config setting
MAX_SEQUENCE_LENGTH = 100
img_feat_dim = 4096
wd_embd_dim = 300
embedding_dim =1024
batch_size = 1
epochs = 15
modes = [0, 1, 2] # lstm mode
dataset = 'coco'
word_count_threshold = 5
margin = 0.005

def csvnumread(filename):
    out_list = []
    with open(filename, "r") as readtextfile:
        reader = csv.reader(readtextfile)
        for row in reader:
            if (len(row)>1):
                out_list.append([int(i) for i in row])
    return out_list

def init(param):
    global batch_size
    batch_size = 128
    global embedding_dim
    embedding_dim = param['image_encoding_size']
    global word_embd_dim
    word_embd_dim = param['word_encoding_size']
    global epochs
    epochs = param['max_epochs']    
    

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
    	
    diagonal = tf.diag_part(errors) # column vector
    # all constrastive image for each sentence
    cost_s = K.maximum(0.0, margin - errors + diagonal) 
    # all contrastive sentences for each image
    cost_im = K.maximum(0.0, margin - errors + K.reshape(diagonal,[-1, 1]))  

    cost_tot = cost_s + cost_im

    cost_tot = tf.matrix_set_diag(cost_tot, tf.zeros(batch_size))

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
    inds_true = tf.range(batch_size)
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
                                wd_embd_dim,
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
    Encode_sent = keras.layers.recurrent.GRU(embedding_dim, name='gru')(x1)
    
    Encode_sent_normed = keras.layers.Lambda(lambda x: K.abs(K.l2_normalize(
            x, axis=1)), name='sentFeaNorm')(Encode_sent)
    
    # encoding image feat
    Encode_img = Dense(embedding_dim, activation='linear', 
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

def main(params): 
    print('Loading data')

    num_words, embedding_matrix, train_data, valid_data, test_data = loadData()
    print(np.shape(train_data[0]),np.shape(valid_data[0]))
    train_num = math.floor(np.size(train_data[0],0)/batch_size) * batch_size
    valid_num = math.floor(np.size(valid_data[0],0)/batch_size) * batch_size

    print(train_num, valid_num)

    fake_label = np.zeros(train_num)
    train_label = [fake_label, fake_label]
    valid_data[0] = valid_data[0][0:valid_num]
    valid_data[1] = valid_data[1][0:valid_num]
    fake_label_v = np.zeros(valid_num)
    valid_label = [fake_label_v, fake_label_v]

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
    
    # Perform training by manually shuffling to remove duplicates
    sent_data = train_data[0]
    img_data = train_data[1]

    train_image = csvnumread('./data/VIST/train_image.csv')
    train_allImage = []
    for i in train_image:
        for j in i:
            train_allImage.append(j)   
    epochs = 20
    train_allImage = np.array(train_allImage)

    #text_data_idx = np.arange(200704)
    iterations = math.floor(np.size(img_data, 0)/batch_size)
    
    sent_data = sent_data[0:train_num, :]
    img_data = img_data[0:train_num, :]
    BS = 128
    train_allImage = train_allImage[0:train_num]
    for i in range(0,epochs):
        xtrain = np.zeros((BS*iterations, 100), dtype=float)
        ytrain = np.zeros((BS*iterations, 4096), dtype=float)
        tempIDS = train_allImage
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
                sampled_arr = list(np.random.choice(np.size(tempIDS, 0), BS, 
                                                    replace=False, p=p))
            except ValueError:
                p[:] = 1/np.size(p,0)
                sampled_arr = list(np.random.choice(np.size(tempIDS, 0), BS, 
                                                    replace=False, p=p))
            
            ytrain[BS*j:BS*(j+1), :] = img_data[sampled_arr, :]
            xtrain[BS*j:BS*(j+1), :] = sent_data[sampled_arr, :]            
            if j>(iterations-100):
                sampled_arr = list(set(sampled_arr)-ToNotDel)
            
            tempIDS[sampled_arr] = -1
            textIDS[sampled_arr] = -1
            p[sampled_arr] = -1
            
        train_input = [xtrain, ytrain]
        history = model.fit(train_input, train_label,
                            validation_data = (valid_data, valid_label), 
                            batch_size=128, 
                            verbose=1, 
                            callbacks=[checkpointer])

    average_time_per_epoch = (time.time() - start_time) / epochs

    results.append((history, average_time_per_epoch))
    model.save('Stage1_VIST_Aug24.h5')
    batch_size = len(test_data[0])
    model_test = net_arch(num_words, embedding_matrix)
    model_test.load_weights('Stage1_VIST_Aug24.h5',by_name=True)
    [loss1, rec1] = model_test.predict(test_data,  batch_size=batch_size)
    print('predict res: loss:{} recall@1:{}'.format(np.mean(loss1), 
                                                    np.mean(rec1)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # global setup settings, and checkpoints
    parser.add_argument('-d', '--dataset', dest='dataset', default='coco', 
                        help='dataset: flickr8k/flickr30k')
    parser.add_argument('--fappend', dest='fappend', type=str, 
                        default='baseline', 
                        help='append this string to checkpoint filenames')
    parser.add_argument('-o', '--checkpoint_output_directory', 
                        dest='checkpoint_output_directory', type=str,
                        default='cv/', 
                        help='output directory to write checkpoints to')
    parser.add_argument('--worker_status_output_directory', 
                        dest='worker_status_output_directory', 
                        type=str, default='status/', 
                        help='directory to write worker status JSON blobs to')
    parser.add_argument('--write_checkpoint_ppl_threshold', 
                        dest='write_checkpoint_ppl_threshold', type=float,
                        default=-1, 
                        help='ppl thresh above which we dont write a chckpnt')
    parser.add_argument('--init_model_from', dest='init_model_from', type=str, 
                        default='',
                        help='init the model param from specific checkpoint?')

    # model parameters
    parser.add_argument('--image_encoding_size', dest='image_encoding_size', 
                        type=int, default=1024,
                        help='size of the image encoding')
    parser.add_argument('--word_encoding_size', dest='word_encoding_size', 
                        type=int, default=300,
                        help='size of word encoding')
    parser.add_argument('--hidden_size', dest='hidden_size', type=int, 
                        default=256,
                        help='size of hidden layer in generator RNNs')
    # lstm-specific params
    parser.add_argument('--tanhC_version', dest='tanhC_version', type=int, 
                        default=0, help='use tanh version of LSTM?')
    # rnn-specific params
    parser.add_argument('--rnn_relu_encoders', dest='rnn_relu_encoders', 
                        type=int, default=0,
                        help='relu encoders before going to RNN?')
    parser.add_argument('--rnn_feed_once', dest='rnn_feed_once', type=int, 
                        default=0,
                        help='feed image to the rnn only single time?')

    # optimization parameters
    parser.add_argument('-c', '--regc', dest='regc', type=float, default=1e-8, 
                        help='regularization strength')
    parser.add_argument('-m', '--max_epochs', dest='max_epochs', type=int, 
                        default=50,
                        help='number of epochs to train for')
    parser.add_argument('--solver', dest='solver', type=str, default='rmsprop',
                        help='solver type: vanilla/adagrad/adadelta/rmsprop')
    parser.add_argument('--momentum', dest='momentum', type=float, default=0.0, 
                        help='momentum for vanilla sgd')
    parser.add_argument('--decay_rate', dest='decay_rate', type=float, 
                        default=0.999,
                        help='decay rate for adadelta/rmsprop')
    parser.add_argument('--smooth_eps', dest='smooth_eps', type=float, 
                        default=1e-8,
                        help='epsilon smoothing for rmsprop/adagrad/adadelta')
    parser.add_argument('-l', '--learning_rate', dest='learning_rate', 
                        type=float, default=1e-3,
                        help='solver learning rate')
    parser.add_argument('-b', '--batch_size', dest='batch_size', type=int, 
                        default=128, help='batch size')
    parser.add_argument('--grad_clip', dest='grad_clip', type=float, 
                        default=5,
                        help='clip gradients elementwise. at what threshold?')
    parser.add_argument('--drop_prob_encoder', dest='drop_prob_encoder', 
                        type=float, default=0.5,
                        help='dropout right after the encoder to an RNN/LSTM')
    parser.add_argument('--drop_prob_decoder', dest='drop_prob_decoder', 
                        type=float, default=0.5,
                        help='dropout right before the decoder in an RNN/LSTM')

    # data preprocessing parameters
    parser.add_argument('--word_count_threshold', dest='word_count_threshold', 
                        type=int, default=5,
                        help='if a word less than this number discarded')

    # evaluation parameters
    parser.add_argument('-p', '--eval_period', dest='eval_period', type=float, 
                        default=1.0,
                        help='in units of epochs, how often eval on val set?')
    parser.add_argument('--eval_batch_size', dest='eval_batch_size', type=int, 
                        default=128,
                        help='what batch size to use on val img/sentences?')
    parser.add_argument('--eval_max_images', dest='eval_max_images', type=int, 
                        default=-1,
                        help='we can use a small no. of images to get val err')
    parser.add_argument('--min_ppl_or_abort', dest='min_ppl_or_abort', 
                        type=float, default=-1,
                        help='if val perplexity < thresh, the job will abort')

    args = parser.parse_args()
    params = vars(args)  # convert to ordinary dict
    print ('parsed parameters:')
    print (json.dumps(params, indent=2))
    init(params)
    main(params)
    