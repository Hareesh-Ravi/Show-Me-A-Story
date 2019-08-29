# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 18:16:35 2019

@author: HareeshRavi
"""
import csv
from keras import backend as K
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import normalize

# read CSV file of image IDs
def getImgIds(filename):
    out_list = []
    with open(filename, "r") as readtextfile:
        reader = csv.reader(readtextfile)
        for row in reader:
            if (len(row) > 1):
                out_list.append([int(i) for i in row])
    return out_list

# read all sentences per story for all stories as list of lists
def getSent(filename):
    temp = []
    sents = []
    with open(filename) as readtextfile:
        reader = csv.reader(readtextfile)
        for row in reader:
            for col in row:
                temp.append(col)
            sents.append(temp)
            temp = []
    return sents

# write data to CSV file
def write2csv(filename, data):
    
    with open(filename, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerows(data)

# get list of all sentences and corresponding image features 
def flatten_all(imgID_set, imgFea_set, sents_set):
    sents = []
    imgs = []
    ids = []
    for i, imgIDs in enumerate(imgID_set):
        for ind in range(5):
            imgID = str(imgIDs[ind])
            img_fea = imgFea_set[imgID]
            sents.append(sents_set[i][ind])
            imgs.append(img_fea)
            ids.append(int(imgID))
    return sents, imgs, ids

# for baseline and stage1
def MyCustomLoss(yTure, yPred):
    return yPred

# contrastive loss used as part of the baseline network
def contrastive_loss(s_im):
    
    margin = 0.005
    
    # For a minibatch of sentence and image embeddings, 
    # compute the pairwise contrastive loss
    s = s_im[0]
    im = s_im[1]

    # create two tensor 1xnumxdim  numx1xdim
    s2 = K.expand_dims(s, 1)
    im2 = K.expand_dims(im, 0)

    errors = K.sum(K.pow(K.maximum(0.0, s2 - im2), 2), axis=2)    	
    diagonal = tf.diag_part(errors)
    
    # all constrastive image for each sentence
    cost_s = K.maximum(0.0, margin - errors + diagonal) 
    
    # all contrastive sentences for each image
    cost_im = K.maximum(0.0, margin - errors + K.reshape(diagonal,[-1, 1]))  

    cost_tot = cost_s + cost_im

    cost_tot = tf.matrix_set_diag(cost_tot, tf.zeros(tf.shape(s)[0]))

    return K.sum(cost_tot)

# this is one-to-one retrieval accuracy used for pretraining stage 1
def retriv_acc(s_im):
    
    # For a minibatch of sentence and image embeddings, 
    # compute the retrieval accuracy
    s = s_im[0]
    im = s_im[1]

    s2 = K.expand_dims(s, 1)
    im2 = K.expand_dims(im, 0)
    order_violation = K.pow(K.maximum(0.0, s2 - im2), 2)
    errors = K.sum(order_violation, axis=2)
    inds = K.argmin(errors, axis=1)
    inds = tf.cast(inds, tf.int32)
    inds_true = tf.range(tf.shape(s)[0])
    elements_equal_to_value = tf.equal(inds, inds_true)
    as_ints = tf.cast(elements_equal_to_value, tf.int32)
    results = tf.reduce_sum(as_ints)
    results = tf.cast(results, tf.float32)

    return results

# for baseline
def edis_outputshape(input_shape):
    shape = list(input_shape)
    assert len(shape)==2
    outshape = (shape[0][0],1)
    return tuple(outshape)

# to retrieve images given true, pred and imageids. 
def retrieve_images(truevec, predvec, imageids):
    
    imageidx = []
    y_true = []
    for i in range(len(predvec)):
        for j in range(5):
            imageidx.append(imageids[i][j])
            y_true.append(truevec[i][j])
            
    # remove identical images so that distance calculation is not repeated
    _, mod_idx = np.unique(imageidx, axis=0, return_index=True)
    mod_idx = mod_idx.tolist()
    truenorm = normalize(np.abs(y_true[mod_idx, :]),norm='l2',axis=1) 
    
    predimageids = []
    outtemp = []
    for i in range(0, np.size(predvec, 0)):
        for j in range(0, np.size(predvec, 1)):
            prednorm = normalize(np.abs(predvec[i][j]).reshape(1, -1), 
                                 norm='l2', axis=1)
            error = np.subtract(prednorm, truenorm)
            subtemp = np.sum(np.square(np.maximum(0, error)), axis=1)             
#            temp1 = np.argpartition(subtemp, range(topk))
#            idx = [imageidx[mod_idx[tmp]] for tmp in temp1[:topk]]
            outtemp.append(imageidx[mod_idx[np.argmin(subtemp)]])  
        predimageids.append(outtemp)      
        outtemp = []
        print('retrieved for {}/{} stories'.format(i, len(predvec)), end='\r')
        
    return predimageids