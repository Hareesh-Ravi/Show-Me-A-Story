# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 15:18:19 2017

@author: HareeshRavi
"""
import csv
from keras.applications.vgg16 import VGG16
import json
import pickle
import numpy as np
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import time
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.models import Model
import utils_vist

# Function to extract final FC layer features from VGG 16
def ExtractFeat(model, imgname, image_size):
    try:
        try:    
            img = image.load_img(imgname, target_size = image_size)
        except FileNotFoundError:
            try:
                img = image.load_img(imgname[0:len(imgname)-3] + 'png', 
                                     target_size = image_size)
            except FileNotFoundError:
                 img = image.load_img(imgname[0:len(imgname)-3] + 'gif', 
                                      target_size = image_size)
    except OSError:
        
        return None
        
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    FeatImages = model.predict(img)
    return FeatImages


# SCript that runs through list of list "input array" of image file names to 
# read and extract features for each image. Store it in a JSON file for further
# retrieval
def Feat_forArray(inputArray, imagepath, model, image_size):
    
    FeatImages = {}
    story_wo_image = dict()
    for i in range(len(inputArray)):
        for j in range(5):
            curimg = str(inputArray[i][j])
            if curimg not in FeatImages:
                imgname = imagepath + str(curimg) + '.jpg'
                feat = ExtractFeat(model, imgname, image_size)
                try:
                    FeatImages[curimg] = feat.flatten().tolist()
                except AttributeError:
                    if i in story_wo_image:
                        story_wo_image[i].append(int(curimg))
                    else:
                        story_wo_image[i] = [curimg]
            else:
                pass
        print('stories processed: {}/{}'.format(i, len(inputArray)), 
              end='\r')        
   
    return FeatImages, story_wo_image            


# main func to extract VGG 16 features for images in VIST data
def main_func(datadir, process, model, isprune):
    
    print('processing {0:s} data'.format(process))
    # Load story to imageids csv file
    imageids = utils_vist.getImgIds(datadir + process + '/VISTImage' + 
                          process + 'Data.csv')  

    # image2path file
    imagepath = datadir + 'raw/images/' + process + '/'
    
    starttime = time.time()
    #Extract features for each unique image for training, testing and validation
    img_feats, story_noimg = Feat_forArray(imageids, imagepath, model, 
                                           image_size)
    print('features extracted for {0:s} in {1:f} seconds'.format(process, 
            time.time() - starttime))
    
    if isprune:
        # then post process VIST data to remove stories that do not have 
        # all the images present in the data.
        indexes = sorted(list(story_noimg.keys()), reverse=True)
        for index in indexes:
            del imageids[index]
        
        # do same for text stories as well 
        stories = utils_vist.getSent(datadir + process + '/VISTText' + 
                          process + 'Data.csv')
        
        for index in indexes:
            del stories[index]  
        
        # save deleted imageids and stories as CSV for further use
        utils_vist.write2csv(datadir + process + "/" + process + "_image.csv", 
                        imageids)
        utils_vist.write2csv(datadir + process + "/" + process + "_text.csv",
                        stories)
        
    return img_feats, story_noimg


if __name__ == '__main__':
    #organize "Story in sequence" of VIST dataset as story x sequence
        
    datadir = './data/'
    process = ['test', 'val', 'train']
    isprune = True
    
    # Load VGG16 model for extracting features for images in data. Save the 
    # extracted features   
    base_model = VGG16(weights= 'imagenet', include_top=True)
    model = Model(input=base_model.input, 
                  output=base_model.get_layer('fc2').output)
    model.summary()
    image_size = (224, 224)
    
    print('loaded VGG model...')
    for proc in process:
        img_feats, storynoimg = main_func(datadir, proc, model, isprune)
    
        with open(datadir + proc + '/VIST_ImageFeat' + 
                  proc + '.json', 'w') as JITE:
            json.dump(img_feats, JITE)
        with open(datadir + proc + '/VIST_StoryNoImg_' + 
                  proc + '.json', 'w') as JITE:
            json.dump(storynoimg, JITE)
        
        
    

