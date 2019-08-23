# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 13:26:05 2017

@author: HareeshRavi
"""

import json
import csv
import os
import utils_vist
import config_all

# process annotations dict to get image ids and text for stories
# This gives two "No.Of.Stories x 5" sized matrices - one having text and
# another with image names (IDs)
def sis_2_csv(data, process, savedir):
    
    # initialize stuff
    storyId = 0
    ImageData = []
    TextData = []
    TotImageData = []
    TotTextData = []
    StoryCount = -1
    no_of_samples = len(data)
    
    for i in range(no_of_samples):
        storyTempId = data[i][0]["story_id"]
    
        if storyTempId != storyId:
            storyId = storyTempId
            
            StoryCount = StoryCount + 1
            
            TotImageData.append(ImageData)
            TotTextData.append(TextData)
            
            ImageData = []
            TextData = []
        
        ImageData.append(data[i][0]["photo_flickr_id"]) 
    
        TextData.append(data[i][0]["text"])
    
        print('stories processed: {}/{}'.format(i, no_of_samples), end='\r')

    TotImageData.pop(0) 
    TotTextData.pop(0)
    
    # save extracted files
    filename = os.path.join(savedir, process)
    try: 
        os.makedirs(filename)
    except FileExistsError:
        pass
    
    # save files
    utils_vist.write2csv(filename + ("/" + process + "_imageids.csv"), 
                         TotImageData)
    utils_vist.write2csv(filename + ("/" + process + "_stories.csv"), 
                         TotTextData)
        
    return TotImageData, TotTextData

# process annotations dict to get image ids and text for stories
# This gives two "No.Of.Stories x 5" sized matrices - one having text and
# another with image names (IDs)
def dii_2_json(data, process, savedir):
    
    caption_data = {}
    
    no_of_caps = len(data)
    for i in range(no_of_caps):
        storyTempId = data[i][0]["photo_flickr_id"]
    
        if storyTempId in caption_data:
            caption_data[storyTempId] = (caption_data[storyTempId] + 
                                         [data[i][0]["text"]])
        else:
            caption_data[storyTempId] = [data[i][0]["text"]]
        
        print('captions processed: {}/{}'.format(i, no_of_caps), end='\r')
    
    # save extracted files
    filename = os.path.join(savedir, process)
    try: 
        os.makedirs(filename)
    except FileExistsError:
        pass
    
    with open(filename + '/' + process + '_captions.json','w') as file:
        json.dump(caption_data,file)

if __name__ == '__main__':
    
    try:
        config = json.load(open('config.json'))
    except FileNotFoundError:
        config = config_all.create_config()
    datadir = config['datadir'] + 'raw/'
    savedir = config['datadir']
    
    #organize "Story in sequence" of VIST dataset as story x sequence

    process = ['train', 'val', 'test']
    for proc in process:
        json_data=open(datadir + 'sis/' + 
                       proc + '.story-in-sequence.json').read()
        data = json.loads(json_data)
        annotations = data["annotations"]
        
        imagedata, textdata = sis_2_csv(annotations, proc, savedir, config)
    
    # Organize description in isolation for VIST dataset
     
    for proc in process:
        json_data=open(datadir + 'dii/' + 
                       proc + '.description-in-isolation.json').read()
        data = json.loads(json_data)
        
        annotations = data["annotations"]
        
        capdata = dii_2_json(annotations, proc, savedir, config)
    
        


