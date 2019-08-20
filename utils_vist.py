# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 18:16:35 2019

@author: HareeshRavi
"""
import csv

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

