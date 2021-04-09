#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: trivizakis

@github: github.com/trivizakis
"""
import pickle as pkl
import pandas as pd
import numpy as np
import time
import sys
import os

import keras.backend as K

sys.path.append('../transfer_learning')
from transfer_learning import Transferable_Networks

def classify(features, labels_per_patient, labels_per_slice, hypes):
    performance={}
    model_names = np.array(list(features.keys()))
    for model_name in model_names:
        features_=features[model_name]
        
        if hypes["num_classes"] == 2:
            print(model_name+": Binary Classification")
            classification_type = " binary"
        else:
            print(model_name+": Multiclass Classification")      
            classification_type = " multiclass"  
        performance[model_name+classification_type] = tn.classification_per_patient(hypes,features_,labels_per_patient,labels_per_slice)
    return pd.DataFrame.from_dict(performance)
    
def save_df_to_csv(df, file_name):
    df.to_csv(file_name, index = True) 
    
def padding2d(img,max_x):
    #pads only the x,y not the z
    if not img.shape[0] % 2:
        diff = max_x - img.shape[0]
        diff_2 = int(diff/2)
        temp = np.pad(img,[(diff_2,diff_2),(0,0)],mode="constant", constant_values=0)
    else:
        diff = max_x - img.shape[0] - 1
        diff_2 = int(diff/2)
        temp = np.pad(img,[(diff_2,diff_2),(0,0)],mode="constant", constant_values=0)
        temp = np.pad(temp,[(1,0),(0,0)],mode="constant", constant_values=0)
#        print(temp.shape)        
    if not img.shape[1] % 2:
        diff = max_x - img.shape[1]
        diff_2 = int(diff/2)
        temp = np.pad(temp,[(0,0),(diff_2,diff_2)],mode="constant", constant_values=0)
    else:
        diff = max_x - img.shape[1] - 1
        diff_2 = int(diff/2)
        temp = np.pad(temp,[(0,0),(diff_2,diff_2)],mode="constant", constant_values=0)
        temp = np.pad(temp,[(0,0),(1,0)],mode="constant", constant_values=0)    
    # print(temp.shape)
    return temp

dataset_dir = "dataset/"

hypes={}
hypes["input_shape"] = (250,250,1)
hypes["num_classes"] = 3 
hypes["pooling"] = None
hypes["kernel"] = "rbf" #classifier
hypes["folds"] = 10
hypes["normalization"] = "standardize" #feature-level

model_names = ["xception",
                "vgg16","vgg19",
                "resnet50","resnet101","resnet152",
                 "resnetv250","resnetv2101","resnetv2152",
                 "inc3", "incr2",
                 "mobile","mobilev2",
                 "densenet121", "densenet169", "densenet201",
                "nasnetm","nasnetl"]

#image id based labeling
labels_3class = {}
images_ = {}
for _, _, imgs in os.walk(dataset_dir):
    for img in imgs:
        images_[img]=np.load(dataset_dir+img).astype(np.float32)
        if "normal" in img:
            labels_3class[img] = 0
        elif "Ganglioneuroblastoma" in img:
            labels_3class[img] = 1
        elif "Neuroblastoma" in img:
            labels_3class[img] = 2
        elif "others" in img:
            labels_3class[img] = 2
            

#statistics for image normalization prior to padding/cropping
for index,img in enumerate(list(images_.keys())):
    if index == 0:
        raveled_dataset = images_[img].ravel()
    else:
        raveled_dataset = np.concatenate((raveled_dataset,images_[img].ravel()))
        
max_= raveled_dataset.max().astype(np.float32)
min_= raveled_dataset.min().astype(np.float32)
std_= raveled_dataset.std().astype(np.float32)
mean_= raveled_dataset.mean().astype(np.float32)

new_labels_3class = {}
new_imgs = {}
for img in list(images_.keys()):    
    # normalized = (images_[img] - mean_) / std_
    # normalized[normalized>1]=1
    # normalized[normalized<-1]=-1  
    
    normalized = (images_[img].astype(np.float32) - min_) / (max_ - min_)
    normalized[normalized>1]=1
    normalized[normalized<0]=0
    
    x,y = normalized.shape 
    
    if x<hypes["input_shape"][0] and y<hypes["input_shape"][1]:
        new_imgs[img] =  np.expand_dims(padding2d(normalized,hypes["input_shape"][0]),axis=-1).astype(np.float32)
        new_labels_3class[img] = labels_3class[img]
    elif x>hypes["input_shape"][0] and y<hypes["input_shape"][1]:
        # diff = x - hypes["input_shape"][0]
        howmanytimes = x//hypes["input_shape"][0]
        
        rest =  howmanytimes*hypes["input_shape"][0] - x
        if rest<-60:
            new_imgs[img+" "+str(howmanytimes)] =  np.expand_dims(padding2d(normalized[rest:,:],hypes["input_shape"][0]),axis=-1).astype(np.float32)
            new_labels_3class[img+" "+str(howmanytimes)] = labels_3class[img]      
        for i in range(0,howmanytimes):
            border0 = (howmanytimes-1-i)*hypes["input_shape"][0]
            border = x-hypes["input_shape"][0]*i + rest
            new_imgs[img+" "+str(i)] =  np.expand_dims(padding2d(normalized[hypes["input_shape"][0]*i:border,:],hypes["input_shape"][0]),axis=-1).astype(np.float32)
            new_labels_3class[img+" "+str(i)] = labels_3class[img]
    elif x<hypes["input_shape"][0] and y>hypes["input_shape"][1]:
        # diff = y - hypes["input_shape"][1]
        howmanytimes = y//hypes["input_shape"][1]
        
        rest =  howmanytimes*hypes["input_shape"][1] - y
        if rest<-60:
            new_imgs[img+" "+str(howmanytimes)] =  np.expand_dims(padding2d(normalized[:,rest:],hypes["input_shape"][1]),axis=-1).astype(np.float32)
            new_labels_3class[img+" "+str(howmanytimes)] = labels_3class[img]
        for i in range(0,howmanytimes):
            border0 = (howmanytimes-1-i)*hypes["input_shape"][1]
            border = y-hypes["input_shape"][1]*i + rest
            new_imgs[img+" "+str(i)] =  np.expand_dims(padding2d(normalized[:,border0:border],hypes["input_shape"][1]),axis=-1).astype(np.float32)
            new_labels_3class[img+" "+str(i)] = labels_3class[img] 
    elif x>hypes["input_shape"][0] and y>hypes["input_shape"][1]:
        if x>y:
            temp = padding2d(normalized,x).astype(np.float32)
            dim = x
        elif x<y:
            temp = padding2d(normalized,y).astype(np.float32)
            dim = y
        elif x==y:
            temp = normalized.astype(np.float32)
            dim = x
            
        howmanytimes = dim//hypes["input_shape"][1]
        
        rest =  howmanytimes*hypes["input_shape"][1] - dim
        if rest<-60:
            new_imgs[img+" "+str(howmanytimes)] =  np.expand_dims(padding2d(normalized[rest:,rest:],hypes["input_shape"][1]),axis=-1).astype(np.float32)
            new_labels_3class[img+" "+str(howmanytimes)] = labels_3class[img]
        for i in range(0,howmanytimes):
            border0 = (howmanytimes-1-i)*hypes["input_shape"][1]
            border = dim-hypes["input_shape"][1]*i + rest #either x or y : it the same
            new_imgs[img+" "+str(i)] =  np.expand_dims(padding2d(normalized[border0:border,border0:border],hypes["input_shape"][1]),axis=-1).astype(np.float32)
            new_labels_3class[img+" "+str(i)] = labels_3class[img]
    else:
        print("Unknown: "+img+" "+str(normalized.shape))
       
thres=0.0
for thres in [0.0,0.1,0.2,0.3,0.4,0.5]:   
    deep_features = {}
    deep_features_avg = {}
    deep_features_max = {}
    time_to_infer={}
    for model_name in model_names:
        print(model_name)
        #clear session in every iteration        
        K.clear_session()
        
        #NO POOLING
        #init object
        tn = Transferable_Networks(hypes)
            
        #timer STARTS
        start_time = time.time()
        deep_features[model_name] = tn.extract_features(new_imgs,model_name=model_name,threshold=thres)
        time_to_infer["Model: "+model_name+" Thresshold: "+str(thres)] = time.time() - start_time
    
        #AVG POOLING
        hypes["pooling"]="avg"
        
        K.clear_session()
        
        #init object
        tn = Transferable_Networks(hypes)
            
        deep_features_avg[model_name] = tn.extract_features(new_imgs,model_name=model_name,threshold=thres)
        
        #MAX POOLING
        hypes["pooling"]="max"
        
        K.clear_session()
        
        #init object
        tn = Transferable_Networks(hypes)
            
        deep_features_max[model_name] = tn.extract_features(new_imgs,model_name=model_name,threshold=thres)
        
    os.makedirs("performance_reports/thres_"+str(thres))
    
    with open("performance_reports/thres_"+str(thres)+"/deep_features_raw.pkl","wb") as file:
        pkl.dump(deep_features,file)
    
    with open("performance_reports/thres_"+str(thres)+"/deep_features_avg.pkl","wb") as file:
        pkl.dump(deep_features_avg,file)
        
    with open("performance_reports/thres_"+str(thres)+"/deep_features_max.pkl","wb") as file:
        pkl.dump(deep_features_max,file)
        
    # classification
    perf_df_raw = classify(deep_features,labels_3class,new_labels_3class,hypes)
    perf_df_avg = classify(deep_features_avg,labels_3class,new_labels_3class,hypes)
    perf_df_max = classify(deep_features_max,labels_3class,new_labels_3class,hypes)
    
    save_df_to_csv(perf_df_raw, "performance_reports/thres_"+str(thres)+"/performance_raw.csv")
    save_df_to_csv(perf_df_avg, "performance_reports/thres_"+str(thres)+"/performance_avg.csv")
    save_df_to_csv(perf_df_max, "performance_reports/thres_"+str(thres)+"/performance_max.csv")