from torch.autograd import Variable
from numpy.lib.stride_tricks import sliding_window_view
import torch 
import cv2

import numpy as np 
#import pandas as pd
import matplotlib.pyplot as plt 

from tqdm import tqdm

import os 
import copy
import glob
from torch import nn # neural netowrk 
import timm

import segmentation_models_pytorch as smp
import random
from segmentation_models_pytorch.losses import DiceLoss, LovaszLoss, FocalLoss, JaccardLoss

from math import log

from torch import optim
import torchvision.models as models
import torch.nn.functional as f
import matplotlib
matplotlib.use('agg')
DEVICE = torch.device('cuda')
EPOCHS = 50 #25 training iterations
LR = 0.00001 #decay learning rate
BATCH_SIZE = 4
HEIGHT = 288
WIDTH = 480
ENCODER = 'resnet34'
WEIGHTS = 'imagenet'
DATA_URL = "/cs/student/projects1/2019/nsultana/"

BINARY_MODE = 'binary'
MULTILABEL_MODE = 'multilabel'
MULTICLASS_MODE = 'multiclass'
from torch.nn.modules.loss import _Loss

def calculate_weights(masks):
    labels = [0,1,2,3,4,5,6,7]
    weights_list = dict.fromkeys(labels,0)
    new_weights = []
    max_weights = dict.fromkeys(labels,0)
    ground_truth = masks.to('cpu').flatten().numpy().astype(int)

    total_weight = 0
    for label in labels:
        count = (ground_truth == label).sum()
        weights_list[label] += count
        total_weight +=count
    
    for label in labels:
        weight = weights_list[label]
        if weight == 0:
            new_weights.append(0)
        else:
            new_weights.append(1 - (weight/total_weight))

    return new_weights

def train_function(data_loader, model, optimizer):
    model.train()
    total_loss = 0.0
    for images, masks in tqdm(data_loader):
        loss = torch.tensor(0.0).to(DEVICE)
        count = torch.tensor(0.0).to(DEVICE)
        images = images.to(device = DEVICE)
        masks = masks.to(device = DEVICE, dtype=torch.long)
        optimizer.zero_grad()
        

        #ensure correct dimensions
        if images.dim()!=5:
            images = images.unsqueeze(0)
        if masks.dim()!=4:
            masks = masks.unsqueeze(0)
        
        logits = model(images, masks)
        
        # weights =  [0.6, 0.6, 0.7, 0.8, 0.9, 2.0, 0.9, 0.8, 0.8]
        # weights =  [0.8, 0.9, 1.0, 0.9, 0.8, 0.7]
        weights =  [0.0, 1.0, 0.0, 0.0, 0.0]
        ce_weights = calculate_weights(masks)
        ce_weights = torch.tensor(ce_weights,dtype=torch.float).to(DEVICE)
        weights = torch.tensor(weights,dtype=torch.float).to(DEVICE)
        # masks = masks[:,2:,:,:]

        
        for i in range (0, logits.shape[2]):
            logit = logits[:,:,i,:,:] 

            mask = masks[:,i+3,:,:]
            mask = mask.contiguous().long()
            
            lovasz = LovaszLoss(mode = 'multiclass', ignore_index=-1)(logit, mask)
            criterion = nn.CrossEntropyLoss(weight = ce_weights, ignore_index=-1)
            ce_logit = criterion(logit, mask)
            #loss += (lovasz + ce_logit) * weights[i]
            loss = (lovasz + ce_logit) 
            
            # print(loss.dtype)
            
            # count+= weights[i]
        
        # loss = loss / count
        output = 0
        loss = loss + output
        loss.backward()

        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)

def eval_function(data_loader, model):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for images, masks in tqdm(data_loader):
            loss = torch.tensor(0.0).to(DEVICE)
            count = torch.tensor(0.0).to(DEVICE)
            images = images.to(device = DEVICE)
            masks = masks.to(device = DEVICE, dtype=torch.long)
            # optimizer.zero_grad()

            #ensure correct dimensions
            if images.dim()!=5:
                images = images.unsqueeze(0)
            if masks.dim()!=4:
                masks = masks.unsqueeze(0)
            
            logits = model(images, masks)
            
            

            weights =  [0.9, 1.0, 0.9, 0.8, 0.8]
            weights =  [0.6, 0.6, 0.7, 0.8, 0.9, 2.0, 0.9, 0.8, 0.8]
            # weights =  [0.8, 0.9, 1.0, 0.9, 0.8, 0.7]
            #weights =  [0.0, 1.0, 0.0, 0.0, 0.0]
            ce_weights = calculate_weights(masks)
            ce_weights = torch.tensor(ce_weights,dtype=torch.float).to(DEVICE)
            weights = torch.tensor(weights,dtype=torch.float).to(DEVICE)
            #masks = masks[:,2:,:,:]
            
            for i in range (0, logits.shape[2]):
                logit = logits[:,:,i,:,:] 
                
                mask = masks[:,i+3,:,:]
                mask = mask.contiguous().long()
                lovasz = LovaszLoss(mode = 'multiclass', ignore_index=-1)(logit, mask)
                criterion = nn.CrossEntropyLoss(weight = ce_weights, ignore_index=-1)
                ce_logit = criterion(logit, mask)
                # loss += (lovasz + ce_logit) * weights[i]
                loss = (lovasz + ce_logit)
                #count+= weights[i]
                
            #loss = loss / count

            total_loss += loss.item()
    return total_loss / len(data_loader)


