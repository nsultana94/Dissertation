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



def train_function(data_loader, model, optimizer):

  model.train()
  total_loss = 0.0

  for images, masks in tqdm(data_loader):

    images = images.to(device = DEVICE)
    masks = masks.to(device = DEVICE, dtype=torch.long)
    
    
    # make sure gradients are 0
    optimizer.zero_grad()
    logits = []
    cell_states = []

    if images.dim()!=5:
      images = images.unsqueeze(0)
      
    if masks.dim()!=4:
      masks = masks.unsqueeze(0)

    logits, cell_states = model(images, masks)

    
    if logits.dim() !=5:
      logits = logits.unsqueeze(0)

    losses = []
    loss = 0
    count = 0

    # weights =  [0.5, 0.7, 0.9, 2.0, 0.9, 0.7, 0.5]

    weights =  [0.8, 0.9, 1.0, 0.9, 0.8, 0.7]
    #weights = [0.2,0.6, 0.2]
    ce_weights = calculate_weights(masks)

    ce_weights = torch.tensor(ce_weights,dtype=torch.float).to(DEVICE)
   
    
     
    
    # logit shape is  4,8,6,288,480
  
    for i in range (1, logits.shape[2]):
   
        logit = logits[:,:,i,:,:] #iterate per frame
        
        #mask = masks[:,i,:,:]
        mask = masks[:,i+1,:,:]

        mask = mask.contiguous().long()
        loss_per_frame = LovaszLoss(mode = 'multiclass', ignore_index=-1)(logit, mask)
        criterion = nn.CrossEntropyLoss(weight = ce_weights, ignore_index=-1)
        ce_logit = criterion(logit, mask)

        loss += (loss_per_frame + ce_logit)  * weights[i]
        count+= weights[i]
  
    loss = loss / count

    loss.backward() #backpropagation

    optimizer.step() #update weights

    total_loss += loss.item()
    
  return total_loss / len(data_loader)



def eval_function(data_loader, model):

  model.eval() 
  total_loss = 0.0

  with torch.no_grad():
    for images, masks in tqdm(data_loader):

      images = images.to(device = DEVICE)
      masks = masks.to(device = DEVICE, dtype=torch.long)
      logits = []
      cell_states = []

      if images.dim()!=5:
        images = images.unsqueeze(0)
      
      if masks.dim()!=4:
        masks = masks.unsqueeze(0)


      logits, cell_states = model(images, masks)

      if logits.dim() !=5:
        logits = logits.unsqueeze(0)

      losses = []
      loss = 0
      count = 0
      weights =  [0.8, 0.9, 1.0, 0.9, 0.8, 0.7]

      ce_weights = calculate_weights(masks)
 
     

      ce_weights = torch.tensor(ce_weights,dtype=torch.float).to(DEVICE)
        # logit shape is  4,8,6,288,480
      
      for i in range (1, logits.shape[2]):

        logit = logits[:,:,i,:,:] #iterate per frame

        mask = masks[:,i+1,:,:]
        mask = mask.contiguous().long()
        loss_per_frame = LovaszLoss(mode = 'multiclass', ignore_index=-1)(logit, mask)

        criterion = nn.CrossEntropyLoss(weight = ce_weights, ignore_index=-1)
        ce_logit = criterion(logit, mask)
        #ce_logit = ce_logit * 0.25
        loss += (loss_per_frame + ce_logit)  * weights[i]

        count+= weights[i]

      loss = loss / count


      total_loss += loss.item()

  return total_loss / len(data_loader)


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