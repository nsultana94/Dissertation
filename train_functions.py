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
from segmentation_models_pytorch.losses import DiceLoss, LovaszLoss, FocalLoss, JaccardLoss

from math import log

from torch import optim
import torchvision.models as models
import torch.nn.functional as f

DEVICE = torch.device('cuda')
EPOCHS = 50 #25 training iterations
LR = 0.00001 #decay learning rate
BATCH_SIZE = 4
HEIGHT = 288
WIDTH = 480
ENCODER = 'resnet34'
WEIGHTS = 'imagenet'
DATA_URL = "/cs/student/projects1/2019/nsultana/"


def train_function(data_loader, model, optimizer):

  model.train()
  total_loss = 0.0

  for images, masks in tqdm(data_loader):

    images = images.to(device = DEVICE)
    masks = masks.to(device = DEVICE, dtype=torch.long)
    
    
    # make sure gradients are 0
    optimizer.zero_grad()
    logits = []
    triplet_sequences = []
    for i in range (0, len(images)):

      logits_mask = model(images[i], masks[i])
      logits.append(logits_mask)
      triplet_sequences.append(logits_mask.flatten())
        
    logits = torch.stack(logits)
    triplet_sequences = torch.stack(triplet_sequences)

    losses = []
    loss = 0
    count = 0
   
    weights =  [0.5, 0.7, 1.0, 0.7, 0.5, 0.1]
    
   


    ce_weights = calculate_weights(masks)
    # max_value = max(ce_weights)
    # ce_weights = list(map(lambda value: value / max_value, ce_weights))
    ce_weights = torch.tensor(ce_weights,dtype=torch.float).to(DEVICE)
    # logit shape is  4,8,6,288,480
  
    for i in range (0, 6):

      logit = logits[:,:,i,:,:] #iterate per frame
      mask = masks[:,i,:,:]
      mask = mask.contiguous()
      loss_per_frame = LovaszLoss(mode = 'multiclass', ignore_index=-1)(logit, mask)
      criterion = nn.CrossEntropyLoss(weight = ce_weights, ignore_index=-1)
      ce_logit = criterion(logit, mask)
      ce_logit = ce_logit * 0.5
    
      loss += (loss_per_frame + ce_logit)  * weights[i]
      count+= weights[i]


    loss = loss / count
    

    # calculate triplet loss 
    
    array = np.array([0,1,2,3,4,5])
    arr = sliding_window_view(array, window_shape = 3)
    triplet_loss = nn.TripletMarginLoss(margin=0.0, p=2)
    output = 0
    
    # for triplet in arr:
     
    #   anchor = logits[:,:,triplet[0],:, :].flatten()
    #   positive = logits[:,:,triplet[1],:, :].flatten()
    #   negative = logits[:,:,triplet[2],:, :].flatten()
    #   output += triplet_loss(anchor, positive, negative)
    
    # output = output / len(arr)
    
    loss = loss + output  



    

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
      for i in range (0, len(images)):
       
        logits.append(model(images[i], masks[i]))
      logits = torch.stack(logits)
      losses = []
      loss = 0
      count = 0
      weights =  [0.5, 0.7, 1.0, 0.7, 0.5, 0.1]
      ce_weights = calculate_weights(masks)
      # max_value = max(ce_weights)
      #ce_weights = list(map(lambda value: value / max_value, ce_weights))
      ce_weights = torch.tensor(ce_weights,dtype=torch.float).to(DEVICE)
        # logit shape is  4,8,6,288,480
      
      for i in range (0, 6):

        logit = logits[:,:,i,:,:] #iterate per frame
        mask = masks[:,i,:,:]
        mask = mask.contiguous()
        loss_per_frame = LovaszLoss(mode = 'multiclass', ignore_index=-1)(logit, mask)
        criterion = nn.CrossEntropyLoss(weight = ce_weights, ignore_index=-1)
        
        ce_logit = criterion(logit, mask)
        ce_logit = ce_logit * 0.5
        loss += (loss_per_frame + ce_logit) * weights[i]
        count+= weights[i]
      loss = loss / count
   
      # triplet loss
      array = np.array([0,1,2,3,4,5])
      arr = sliding_window_view(array, window_shape = 3)
      triplet_loss = nn.TripletMarginLoss(margin=0.0, p=2)
      output = 0
      
      # for triplet in arr:
      
      #   anchor = logits[:,:,triplet[0],:, :].flatten()
      #   positive = logits[:,:,triplet[1],:, :].flatten()
      #   negative = logits[:,:,triplet[2],:, :].flatten()
      #   output += triplet_loss(anchor, positive, negative)
      
      # output = output / len(arr)
      loss = loss + output  



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
            new_weights.append(total_weight * (1 / weight))
    
    return new_weights