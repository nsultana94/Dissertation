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

# def isnan(x):
#     return x != x


# def mean(values, ignore_nan=False, empty=0):
#     """Nanmean compatible with generators."""
#     values = iter(values)
#     if ignore_nan:
#         values = ifilterfalse(isnan, values)
#     try:
#         n = 1
#         acc = next(values)
#     except StopIteration:
#         if empty == "raise":
#             raise ValueError("Empty mean")
#         return empty
#     for n, v in enumerate(values, 2):
#         acc += v
#     if n == 1:
#         return acc
#     return acc / n

# def _lovasz_softmax(probas, labels, ce_weights, classes="present", per_image=False, ignore_index=None):
#     """Multi-class Lovasz-Softmax loss
#     Args:
#         @param probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1).
#         Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
#         @param labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
#         @param classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
#         @param per_image: compute the loss per image instead of per batch
#         @param ignore_index: void class labels
#     """
#     if per_image:
#         loss = mean(
#             _lovasz_softmax_flat(*_flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore_index), classes=classes, ce_weights= ce_weights)
#             for prob, lab in zip(probas, labels)
#         )
#     else:
#         loss = _lovasz_softmax_flat(*_flatten_probas(probas, labels, ignore_index), classes=classes, ce_weights = ce_weights)
#     return loss

# def _lovasz_grad(gt_sorted):
#     """Compute gradient of the Lovasz extension w.r.t sorted errors
#     See Alg. 1 in paper
#     """
#     p = len(gt_sorted)
#     gts = gt_sorted.sum()
#     intersection = gts - gt_sorted.float().cumsum(0)
#     union = gts + (1 - gt_sorted).float().cumsum(0)
#     jaccard = 1.0 - intersection / union
#     if p > 1:  # cover 1-pixel case
#         jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
#     return jaccard

# def _lovasz_softmax_flat(probas, labels,ce_weights, classes="present"):
#     """Multi-class Lovasz-Softmax loss
#     Args:
#         @param probas: [P, C] Class probabilities at each prediction (between 0 and 1)
#         @param labels: [P] Tensor, ground truth labels (between 0 and C - 1)
#         @param classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
#     """
#     if probas.numel() == 0:
#         # only void pixels, the gradients should be 0
#         return probas * 0.0
#     C = probas.size(1)
#     losses = []
#     class_to_sum = list(range(C)) if classes in ["all", "present"] else classes
#     for c in class_to_sum:
#         fg = (labels == c).type_as(probas)  # foreground for class c
#         if classes == "present" and fg.sum() == 0:
#             continue
#         if C == 1:
#             if len(classes) > 1:
#                 raise ValueError("Sigmoid output possible only with 1 class")
#             class_pred = probas[:, 0]
#         else:
#             class_pred = probas[:, c]
#         errors = (fg - class_pred).abs()
#         errors_sorted, perm = torch.sort(errors, 0, descending=True)
#         perm = perm.data
#         fg_sorted = fg[perm]
#         losses.append(torch.dot(errors_sorted, _lovasz_grad(fg_sorted)))
    
    
#     weighted_loss= 0
#     total = 0
#     for i in range(0, len(losses)):

#       weighted_loss+= losses[i] * ce_weights[i]
#       total += ce_weights[i]
    
#     loss_value = weighted_loss / total


#     return loss_value

# def _flatten_probas(probas, labels, ignore=None):
#     """Flattens predictions in the batch"""
#     if probas.dim() == 3:
#         # assumes output of a sigmoid layer
#         B, H, W = probas.size()
#         probas = probas.view(B, 1, H, W)

#     C = probas.size(1)
#     probas = torch.movedim(probas, 1, -1)  # [B, C, Di, Dj, ...] -> [B, Di, Dj, ..., C]
#     probas = probas.contiguous().view(-1, C)  # [P, C]

#     labels = labels.view(-1)
#     if ignore is None:
#         return probas, labels
#     valid = labels != ignore
#     vprobas = probas[valid]
#     vlabels = labels[valid]
#     return vprobas, vlabels

    
# class LovaszLoss(_Loss):
#     def __init__(self, per_image=False, ignore=None, classes=None, ce_weights = None):
#         super().__init__()
#         self.ignore = ignore
#         self.per_image = per_image
#         self.classes = classes
#         self.ce_weights =ce_weights
#     def forward(self, logits, target):
#         logits = logits.softmax(dim=1)
#         return _lovasz_softmax(logits, target, per_image=self.per_image, ignore_index=self.ignore, classes=self.classes, ce_weights =self.ce_weights)
    
def train_function(data_loader, model, optimizer, unet):

  model.train()
  total_loss = 0.0

  for images, masks in tqdm(data_loader):

    images = images.to(device = DEVICE)
    masks = masks.to(device = DEVICE, dtype=torch.long)
    
    
    # make sure gradients are 0
    optimizer.zero_grad()
    logits = []
    cell_states = []


    # for i in range (0, len(images)):
    #   # unet.eval()
    #   # with torch.no_grad():
    #   #   logit_mask = unet(images[i][0].unsqueeze(0))
    #   # predictions = torch.nn.functional.softmax(logit_mask, dim=1)
    #   # initial_mask =torch.argmax(predictions, dim=1)
    #   # initial_mask = initial_mask.unsqueeze(0)
    #   # masks[i][0] = initial_mask
    #   logits_mask, cell_state = model(images[i], masks[i])

    #   logits.append(logits_mask)
    #   cell_states.append(cell_state)
    
    logits, cell_states = model(images, masks)

    # logits = torch.stack(logits)
    # cell_states = torch.stack(cell_states)
    
    if logits.dim() !=5:
      logits = logits.unsqueeze(0)

    losses = []
    loss = 0
    count = 0

    # weights =  [0.5, 0.7, 0.9, 2.0, 0.9, 0.7, 0.5]
    weights =  [0.6, 0.8, 0.9, 2.0]
    #weights = [0.2,0.6, 0.2]
    ce_weights = calculate_weights(masks)
    lovasz_weights = [1.0,1.0,1.0,1.0,1.0,1.0,1.0, 1.0]

    ce_weights = torch.tensor(ce_weights,dtype=torch.float).to(DEVICE)
    lovasz_weights = torch.tensor(lovasz_weights,dtype=torch.float).to(DEVICE)
    
     
     

    # logit shape is  4,8,6,288,480
  
    for i in range (0, logits.shape[2]):
   
        logit = logits[:,:,i,:,:] #iterate per frame

        mask = masks[:,i,:,:]

        mask = mask.contiguous().long()
        loss_per_frame = LovaszLoss(mode = 'multiclass', ignore_index=-1)(logit, mask)
        # loss_per_frame= LovaszLoss(ignore=-1, classes = "present", ce_weights = lovasz_weights)(logit, mask)
        criterion = nn.CrossEntropyLoss(weight = ce_weights, ignore_index=-1)
        ce_logit = criterion(logit, mask)
    
        loss += (loss_per_frame + ce_logit)  * weights[i]
        #loss += loss_per_frame  * weights[i]
        count+= weights[i]
  
    loss = loss / count
   
    # calculate triplet loss 
    
    array = np.array([0,1,2,3,4,5])

    triplets = [[1,2,3],[2,3,4], [3,4,5], [0,1,2]]
    triplet_loss = nn.TripletMarginLoss(margin=0.0, p=2)
    output = 0

    # for i in range (0,25):
    #     t1 = random.randint(0, 3)
    #     t2 = random.randint(t1+1, 4)
    #     t3= random.randint(t2+1, 5)
    #     triplet = [t1,t2,t3]
    #     if triplet not in triplets:
    #         triplets.append(triplet)
    
    # for triplet in triplets:
    #     anchor = cell_states[triplet[0],:,:,:,:].flatten()
    #     positive = cell_states[triplet[1],:,:,:,:].flatten()
    #     negative = cell_states[triplet[2],:,:,:,:].flatten()
    #     output+= triplet_loss(anchor, positive, negative)
        
    
    
    # for triplet in arr:
    #     anchor = logits[:,7,triplet[0],:,:].flatten()
    #     positive = logits[:,7,triplet[1],:,:].flatten()
    #     negative = logits[:,7,triplet[2],:,:].flatten()
    #     output+= triplet_loss(anchor, positive, negative)
    
    

    # output = output / len(triplets)
    # output = output * 0.25
    
    loss = loss + output  

    loss.backward() #backpropagation

    optimizer.step() #update weights

    total_loss += loss.item()
    
  return total_loss / len(data_loader)



def eval_function(data_loader, model, unet):

  model.eval() 
  total_loss = 0.0

  with torch.no_grad():
    for images, masks in tqdm(data_loader):

      images = images.to(device = DEVICE)
      masks = masks.to(device = DEVICE, dtype=torch.long)
      logits = []
      cell_states = []
      # for i in range (0, len(images)):
      #   # unet.eval()
      #   # with torch.no_grad():
      #   #     logit_mask = unet(images[i][0].unsqueeze(0))
      #   # predictions = torch.nn.functional.softmax(logit_mask, dim=1)
      #   # initial_mask =torch.argmax(predictions, dim=1)
      #   # initial_mask = initial_mask.unsqueeze(0)
      #   # masks[i][0] = initial_mask
       
      #   logits_mask, cell_state = model(images[i], masks[i])
      #   logits.append(logits_mask)
      #   cell_states.append(cell_state)

      logits, cell_states = model(images, masks)
    
      # logits = torch.stack(logits)
      # cell_states = torch.stack(cell_states)
      
      if logits.dim() !=5:
        logits = logits.unsqueeze(0)
    
    
      # logits = torch.stack(logits)
      # cell_states = torch.stack(cell_states)
      losses = []
      loss = 0
      count = 0
      #weights =  [0.6, 0.8, 0.9, 2.0, 0.9, 0.8, 0.6]
      weights =  [0.6, 0.8, 0.9, 2.0]
      #weights = [0.2,0.6, 0.2]
      ce_weights = calculate_weights(masks)
      lovasz_weights = [1.0,1.0,1.0,1.0,1.0,1.0,1.0, 1.0]
     

      ce_weights = torch.tensor(ce_weights,dtype=torch.float).to(DEVICE)
      lovasz_weights = torch.tensor(lovasz_weights,dtype=torch.float).to(DEVICE)
        # logit shape is  4,8,6,288,480
      
      for i in range (0, logits.shape[2]):

        logit = logits[:,:,i,:,:] #iterate per frame

        mask = masks[:,i,:,:]
        mask = mask.contiguous().long()
        loss_per_frame = LovaszLoss(mode = 'multiclass', ignore_index=-1)(logit, mask)
        # loss_per_frame= LovaszLoss(ignore=-1, classes = "present", ce_weights = lovasz_weights)(logit, mask)
        criterion = nn.CrossEntropyLoss(weight = ce_weights, ignore_index=-1)
        ce_logit = criterion(logit, mask)
        #ce_logit = ce_logit * 0.25
    
        loss += (loss_per_frame + ce_logit)  * weights[i]
        #loss += loss_per_frame  * weights[i]
        count+= weights[i]

      loss = loss / count
   
      # triplet loss
      array = np.array([0,1,2,3,4,5])
      triplets = [[1,2,3],[2,3,4], [3,4,5], [0,1,2]]
      triplet_loss = nn.TripletMarginLoss(margin=0.0, p=2)
      output = 0

      # for i in range (0,25):
      #   t1 = random.randint(0, 3)
      #   t2 = random.randint(t1+1, 4)
      #   t3= random.randint(t2+1, 5)
      #   triplet = [t1,t2,t3]
      #   if triplet not in triplets:
      #       triplets.append(triplet)

      
      # for triplet in triplets:
      #   anchor = cell_states[triplet[0],:,:,:,:].flatten()
      #   positive = cell_states[triplet[1],:,:,:,:].flatten()
      #   negative = cell_states[triplet[2],:,:,:,:].flatten()
      #   output+= triplet_loss(anchor, positive, negative)
            
    

      # output = output / len(triplets)
      # output = output * 0.25
    # print(loss)
   
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
            new_weights.append(1 - (weight/total_weight))

    return new_weights