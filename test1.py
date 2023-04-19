# -*- coding: utf-8 -*-
"""conv lstm.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1pk_9y1txgv_KprxWLG0C-PkaUjAF-E3J

# Installation
"""

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

from networks import SegmentationModel, ConvLSTMCell, LSTMModel, Network
from dataloader import get_train_augs, get_test_augs, get_valid_augs, SegmentationDataset
from train_functions import train_function, eval_function
DEVICE = torch.device('cuda') 
# DEVICE = 'cuda' #Cuda as using GPU
import pandas as pd
from sklearn.metrics import jaccard_score, accuracy_score
import matplotlib
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
# from initialiser_train import UnetInitialiser

import logging 
logging.basicConfig(filename="std.log", 
					format='%(asctime)s %(message)s', 
					filemode='w')

logger=logging.getLogger() 
logger.setLevel(logging.DEBUG) 

EPOCHS = 50 #25 training iterations
LR = 0.00001 #decay learning rate
BATCH_SIZE = 4
HEIGHT = 288
WIDTH = 480
ENCODER = 'resnet34'
WEIGHTS = 'imagenet'
DATA_URL = "/cs/student/projects1/2019/nsultana/"

import random
training_images = (glob.glob(f"{DATA_URL}new_data/Training1/*.npz"))



testing_images = (glob.glob(f"{DATA_URL}new_data/Testing/*.npz"))

validation_images = (glob.glob(f"{DATA_URL}new_data/Validation/*.npz"))
import warnings
warnings.filterwarnings('always')  # "error", "ignore", "always", "default", "module" or "once"

"""# Set up model"""

import albumentations as A
from albumentations.pytorch import ToTensorV2


model = SegmentationModel()
model = model.to(device =DEVICE) #i.e CUDA



"""# Set up dataset and data loader"""

class UnetInitialiser(nn.Module):
    def __init__(self, encoder,convlstm,decoder, head, sizes):
        super(UnetInitialiser, self).__init__()
        self.encoder = encoder
        self.convlstm = convlstm
        self.decoder = decoder
        self.head = head


    def forward(self, images, masks = None):

        
        
        image_0 = images[:,0,:,:,:]
        image_1 = images[:,1,:,:,:]
        image_2 = images[:,2,:,:,:]
        if masks!=None:
            mask = masks[:,2,:,:]
            mask = mask.contiguous().long()

        
        fv1 = self.encoder(image_0)
        fv2 = self.encoder(image_1)
        x_tilda = self.encoder(image_2)
        features = x_tilda.copy()
        

        cell_states = []
        hidden_states = []


        c0 = fv1[5]
        h0 = fv2[5]

        x_tilda = self.encoder(image_2)
        feature_vector = x_tilda[5]

        c_next,h_next = self.convlstm(feature_vector, h0, c0)
        x_tilda[5] = h_next
        decoder_output = self.decoder(*x_tilda)

        logits_mask = self.head(decoder_output)
        # if masks == None:
        #   return logits_mask
        c0 = c_next
        h0 = h_next

        length = images.shape[1]
        # logits.append(logits_mask)
        logits = []

        for i in range(3,length):
            image = images[:,i,:,:,:]
            x_tilda = self.encoder(image)

            feature_vector = x_tilda[5]

            c_next,h_next = self.convlstm(feature_vector, h0, c0)
            

            x_tilda[5] = h_next
            decoder_output = self.decoder(*x_tilda)

            c0 = c_next
            h0 = h_next
            
            logits_mask = self.head(decoder_output)
            
            #logits_mask = logits_mask.squeeze(0)
            logits.append(logits_mask)
            
        if masks == None:
          return logits_mask
        logits = torch.stack(logits, dim=1)
        
        ce_weights = calculate_weights(masks)
        ce_weights = torch.tensor(ce_weights,dtype=torch.float).to(DEVICE)
        if logits.dim()!= 5:
            logits_mask.unsqueeze(0)
        
        logits = logits.transpose(2,1)
        
        mask = masks[:,3,:,:]
        mask = mask.contiguous().long()
        lovasz = LovaszLoss(mode = 'multiclass', ignore_index=-1)(logits_mask, mask)
        criterion = nn.CrossEntropyLoss(weight = ce_weights, ignore_index=-1)
        ce_logit = criterion(logits_mask, mask)
            # loss += (lovasz + ce_logit) * weights[i]
        loss = (lovasz + ce_logit)



    
        return logits, loss

# class UnetInitialiser(nn.Module):
#     def __init__(self, encoder,convlstm,decoder, head, sizes):
#         super(UnetInitialiser, self).__init__()
#         self.encoder = encoder
#         self.convlstm = convlstm
#         self.decoder = decoder
#         self.head = head


#     def forward(self, images, masks = None):

        
        
#         image_0 = images[:,0,:,:,:]
#         image_1 = images[:,1,:,:,:]
#         image_2 = images[:,2,:,:,:]
#         if masks!=None:
#             mask = masks[:,2,:,:]
#             mask = mask.contiguous().long()

        
#         fv1 = self.encoder(image_0)
#         fv2 = self.encoder(image_1)
#         x_tilda = self.encoder(image_2)
#         features = x_tilda.copy()
        

#         cell_states = []
#         hidden_states = []


#         c0 = fv1[5]
#         h0 = fv2[5]

#         x_tilda = self.encoder(image_2)
#         feature_vector = x_tilda[5]

#         c_next,h_next = self.convlstm(feature_vector, h0, c0)
#         x_tilda[5] = h_next
#         decoder_output = self.decoder(*x_tilda)

#         logits_mask = self.head(decoder_output)

#         c0 = c_next
#         h0 = h_next

#         length = images.shape[1]
#         # logits.append(logits_mask)
#         logits = []

#         for i in range(3,length):
#             image = images[:,i,:,:,:]
#             x_tilda = self.encoder(image)

#             feature_vector = x_tilda[5]

#             c_next,h_next = self.convlstm(feature_vector, h0, c0)
            

#             x_tilda[5] = h_next
#             decoder_output = self.decoder(*x_tilda)

#             c0 = c_next
#             h0 = h_next
            
#             logits_mask = self.head(decoder_output)
            
#             #logits_mask = logits_mask.squeeze(0)
#             logits.append(logits_mask)
            
#         logits = torch.stack(logits, dim=1)
        
#         ce_weights = calculate_weights(masks)
#         ce_weights = torch.tensor(ce_weights,dtype=torch.float).to(DEVICE)
#         if logits.dim()!= 5:
#             logits_mask.unsqueeze(0)
        
#         logits = logits.transpose(2,1)
        
#         mask = masks[:,3,:,:]
#         mask = mask.contiguous().long()
#         lovasz = LovaszLoss(mode = 'multiclass', ignore_index=-1)(logits_mask, mask)
#         criterion = nn.CrossEntropyLoss(weight = ce_weights, ignore_index=-1)
#         ce_logit = criterion(logits_mask, mask)
#             # loss += (lovasz + ce_logit) * weights[i]
#         loss = (lovasz + ce_logit)

#         # for i in range (0, logits.shape[2]):
#         #     logit = logits[:,:,i,:,:] 
            
#         #     mask = masks[:,i+3,:,:]
#         #     mask = mask.contiguous().long()
#         #     lovasz = LovaszLoss(mode = 'multiclass', ignore_index=-1)(logit, mask)
#         #     criterion = nn.CrossEntropyLoss(weight = ce_weights, ignore_index=-1)
#         #     ce_logit = criterion(logit, mask)
#         #     # loss += (lovasz + ce_logit) * weights[i]
#         #     loss = (lovasz + ce_logit)

    
#         return logits, loss

trainset = SegmentationDataset("Training", get_train_augs(), training_images)
validset = SegmentationDataset("Validation", get_valid_augs(), validation_images)
testset = SegmentationDataset("Testing", get_test_augs(), testing_images)

from torch.utils.data import DataLoader
trainloader = DataLoader(trainset, batch_size = 4, shuffle = True,num_workers=2) #every epoch batches shuffles
validloader = DataLoader(validset, batch_size = 4, shuffle = True,num_workers=2)


model = SegmentationModel()
model = model.to(device =DEVICE); #i.e CUDA

model_summary = model.show()
encoder = model_summary.encoder
decoder = model_summary.decoder
head = model_summary.segmentation_head
"""# Loading model"""
sizes = [64,128,256,512]
convlstm  = ConvLSTMCell(input_size = 512, hidden_size = 512)
initialiser = UnetInitialiser(encoder,convlstm,decoder,head, sizes=sizes)
initialiser = initialiser.to(device = DEVICE)

initialiser.load_state_dict(torch.load(f'{DATA_URL}Models/U-net/initialiser_3frames.pt'))

# checkpoint = torch.load(f'{DATA_URL}Models/lstm_unet_scratch_gf_continue_2.pt')
# initialiser.load_state_dict(checkpoint['model_state_dict'])

# encoder = initialiser.encoder
# decoder = initialiser.decoder
# convlstm = initialiser.convlstm
# head = initialiser.head

# new_model = Network(initialiser, encoder,convlstm,decoder, head)
# new_model = new_model.to(device = DEVICE)
# new_model.load_state_dict(torch.load(f'{DATA_URL}Models/new_model_bias.pt'))



import sklearn.metrics as skm
print("testing model 1")
def initialiseDictionary():
  labels = [0,1,2,3,4,5,6,7]
  label_stats = {}
  for label in labels:
    label_stats[label] = {'tp': 0, 'fn': 0, 'fp': 0}
  return label_stats


stats =initialiseDictionary()

labels = [0,1,2,3,4,5,6,7]

# matplotlib.use('tkagg')
images, masks = testset[20]

    # plt.imshow(initial_mask.cpu().squeeze(0))
    # plt.show()
    

# with torch.no_grad():
#   new_model.eval()
#   logits, loss = new_model(images.unsqueeze(0).to(DEVICE), masks)
# # logits = logits.squeeze(0)
# # logits = logits.permute(1,0,2,3)
# i = 0
# for logit in logits:
#   predictions =  torch.nn.functional.softmax(logit, dim=0)
#   pred_labels = torch.argmax(predictions, dim=0)
#   prediction = pred_labels.to('cpu')
#   plt.imshow(pred_labels.detach().cpu().squeeze(0))
#   plt.savefig(f'prediction_new_{i}.png')
#   i+=1
# i = 0







for idx in range (0, len(testset)):
  initialiser.eval()
  images, masks = testset[idx]

  

  logits = initialiser(images.unsqueeze(0).to(DEVICE))

  
  logits = logits.squeeze(0)
  # logits = logits.permute(1,0,2,3)

  predictions =  torch.nn.functional.softmax(logits, dim=0)
  
  pred_labels = torch.argmax(predictions, dim=0)
  

  prediction = pred_labels.to('cpu').flatten().numpy()
 
  ground_truth = masks[3].to('cpu').flatten().numpy()


  conf_matrix = skm.multilabel_confusion_matrix(ground_truth, prediction,labels=labels)
  for label in labels:
    stats[label]['tp'] += conf_matrix[label][1][1] 
    stats[label]['fn'] += conf_matrix[label][1][0] 
    stats[label]['fp'] += conf_matrix[label][0][1]
  
miou = 0
for label in labels:
    tp = stats[label]['tp'] 
    fn = stats[label]['fn'] 
    fp = stats[label]['fp'] 
    iou = tp / ( fp + tp + fn)
    miou+=iou
    print(f"class {label} iou: {iou}")
miou = miou / len(labels)
print(f"miou : {miou}")
