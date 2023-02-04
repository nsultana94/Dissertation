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

from networks import SegmentationModel, ConvLSTMCell, LSTMModel, Initializer
from dataloader import get_train_augs, get_test_augs, get_valid_augs, SegmentationDataset
from train_functions import train_function, eval_function
DEVICE = torch.device('cuda') 
# DEVICE = 'cuda' #Cuda as using GPU



EPOCHS = 50 #25 training iterations
LR = 0.00001 #decay learning rate
BATCH_SIZE = 4
HEIGHT = 288
WIDTH = 480
ENCODER = 'resnet34'
WEIGHTS = 'imagenet'
DATA_URL = "/cs/student/projects1/2019/nsultana/"

import random
training_images = (glob.glob(f"{DATA_URL}Data/Training/*.npz"))


testing_images = (glob.glob(f"{DATA_URL}Data/Testing/*.npz"))

validation_images = (glob.glob(f"{DATA_URL}Data/Validation/*.npz"))


"""# Set up model"""

import albumentations as A
from albumentations.pytorch import ToTensorV2


model = SegmentationModel()
model = model.to(device =DEVICE); #i.e CUDA


 """# Set up dataset and data loader"""



trainset = SegmentationDataset("Training", get_train_augs(), training_images)
validset = SegmentationDataset("Validation", get_valid_augs(), validation_images)
testset = SegmentationDataset("Testing", get_test_augs(), testing_images)

from torch.utils.data import DataLoader
trainloader = DataLoader(trainset, batch_size = 4, shuffle = True,num_workers=2) #every epoch batches shuffles
validloader = DataLoader(validset, batch_size = 4, shuffle = True,num_workers=2)



"""# Training model"""

model.load_state_dict(torch.load(f'{DATA_URL}Models/best_model_aug.pt'))
model_summary = model.show()
encoder = model_summary.encoder
initializer = Initializer()
decoder = model_summary.decoder
head = model_summary.segmentation_head

for name,param in encoder.named_parameters():
  param.requires_grad = False

for name,param in decoder.named_parameters():
  param.requires_grad = False

convlstm  = ConvLSTMCell(input_size = 512, hidden_size = 512)
new_model = LSTMModel(initializer,encoder,convlstm,decoder, head)
new_model = new_model.to(device = DEVICE)



# #new_model.load_state_dict(torch.load(f'{DATA_URL}Models/fine_tuned_2.pt'))

# # checkpoint = torch.load(f'{DATA_URL}Models/conv_lstm_1_current.pt')
# # new_model.load_state_dict(checkpoint['model_state_dict'])

# # epoch_start = checkpoint['epoch']
# # loss = checkpoint['loss']
# # best_valid_loss = checkpoint['best_loss']
# # training_loss = checkpoint['train_loss']

EPOCHS = 50
best_valid_loss = np.inf

valid_losses = []
train_losses = []

lrs = []

#0.12911548332047107 epoch 28 
LR = 0.00001
optimizer = torch.optim.Adam(new_model.parameters(), lr = LR)

#optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
number_epoch_to_save = 5

for epoch in range(0,EPOCHS):


  train_loss = train_function(trainloader, new_model, optimizer)
  valid_loss = eval_function(validloader, new_model)
  train_losses.append(train_loss)
  valid_losses.append(valid_loss)


  if valid_loss < best_valid_loss: #if best valid loss then upate new model
    torch.save(new_model.state_dict(), f'{DATA_URL}Models/lstm_weightedce_lovasz.pt')
    print("Saved model")
    best_valid_loss = valid_loss

  #scheduler.step()



  if epoch % number_epoch_to_save == 0:

    torch.save({
              'epoch': epoch,
              'model_state_dict': new_model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(),
              'loss': valid_loss,
              'best_loss': best_valid_loss,
              'train_loss': train_loss
              }, f'{DATA_URL}Models/conv_lstm_1_current.pt')
  
#   #lrs.append(scheduler.get_last_lr())
  
  print(f"Epoch : {epoch+1} Train_loss : {train_loss} Valid_loss : {valid_loss} Learning rate:  ")
  

# """# Only training conv lstm and decoder + encoder"""



# # encoder = new_model.getEncoder()
# # decoder = new_model.getDecoder()

# # for name,param in encoder.named_parameters():
# #   print(param.requires_grad)
# #   break

# # new_model.load_state_dict(torch.load(f'{DATA_URL}Models/conv_lstm_triplet_loss_continued.pt'))

# # initializer = new_model.getInitializer()
# # encoder = new_model.getEncoder()
# # decoder = new_model.getDecoder()
# # convlstm = new_model.getLSTM()
# # for name,param in initializer.named_parameters():
# #   param.requires_grad = False

# # for name,param in convlstm.named_parameters():
# #   param.requires_grad = True

# # for name,param in encoder.named_parameters():
# #   param.requires_grad = True

# # for name,param in decoder.named_parameters():
# #   param.requires_grad = True

# # train_losses = []
# # valid_losses = []


# # LR = 0.00001
# # optimizer = torch.optim.Adam(new_model.parameters(), lr = LR)





