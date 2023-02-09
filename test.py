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
model = model.to(device =DEVICE) #i.e CUDA


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

convlstm  = ConvLSTMCell(input_size = 512, hidden_size = 512)
new_model = LSTMModel(initializer,encoder,convlstm,decoder, head)
new_model = new_model.to(device = DEVICE)



new_model.load_state_dict(torch.load(f'{DATA_URL}Models/weightedce_025_finetuned.pt'))

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


for idx in range (0, len(testset)):
  new_model.eval()
  images, masks = testset[idx]
  with torch.no_grad():
    logits = new_model(images, masks)
  logits = logits.permute(1,0,2,3)
  
  predictions =  torch.nn.functional.softmax(logits[2], dim=0)
  pred_labels = torch.argmax(predictions, dim=0)


  prediction = pred_labels.to('cpu').flatten().numpy()
 
  ground_truth = masks[3].to('cpu').flatten().numpy()


  conf_matrix = skm.multilabel_confusion_matrix(ground_truth, prediction,labels=labels)
  for label in labels:
    stats[label]['tp'] += conf_matrix[label][1][1] 
    stats[label]['fn'] += conf_matrix[label][1][0] 
    stats[label]['fp'] += conf_matrix[label][0][1]
  
for label in labels:
    tp = stats[label]['tp'] 
    fn = stats[label]['fn'] 
    fp = stats[label]['fp'] 
    iou = tp / ( fp + tp + fn)
    print(f"class {label} iou: {iou}")

def evaluate_continuous_video(unet, model, images):
  stats =initialiseDictionary()
  labels = [0,1,2,3,4,5,6,7]
  for i in range(0, len(images)):
    
    array = np.load(images[i])
    
    masks = array['masks']
    
    image_name = os.path.basename(images[i]).replace(".npz", "")
    
    image_paths = sorted(glob.glob(f"{DATA_URL}Data/Testing_Images/{image_name}*"))
    
    new_images = []
   
    for path in image_paths:
      # i = os.path.basename(path).split("_")[2].split(".")[0].replace("frame", "")
      
        image = cv2.imread(path)
      #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      # i = int(i)
   
      # if i != 10:
       
      #   cv2.imwrite(f"{DATA_URL}Data/Testing_Images/{image_name}_frame0{i}.png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
      # else:
        
      #   cv2.imwrite(f"{DATA_URL}Data/Testing_Images/{image_name}_frame{i}.png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
      
        image = (cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0)
        
        image = image.transpose(2,0,1)
        image = torch.Tensor(image)
        new_images.append(image)

    if len(new_images) != 0:
      
      new_images = np.stack(new_images, axis = 0)
      new_images = torch.Tensor(new_images)

      logit_mask = unet(new_images[0].unsqueeze(0).to(DEVICE))

      predictions = torch.nn.functional.softmax(logit_mask, dim=1)
      initial_mask =torch.argmax(predictions, dim=1)
      logits = model(new_images, initial_mask.squeeze(0))
      
      logits = logits.permute(1,0,2,3)

      predictions =  torch.nn.functional.softmax(logits[9], dim=0)
      pred_labels = torch.argmax(predictions, dim=0)


      prediction = pred_labels.to('cpu').flatten().numpy()
      plt.imshow(pred_labels.cpu())
      plt.show()
        
      ground_truth = masks[3].flatten()
        #print(set(prediction))
      plt.imshow(masks[3])
      plt.show()
      

      conf_matrix = skm.multilabel_confusion_matrix(ground_truth, prediction,labels=labels)
      for label in labels:
        stats[label]['tp'] += conf_matrix[label][1][1] 
        stats[label]['fn'] += conf_matrix[label][1][0] 
        stats[label]['fp'] += conf_matrix[label][0][1]
      
  for label in labels:
    tp = stats[label]['tp'] 
    fn = stats[label]['fn'] 
    fp = stats[label]['fp'] 
    iou = tp / ( fp + tp + fn)
    print(f"class {label} iou: {iou}")
    
  return new_images

#images = evaluate_continuous_video(model, new_model, testing_images)

