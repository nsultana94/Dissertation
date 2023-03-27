import sys
sys.path.append("..")


from unet_train_functions import eval_function, train_function
from model import SegmentationModel
from unet_dataloader import get_test_augs
from unet_dataloader import get_valid_augs
from unet_dataloader import get_train_augs
from unet_dataloader import SegmentationDataset
from networks import SegmentationModel, ConvLSTMCell, Initializer
from segmentation_models_pytorch.losses import DiceLoss, LovaszLoss, FocalLoss, JaccardLoss


import logging 
logging.basicConfig(filename="std.log", 
					format='%(asctime)s %(message)s', 
					filemode='w')

logger=logging.getLogger() 
logger.setLevel(logging.DEBUG) 

import torch 
import cv2

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from torch import nn
import sklearn.metrics as skm

from tqdm import tqdm

import os 
import copy
import glob

DATA_URL = "/cs/student/projects1/2019/nsultana/"
DEVICE = 'cuda' #Cuda as using GPU
EPOCHS = 50 #25 training iterations
LR = 0.001 #decay learning rate
IMAGE_SIZE = 320
HEIGHT = 288
WIDTH = 480
BATCH_SIZE = 4
# Images are irregular shape so need to resize

ENCODER = 'resnet34' 
WEIGHTS = 'imagenet' #use weights from imagenet

import os 
import copy
import glob

train_df = pd.read_csv(f"{DATA_URL}Data/Dataset/train_df_2.csv")
valid_df = pd.read_csv(f"{DATA_URL}Data/Dataset/valid_df_2.csv")
test_df = pd.read_csv(f"{DATA_URL}Data/Dataset/test_df_2.csv")


trainset = SegmentationDataset(train_df, get_train_augs())

validset = SegmentationDataset(valid_df, get_valid_augs())

testset = SegmentationDataset(test_df, get_test_augs())

from torch.utils.data import DataLoader

trainloader = DataLoader(trainset, batch_size = BATCH_SIZE, shuffle = True) #every epoch batches shuffles
validloader = DataLoader(validset, batch_size = BATCH_SIZE, shuffle = True)

model = SegmentationModel()
model.to(DEVICE); #i.e CUDA

# model.load_state_dict(torch.load(f'{DATA_URL}Models/unet_with_lstm.pt'))
model_summary = model.show()
encoder = model_summary.encoder
decoder = model_summary.decoder
head = model_summary.segmentation_head

for name,param in encoder.named_parameters():
  param.requires_grad = False


for name,param in head.named_parameters():
  param.requires_grad = False


for name,param in decoder.named_parameters():
  param.requires_grad = False


convlstm  = ConvLSTMCell(input_size = 512, hidden_size = 512, height=9, width=15)
new_model = Initializer(encoder,convlstm,decoder, head)
new_model = new_model.to(device = DEVICE)


def initialiseDictionary():
  labels = [0,1,2,3,4,5,6,7]
  label_stats = {}
  for label in labels:
    label_stats[label] = {'tp': 0, 'fn': 0, 'fp': 0}
  return label_stats


new_model.load_state_dict(torch.load(f'{DATA_URL}Models/U-net/unet_paper_structure_2.pt'))
stats =initialiseDictionary()

# checkpoint = torch.load(f'{DATA_URL}Models/U-net/batchsize_continue.pt')
# new_model.load_state_dict(checkpoint['model_state_dict'])


for name,param in encoder.named_parameters():
  logger.info(name)

logger.info(f"model")

for name,param in new_model.named_parameters():
  logger.info(name)

labels = [0,1,2,3,4,5,6,7]
for idx in range (0, len(testset)):
  
  image, mask = testset[idx]
  new_model.eval()
  h_next, c_next , logits_mask = new_model(image.to(DEVICE).unsqueeze(0)) # (c, h, w ) -> (1, c, h , w)
  predictions = torch.nn.functional.softmax(logits_mask, dim=1)
  pred_labels = torch.argmax(predictions, dim=1)

  prediction = pred_labels.to('cpu').flatten().numpy()
  ground_truth = mask.to('cpu').flatten().numpy()


  #tp, fp, fn, tn = smp.metrics.get_stats(prediction, ground_truth, mode='multiclass', num_classes = 8)

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






