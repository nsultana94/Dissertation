import warnings

from unet_train_functions import eval_function, train_function
from model import SegmentationModel
from unet_dataloader import get_test_augs
from unet_dataloader import get_valid_augs
from unet_dataloader import get_train_augs
from unet_dataloader import SegmentationDataset
import torch 
import cv2

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 


from tqdm import tqdm

import os 
import copy
import glob
warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"
import sklearn.metrics as skm

from sklearn.metrics import jaccard_score, accuracy_score

DATA_URL = "/cs/student/projects1/2019/nsultana/"
DEVICE = 'cuda' #Cuda as using GPU
EPOCHS = 50 #25 training iterations
LR = 0.001 #decay learning rate
IMAGE_SIZE = 320
HEIGHT = 288
WIDTH = 480
BATCH_SIZE = 48
NO_OF_IMAGES = 100


train_df = pd.read_csv(f"{DATA_URL}Data/Dataset/train_df_1.csv")
valid_df = pd.read_csv(f"{DATA_URL}Data/Dataset/valid_df_1.csv")
test_df = pd.read_csv(f"{DATA_URL}Data/Dataset/test_df_1.csv")

trainset = SegmentationDataset(train_df, get_train_augs())

validset = SegmentationDataset(valid_df, get_valid_augs())

testset = SegmentationDataset(test_df, get_test_augs())



model = SegmentationModel()
model.to(DEVICE); #i.e CUDA

def initialiseDictionary():
  labels = [0,1,2,3,4,5,6,7]
  label_stats = {}
  for label in labels:
    label_stats[label] = {'tp': 0, 'fn': 0, 'fp': 0}
  return label_stats


model.load_state_dict(torch.load(f'{DATA_URL}Models/U-net/batchsize.pt'))
stats =initialiseDictionary()

labels = [0,1,2,3,4,5,6,7]
for idx in range (0, len(testset)):
  
  image, mask = testset[idx]
  model.eval()
  logits_mask = model(image.to(DEVICE).unsqueeze(0)) # (c, h, w ) -> (1, c, h , w)
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


