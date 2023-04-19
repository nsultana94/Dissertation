# -*- coding: utf-8 -*-

import logging 

logging.basicConfig(filename="results.log", 
                    format='%(asctime)s %(message)s', 
                    filemode='w')

logger=logging.getLogger() 
logger.setLevel(logging.DEBUG) 
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

from networks import SegmentationModel, ConvLSTMCell, LSTMModel, Initializer, Initializer2
from dataloader import get_train_augs, get_test_augs, get_valid_augs, SegmentationDataset
from train_functions import train_function, eval_function
DEVICE = torch.device('cuda') 
# DEVICE = 'cuda' #Cuda as using GPU
import pandas as pd
from sklearn.metrics import jaccard_score, accuracy_score
import matplotlib
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

class Network(nn.Module):
        def __init__(self, initializer, encoder,convlstm,decoder, head):
                super(Network, self).__init__()
                self.initializer = initializer
                self.encoder = encoder
                self.decoder = decoder
                self.head = head
                self.convlstm = convlstm
       
                #self.convlstms = nn.ModuleList([ConvLSTMCell(input_size = 512, hidden_size = 512, height=9, width=15) for i in range(num_layers)])
        
        def forward(self, images, masks = None):

            
            logits = []
            if images.dim()!=5:
                images = images.unsqueeze(0)

            c_original, feature_vector, c0,h0, logits_mask = self.initializer(images)
            h_original = h0
            c_original = c0
    
            cell_states = []
            #logits.append(logit_mask)
            length = images.shape[1]
        
            # logits.append(logits_mask)
            
            for i in range(3,length):
                
                image = images[:,i,:,:,:]
                x_tilda = self.encoder(image)

                feature_vector = x_tilda[5]

                h_next,c_next = self.convlstm(feature_vector, h0, c0)
                decoder = self.decoder

                x_tilda[5] = h_next
                decoder_output = decoder(*x_tilda)

                c0 = c_next
                h0 = h_next
                
                
                logits_mask = self.head(decoder_output)
                
                #logits_mask = logits_mask.squeeze(0)
                logits.append(logits_mask)
                
            
            logits = torch.stack(logits, dim=0)
            
            if logits.dim()!= 5:
                logits_mask.unsqueeze(0)
            
            logits = logits.transpose(2,1) # makes in the dimension of classes, no of images, width, height so 8,7,480,288
             
            return logits, h0, c0, feature_vector, c_original


class ConvLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,
                 encoder, decoder, head):
        super(ConvLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        layers = []
        for i in range(0, self.num_layers):
            layers.append(ConvLSTMCell(input_size = self.input_size, hidden_size = self.hidden_size))

        self.layers = nn.ModuleList(layers)
        self.encoder = encoder
        self.decoder = decoder
        self.head = head

    def forward(self, images, masks = None):
        length = images.shape[1] - 2
        image_0 = images[:,0,:,:,:]
        image_1 = images[:,1,:,:,:]
        image_2 = images[:,2,:,:,:]
        if masks!=None:
            mask = masks[:,2,:,:]
            mask = mask.contiguous().long()
            ce_weights = calculate_weights(masks)
            ce_weights = torch.tensor(ce_weights,dtype=torch.float).to(DEVICE)


        fv1 = self.encoder(image_0)
        fv2 = self.encoder(image_1)
        x_tilda = self.encoder(image_2)



        c0 = fv1[5]
        h0 = fv2[5]
        feature_vectors = []
        x_tilda = self.encoder(image_2)
        feature_vector = x_tilda[5]
        
        layer_output_list = []
        x_tildas = []
        for i in range(0, length):
            
            x_tilda = self.encoder(images[:,i+2,:,:,:])
            feature_vectors.append(x_tilda[5])
            
            x_tildas.append(x_tilda)
            feature_vector = x_tilda[5]
            

        feature_vectors = torch.stack(feature_vectors)
        cur_layer_input = feature_vectors

        for layer_idx in range(self.num_layers):

            h = h0
            c = c0
            output_inner = []
            for i in range(0,length):
                
                h, c = self.layers[layer_idx](input_=cur_layer_input[i,:, :, :, :],
                                                    hiddenState=h, cellState=c)
                output_inner.append(h)
               

            layer_output = torch.stack(output_inner, dim=0)
            cur_layer_input = layer_output
            

            # layer_output_list.append(layer_output)
            # last_state_list.append([h, c])
        logits = []
        
        weights = [0.1, 0.25, 0.5, 0.25, 0.1]
        loss = 0
        logits = []
        for i in range(0, len(x_tildas)):
            
            h = cur_layer_input[i,:,:,:,:]
            x_tilda = x_tildas[i]
            x_tilda[5] = h
            decoder_output = self.decoder(*x_tilda)
            logits_mask = self.head(decoder_output)
            if masks!=None:
                mask = masks[:,i+2,:,:]
                mask = mask.contiguous()

                

                lovasz = LovaszLoss(mode = 'multiclass', ignore_index=-1)(logits_mask, mask)
            
                criterion = nn.CrossEntropyLoss(weight = ce_weights, ignore_index=-1)
                ce_logit = criterion(logits_mask, mask)
                loss = loss + (lovasz + ce_logit) * weights[i]
            logits.append(logits_mask)

            # hidden_loss = nn.MSELoss()
            # cell_loss = nn.MSELoss()

            # hidden_state_loss = hidden_loss(h, h0) 
            # cell_state_loss = cell_loss(c, c0)

        # loss = (lovasz_loss + cross_entropy_loss + (cell_state_loss) + hidden_state_loss)
        return loss, logits

trainset = SegmentationDataset("Training", get_train_augs(), training_images)
validset = SegmentationDataset("Validation", get_valid_augs(), validation_images)
testset = SegmentationDataset("Testing", get_test_augs(), testing_images)

from torch.utils.data import DataLoader
trainloader = DataLoader(trainset, batch_size = 4, shuffle = True,num_workers=2) #every epoch batches shuffles
validloader = DataLoader(validset, batch_size = 4, shuffle = True,num_workers=2)



"""# Loading model"""
model_summary = model.show()
encoder = model_summary.encoder
decoder = model_summary.decoder
head = model_summary.segmentation_head

sizes = [64,128,256,512]
convlstm  = ConvLSTMCell(input_size = 512, hidden_size = 512)
# initialiser = UnetInitialiser(encoder,convlstm,decoder,head, sizes=sizes)
initialiser = ConvLSTM(512, 512, 2, encoder, decoder, head)

initialiser = initialiser.to(device = DEVICE)

initialiser.load_state_dict(torch.load(f'{DATA_URL}Models/U-net/4layers_finetuned.pt'))


for name,param in convlstm.named_parameters():
    logger.info(f"{name}, {param.requires_grad}")
    logger.info(f"{param}")

import sklearn.metrics as skm
print("testing model 1")
def initialiseDictionary():
  labels = [0,1,2,3,4,5,6,7]
  label_stats = {}
  for label in labels:
    label_stats[label] = {'tp': 0, 'fn': 0, 'fp': 0, 'tn':0}
  return label_stats


stats =initialiseDictionary()

labels = [0,1,2,3,4,5,6,7]



ious = []

gts = []
preds = []

for idx in range (0, len(testset)):
  initialiser.eval()
  images, masks = testset[idx]

  with torch.no_grad():

    loss, logits = initialiser(images.unsqueeze(0).to(DEVICE))

    # logits, h_next, c_next, feature_vector, c_original = new_model(images.unsqueeze(0).to(DEVICE))
    hidden_loss = nn.L1Loss()


  predictions =  torch.nn.functional.softmax(logits[1].squeeze(0), dim=0)
  
  pred_labels = torch.argmax(predictions, dim=0)
  

  prediction = pred_labels.to('cpu').flatten().numpy()
  preds.append(prediction)
  ground_truth = masks[3].to('cpu').flatten().numpy()
  gts.append(ground_truth)

  conf_matrix = skm.multilabel_confusion_matrix(ground_truth, prediction,labels=labels)
  for label in labels:
    stats[label]['tp'] += conf_matrix[label][1][1] 
    stats[label]['fn'] += conf_matrix[label][1][0] 
    stats[label]['fp'] += conf_matrix[label][0][1]
    stats[label]['tn'] += conf_matrix[label][0][0]


miou = 0
for label in labels:
    tp = stats[label]['tp'] 
    fn = stats[label]['fn'] 
    fp = stats[label]['fp'] 
    tn = stats[label]['tn'] 
    iou = tp / ( fp + tp + fn)
    pa = (tp + tn) / ( fp + tp + fn + tn)
    miou+=iou
    print(f"class {label} iou: {iou} pa: {pa}")
miou = miou / len(labels)
print(f"miou : {miou}")

gts = np.stack(gts)
preds = np.stack(preds)
preds = torch.Tensor(preds)
gts = torch.Tensor(gts)

from torchmetrics.classification import MulticlassJaccardIndex

metric = MulticlassJaccardIndex(num_classes=8, average = None)
print(metric(preds, gts))







def evaluate_continuous_video(model, images):
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
      model.eval()
      
      # logit_mask = unet(new_images[0].unsqueeze(0).to(DEVICE))
      # print(logit_mask.shape)
      
      # # plt.imshow(initial_mask.cpu().squeeze(0))
      # # plt.show()
      
      # initial_mask = initial_mask.unsqueeze(0)
      # print(initial_mask.shape)
      
      new_images = new_images.to(device = DEVICE)
      loss, logits = model(images.unsqueeze(0).to(DEVICE))
      logits = logits.squeeze(0)

    
      logits = logits.permute(1,0,2,3)

      predictions =  torch.nn.functional.softmax(logits[-1].squeeze(0), dim=0)
      pred_labels = torch.argmax(predictions, dim=0)


      prediction = pred_labels.to('cpu').flatten().numpy()

      # for logit in logits:
        
      #   predictions =  torch.nn.functional.softmax(logit, dim=0)
      #   pred_labels = torch.argmax(predictions, dim=0)
      #   prediction = pred_labels.to('cpu').flatten().numpy()
      #   plt.imshow(pred_labels.cpu())
      #   plt.show()
    
      ground_truth = masks[-1].flatten()

      

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
    miou+= iou
    print(f"class {label} iou: {iou}")
  miou = miou / len(labels)
  print(f"miou : {miou}")
  return new_images

# images = evaluate_continuous_video(unet, new_model, testing_images)