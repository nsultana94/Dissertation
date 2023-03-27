# efficientnet as encoder
# segmentation model as unet
# need segmentation loss - using DiceLoss 

from torch import nn # neural netowrk 
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss, LovaszLoss, FocalLoss, JaccardLoss
import torch
ENCODER = 'resnet34' 
WEIGHTS = 'imagenet' #use weights from imagenet
DEVICE = 'cuda' #Cuda as using GPU

from math import log

class SegmentationModel(nn.Module):

  def __init__(self):
    super(SegmentationModel, self).__init__() 

    self.architecture = smp.Unet(
        encoder_name = ENCODER,
        encoder_weights = WEIGHTS, 
        in_channels = 3, #Input is RGB
        classes = 8, # binary segmentation problem 
        activation = None,#no sigmoid or softmax
        aux_params = {'dropout': 0.2, 'classes' :8}
        

    )
  
  def show(self):
    return self.architecture

  def forward(self, images, masks = None):
    logits = self.architecture(images) #probabilities / predictions

    
    if masks!= None:
      
      weights = [4.116647326424263, 24.600245093614593, 191.78790779880697, 240.94195047235274, 7.334747505863925, 10.620043927212807, 2.219872768361696, 38.32265526553685]
      class_weights=torch.tensor(weights,dtype=torch.float).to(DEVICE)
      loss_value = LovaszLoss(mode = 'multiclass')(logits, masks)
      loss_fn= nn.CrossEntropyLoss(weight = class_weights)
      #ce = loss_fn(logits, masks)
      loss = loss_value
      return logits, loss

    return logits
  

