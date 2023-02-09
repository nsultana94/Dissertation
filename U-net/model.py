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
        activation = None #no sigmoid or softmax

    )
  
  def show(self):
    return self.architecture

  def forward(self, images, masks = None):
    logits = self.architecture(images) #probabilities / predictions

    
    if masks!= None:

      #class_weights=torch.tensor(weights,dtype=torch.float).to(DEVICE)
      lovasz = LovaszLoss(mode = 'multiclass')(logits, masks)

      loss = lovasz
      return logits, loss

    return logits
  

