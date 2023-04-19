import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch 

import cv2

import numpy as np 


HEIGHT = 288
WIDTH = 480

DEVICE = torch.device('cuda')
EPOCHS = 50 #25 training iterations
LR = 0.00001 #decay learning rate
BATCH_SIZE = 4
HEIGHT = 288
WIDTH = 480
ENCODER = 'resnet34'
WEIGHTS = 'imagenet'
DATA_URL = "/cs/student/projects1/2019/nsultana/"


def get_train_augs():
  return A.Compose([
      A.RandomCrop(height=192, width=320, p=0.2),
      A.Resize(height = HEIGHT, width = WIDTH),
      A.HorizontalFlip(p = 0.3),
      A.VerticalFlip(p = 0.3),
      A.GaussianBlur(blur_limit=3, p=0.3),
      A.ColorJitter (brightness=0.2, contrast=0.2, saturation=0.2, p=0.3),
      ToTensorV2(),

      
      
  ], additional_targets={'image1': 'image', 'image2': 'image', 'image3': 'image', 'image4': 'image', 'image5': 'image', 'image6': 'image','image7': 'image', 'image8': 'image', 'image9': 'image', 'image10': 'image'
  ,'mask1': 'mask', 'mask2': 'mask', 'mask3': 'mask', 'mask4': 'mask', 'mask5': 'mask', 'mask6': 'mask', 'mask7': 'mask', 'mask8': 'mask', 'mask9': 'mask', 'mask10': 'mask'})

#'image7': 'image', 'image8': 'image', 'image9': 'image', 'image11': 'image'
#'mask7': 'mask', 'mask8': 'mask', 'mask9': 'mask', 'mask10': 'mask'
# for validation and test set
def get_valid_augs():
  return A.Compose([
      
      ToTensorV2()
      
  ], additional_targets={'image1': 'image', 'image2': 'image', 'image3': 'image', 'image4': 'image', 'image5': 'image', 'image6': 'image','image7': 'image', 'image8': 'image', 'image9': 'image', 'image10': 'image'
  ,'mask1': 'mask', 'mask2': 'mask', 'mask3': 'mask', 'mask4': 'mask', 'mask5': 'mask', 'mask6': 'mask', 'mask7': 'mask', 'mask8': 'mask', 'mask9': 'mask', 'mask10': 'mask' })

def get_test_augs():
  return A.Compose([
      
      ToTensorV2()
      
  ], additional_targets={'image1': 'image', 'image2': 'image', 'image3': 'image', 'image4': 'image', 'image5': 'image', 'image6': 'image','image7': 'image', 'image8': 'image', 'image9': 'image', 'image10': 'image'
  ,'mask1': 'mask', 'mask2': 'mask', 'mask3': 'mask', 'mask4': 'mask', 'mask5': 'mask', 'mask6': 'mask', 'mask7': 'mask', 'mask8': 'mask', 'mask9': 'mask', 'mask10': 'mask' })

from torch.utils.data import Dataset
def get_test_augs_unet():
  return A.Compose([
      A.Resize(height=HEIGHT, width=WIDTH),
      #A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
      ToTensorV2()
      
  ])

class SegmentationDataset(Dataset):

  def __init__(self, split, augmentations, sequence):
   # self.df = df
    self.split = split
    self.augmentations = augmentations
    self.sequence = sequence
  
  def __len__(self):
    return len(self.sequence)
  

  def __getitem__(self, idx):
    # image_name = self.sequence[idx]
    # images, masks = generateImagesMasks(image_name, self.split)
    path = self.sequence[idx]
    array = np.load(path)
    
    images = array['images']
    masks = array['masks']
    
    #print(set(masks[0].flatten()))

    images = np.transpose(images, (0, 2,3,1))
    transformed_images = []
    transformed_masks = []
  
    
   

    if self.augmentations:
      data = self.augmentations(image=images[0],image1=images[1],image2=images[2],image3=images[3],image4=images[4],image5=images[5],
                                image6=images[6],image7=images[7],image8=images[8],image9=images[9],image10=images[10],
                                mask=masks[0], mask1=masks[1], mask2=masks[2], mask3=masks[3], mask4=masks[4], mask5=masks[5],mask6=masks[6],
                                mask7=masks[7], mask8=masks[8], mask9=masks[9], mask10=masks[10])
     
      #image7=images[7],image8=images[8],image9=images[9],image10=images[10]
      #mask7=masks[7], mask8=masks[8], mask9=masks[9], mask10=masks[10]

      transformed_images.append(data['image'])
      transformed_images.append(data['image1'])
      transformed_images.append(data['image2'])
      transformed_images.append(data['image3'])
      transformed_images.append(data['image4'])
      transformed_images.append(data['image5'])
      transformed_images.append(data['image6'])
      transformed_images.append(data['image7'])
      transformed_images.append(data['image8'])
      transformed_images.append(data['image9'])
      transformed_images.append(data['image10'])


      transformed_masks.append(data['mask'])
      transformed_masks.append(data['mask1'])
      transformed_masks.append(data['mask2'])
      transformed_masks.append(data['mask3'])
      transformed_masks.append(data['mask4'])
      transformed_masks.append(data['mask5'])
      transformed_masks.append(data['mask6'])
      transformed_masks.append(data['mask7'])
      transformed_masks.append(data['mask8'])
      transformed_masks.append(data['mask9'])
      transformed_masks.append(data['mask10'])

      

      transformed_images = torch.stack(transformed_images)
      transformed_masks = torch.stack(transformed_masks)
      # transformed_masks = transformed_masks[0:7:3]
      # transformed_images = transformed_images[0:7:3]
      # transformed_masks = transformed_masks[1:8:2]
      # transformed_images = transformed_images[1:8:2]
      # transformed_masks = transformed_masks[2:10]
      # transformed_images = transformed_images[2:10]
      # transformed_masks = transformed_masks[2:8]
      # transformed_images = transformed_images[2:8]
      # transformed_masks = transformed_masks[1:8:2]
      # transformed_images = transformed_images[1:8:2]

      transformed_masks = transformed_masks[4:8]
      transformed_images = transformed_images[4:8]


      return transformed_images, transformed_masks
    
    
    # masks = getMaskSequence(transformed_images, mask)


    return images, masks

class SegmentationDatasetUnet(Dataset):

  def __init__(self, df, augmentations):
    self.df = df
    self.augmentations = augmentations 
  
  def __len__(self):
    return len(self.df)
  

  def __getitem__(self, idx):
    row = self.df.iloc[idx]

    image_path = row.images
    mask_path = row.masks 
   
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
     #(h, w, c) whre c is channel 
    # channel is 1 as gray scale
    #print(mask.shape)
    #mask = np.expand_dims(mask, axis = -1) #expand last axis)
   
    if self.augmentations:
      data = self.augmentations(image = image, mask = mask)
     
      image = data['image']
      mask = data['mask']

    #(h,w,c) -> (c,h, w) which is what pytorch uses

    #image = np.transpose(image, (2, 0, 1)).astype(np.float32)
    #mask = np.transpose(mask, (2, 0, 1)).astype(np.float32)
    #print(mask.shape)
    
    #print(mask.shape)
  #converts numpy array to Tensor
    image = torch.Tensor(image) / 255.0 #ensure between 0 - 1 -> normalization
    mask = torch.Tensor(mask)  #either 0 or 1
    
    
    return image, mask
