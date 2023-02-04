import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch 


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
      A.HorizontalFlip(p = 0.5),
      A.VerticalFlip(p = 0.5),
      A.GaussianBlur(blur_limit=3, p=0.5),
      A.ColorJitter (brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
      ToTensorV2(),

      
      
  ], additional_targets={'image1': 'image', 'image2': 'image', 'image3': 'image', 'image4': 'image', 'image5': 'image', 'image6': 'image'
  ,'mask1': 'mask', 'mask2': 'mask', 'mask3': 'mask', 'mask4': 'mask', 'mask5': 'mask', 'mask6': 'mask' })


# for validation and test set
def get_valid_augs():
  return A.Compose([
      
      ToTensorV2()
      
  ], additional_targets={'image1': 'image', 'image2': 'image', 'image3': 'image', 'image4': 'image', 'image5': 'image', 'image6': 'image'
  ,'mask1': 'mask', 'mask2': 'mask', 'mask3': 'mask', 'mask4': 'mask', 'mask5': 'mask', 'mask6': 'mask' })

def get_test_augs():
  return A.Compose([
      
      ToTensorV2()
      
  ], additional_targets={'image1': 'image', 'image2': 'image', 'image3': 'image', 'image4': 'image', 'image5': 'image', 'image6': 'image'
  ,'mask1': 'mask', 'mask2': 'mask', 'mask3': 'mask', 'mask4': 'mask', 'mask5': 'mask', 'mask6': 'mask' })

from torch.utils.data import Dataset


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
                                image6=images[6], mask=masks[0], mask1=masks[1], mask2=masks[2], mask3=masks[3], mask4=masks[4], mask5=masks[5],mask6=masks[6])
     
      

      transformed_images.append(data['image'])
      transformed_images.append(data['image1'])
      transformed_images.append(data['image2'])
      transformed_images.append(data['image3'])
      transformed_images.append(data['image4'])
      transformed_images.append(data['image5'])
      transformed_images.append(data['image6'])

      transformed_masks.append(data['mask'])
      transformed_masks.append(data['mask1'])
      transformed_masks.append(data['mask2'])
      transformed_masks.append(data['mask3'])
      transformed_masks.append(data['mask4'])
      transformed_masks.append(data['mask5'])
      transformed_masks.append(data['mask6'])
      
      
      transformed_images = torch.stack(transformed_images)
      transformed_masks = torch.stack(transformed_masks)
      return transformed_images, transformed_masks
    
    
    # masks = getMaskSequence(transformed_images, mask)


    return images, masks

