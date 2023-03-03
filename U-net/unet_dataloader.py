import albumentations as A
from torch.utils.data import Dataset
HEIGHT = 288
WIDTH = 480
import cv2
import torch

# increases more images for training dataset
from albumentations.pytorch import ToTensorV2

def get_train_augs():
  return A.Compose([
      # A.RandomCrop(height=HEIGHT, width=WIDTH, p=0.5),
      A.Resize(height=HEIGHT, width=WIDTH),
      A.HorizontalFlip(p = 0.5),
      A.VerticalFlip(p = 0.5),
      A.GaussianBlur(blur_limit=3, p=0.5),
      A.ColorJitter (brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
      #A.Normalize(mean=0.5, std=1.0),
      #A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
      ToTensorV2()
      
  ])



# for validation and test set
def get_valid_augs():
  return A.Compose([
      A.Resize(height=HEIGHT, width=WIDTH),
      #A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
      ToTensorV2()
      
  ])

def get_test_augs():
  return A.Compose([
      A.Resize(height=HEIGHT, width=WIDTH),
      #A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
      ToTensorV2()
      
  ])

class SegmentationDataset(Dataset):

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
