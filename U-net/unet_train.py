
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

DATA_URL = "/cs/student/projects1/2019/nsultana/"
DEVICE = 'cuda' #Cuda as using GPU
EPOCHS = 50 #25 training iterations
LR = 0.001 #decay learning rate
IMAGE_SIZE = 320
HEIGHT = 288
WIDTH = 480
BATCH_SIZE = 64
NO_OF_IMAGES = 100
# Images are irregular shape so need to resize

ENCODER = 'resnet34' 
WEIGHTS = 'imagenet' #use weights from imagenet

import os 
import copy
import glob

training_masks = sorted(glob.glob(f"{DATA_URL}Data/Dataset/Training/8_Annotations/*.png"))

training_images = sorted(glob.glob(f"{DATA_URL}Data/Dataset/Training/Images/*.png"))
print("Number of pictures {}".format(len(training_images)))

dict_images = {'masks' : training_masks, 'images' :training_images}

train_df = pd.DataFrame(dict_images)


# images = glob.glob(f"/content/drive/MyDrive/Dissertation/Augmented_images/images_17/*.png")
# masks = glob.glob(f"/content/drive/MyDrive/Dissertation/Augmented_images/masks_17/*.png")
# dict_images = {'masks' : masks, 'images' :images}
# aug_images = pd.DataFrame(dict_images)
# train_df = pd.concat([train_df, aug_images], ignore_index=True)

validation_masks = sorted(glob.glob(f"{DATA_URL}Data/Dataset/Validation/8_Annotations/*.png"))

validation_images = sorted(glob.glob(f"{DATA_URL}Data/Dataset/Validation/Images/*.png"))

print("Number of pictures {}".format(len(validation_images)))

dict_images = {'masks' : validation_masks, 'images' :validation_images}

valid_df = pd.DataFrame(dict_images)

testing_masks = sorted(glob.glob(f"{DATA_URL}Data/Dataset/Testing/8_Annotations/*.png"))

testing_images = sorted(glob.glob(f"{DATA_URL}Data/Dataset/Testing/Images/*.png"))

dict_images = {'masks' : testing_masks, 'images' :testing_images}

test_df = pd.DataFrame(dict_images)


trainset = SegmentationDataset(train_df, get_train_augs())

validset = SegmentationDataset(valid_df, get_valid_augs())

testset = SegmentationDataset(test_df, get_test_augs())

image, mask = testset[0]
plt.imshow(mask)
plt.show()
from torch.utils.data import DataLoader

trainloader = DataLoader(trainset, batch_size = BATCH_SIZE, shuffle = True) #every epoch batches shuffles
validloader = DataLoader(validset, batch_size = BATCH_SIZE, shuffle = True)
image, mask = trainset[90]

model = SegmentationModel()
model.to(DEVICE); #i.e CUDA

optimizer = torch.optim.Adam(model.parameters(), lr = LR)
lambda1 = lambda1 = lambda epoch : pow((1 - epoch / EPOCHS), 0.9)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lambda1) #polynomial


EPOCHS = 30
best_valid_loss = np.Inf

valid_losses = []
train_losses = []

lrs = []


number_epoch_to_save = 5
steps = 5
for epoch in range(0,EPOCHS):


  
  train_loss = train_function(trainloader, model, optimizer)
  valid_loss = eval_function(validloader, model)
  train_losses.append(train_loss)
  valid_losses.append(valid_loss)


  if valid_loss < best_valid_loss: #if best valid loss then upate new model
    torch.save(model.state_dict(), f'{DATA_URL}Models/U-net/batchsize.pt')
    print("Saved model")
    best_valid_loss = valid_loss
  



  if epoch % number_epoch_to_save == 0:

    torch.save({
              'epoch': epoch,
              'model_state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(),
              'loss': valid_loss,
              'best_loss': best_valid_loss,
              'train_loss': train_loss
              }, f'{DATA_URL}Models/U-net/batchsize.pt')
  
  #lrs.append(scheduler.get_last_lr())
  print(f"Epoch : {epoch+1} Train_loss : {train_loss} Valid_loss : {valid_loss} Learning rate:  ")

