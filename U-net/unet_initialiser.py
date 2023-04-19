import sys
sys.path.append("..")


from unet_train_functions import eval_function, train_function
from model import SegmentationModel
from unet_dataloader import get_test_augs
from unet_dataloader import get_valid_augs
from unet_dataloader import get_train_augs
from unet_dataloader import SegmentationDataset
from networks import ConvLSTMCell, Initializer
from segmentation_models_pytorch.losses import DiceLoss, LovaszLoss, FocalLoss, JaccardLoss

import torch 
import cv2

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from torch import nn


from tqdm import tqdm

import os 
import copy
import glob


import logging 
logging.basicConfig(filename="unet5.log", 
					format='%(asctime)s %(message)s', 
					filemode='w')

logger=logging.getLogger() 
logger.setLevel(logging.DEBUG) 

DATA_URL = "/cs/student/projects1/2019/nsultana/"
DEVICE = 'cuda' #Cuda as using GPU
EPOCHS = 100 #25 training iterations
LR = 0.00001 #decay learning rate
IMAGE_SIZE = 320
HEIGHT = 288
WIDTH = 480
BATCH_SIZE = 16
# Images are irregular shape so need to resize

ENCODER = 'resnet34' 
WEIGHTS = 'imagenet' #use weights from imagenet

import os 
import copy
import glob

# train_df = pd.read_csv(f"{DATA_URL}Data/Dataset/train_df_1.csv")
# valid_df = pd.read_csv(f"{DATA_URL}Data/Dataset/valid_df_1.csv")
# test_df = pd.read_csv(f"{DATA_URL}Data/Dataset/test_df_1.csv")

training_masks = sorted(glob.glob(f"{DATA_URL}Data/Dataset/Training/8_Annotations/*.png"))

training_images = sorted(glob.glob(f"{DATA_URL}Data/Dataset/Training/Images/*.png"))
print("Number of pictures {}".format(len(training_images)))

dict_images = {'masks' : training_masks, 'images' :training_images}

train_df = pd.DataFrame(dict_images)




images = sorted(glob.glob(f"{DATA_URL}updated_data/Data/Augmented_images/images_8/*.png"))
masks = sorted(glob.glob(f"{DATA_URL}updated_data/Data/Augmented_images/masks_8/*.png"))
# images = glob.glob(f"/content/drive/MyDrive/Dissertation/Augmented_images/images_17/*.png")
# masks = glob.glob(f"/content/drive/MyDrive/Dissertation/Augmented_images/masks_17/*.png")
dict_images = {'masks' : masks, 'images' :images}
aug_images = pd.DataFrame(dict_images)
train_df = pd.concat([train_df, aug_images], ignore_index=True)



validation_masks = sorted(glob.glob(f"{DATA_URL}Data/Dataset/Validation/8_Annotations/*.png"))

validation_images = sorted(glob.glob(f"{DATA_URL}Data/Dataset/Validation/Images/*.png"))

print("Number of pictures {}".format(len(validation_images)))

dict_images = {'masks' : validation_masks, 'images' :validation_images}

valid_df = pd.DataFrame(dict_images)

testing_masks = sorted(glob.glob(f"{DATA_URL}Data/Dataset/Testing/8_Annotations/*.png"))

testing_images = sorted(glob.glob(f"{DATA_URL}Data/Dataset/Testing/Images/*.png"))

dict_images = {'masks' : testing_masks, 'images' :testing_images}

test_df = pd.DataFrame(dict_images)





train_df = train_df.sample(frac=1).reset_index(drop=True)
valid_df = valid_df.sample(frac=1).reset_index(drop=True)
test_df = test_df.sample(frac=1).reset_index(drop=True)

train_df.to_csv(f"{DATA_URL}Data/Dataset/train_df_2.csv")


valid_df.to_csv(f"{DATA_URL}Data/Dataset/valid_df_2.csv")


test_df.to_csv(f"{DATA_URL}Data/Dataset/test_df_2.csv")

trainset = SegmentationDataset(train_df, get_train_augs())

validset = SegmentationDataset(valid_df, get_valid_augs())

testset = SegmentationDataset(test_df, get_test_augs())

from torch.utils.data import DataLoader

trainloader = DataLoader(trainset, batch_size = BATCH_SIZE, shuffle = True) #every epoch batches shuffles
validloader = DataLoader(validset, batch_size = BATCH_SIZE, shuffle = True)

model = SegmentationModel()
model.to(DEVICE); #i.e CUDA


model_summary = model.show()
encoder = model_summary.encoder
decoder = model_summary.decoder
head = model_summary.segmentation_head




convlstm  = ConvLSTMCell(input_size = 512, hidden_size = 512, height=9, width=15)
#convlstm  = ConvLSTMCell(input_size = 512, hidden_size = 512)
new_model = Initializer(encoder,convlstm,decoder, head)
new_model = new_model.to(device = DEVICE)


LR = 0.0001
optimizer = torch.optim.Adam(new_model.parameters(), lr = LR)


lambda1 = lambda1 = lambda epoch : pow((1 - epoch / EPOCHS), 0.9)

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lambda1) #polynomial


best_valid_loss = np.Inf

valid_losses = []
train_losses = []

lrs = []


number_epoch_to_save = 5
steps = 5

def calculate_weights(masks):
    labels = [0,1,2,3,4,5,6,7]
    weights_list = dict.fromkeys(labels,0)
    new_weights = []
    max_weights = dict.fromkeys(labels,0)
    ground_truth = masks.to('cpu').flatten().numpy().astype(int)

    total_weight = 0
    for label in labels:
        count = (ground_truth == label).sum()
        weights_list[label] += count
        total_weight +=count
    
    for label in labels:
        weight = weights_list[label]
        if weight == 0:
            new_weights.append(0)
        else:
            new_weights.append(1 - (weight/total_weight))

    return new_weights

def train_function(data_loader, model, optimizer):

  model.train()
  total_loss = 0.0

  for images, masks in tqdm(data_loader):

    images = images.to(DEVICE)
    masks = masks.to(DEVICE, dtype=torch.long)

    c_next,h_next, logit = model(images)
    
    # make sure gradients are 0
    optimizer.zero_grad()
    ce_weights = calculate_weights(masks)
    ce_weights = torch.tensor(ce_weights,dtype=torch.float).to(DEVICE)
    loss_per_frame = LovaszLoss(mode = 'multiclass', ignore_index=-1)(logit, masks)
    criterion = nn.CrossEntropyLoss(weight = ce_weights, ignore_index=-1)
    
    ce_logit = criterion(logit, masks)
    # ce_logit = ce_logit * 0.25
    
    loss = (loss_per_frame + ce_logit) 
    #loss = loss_per_frame

    
    loss.backward() #backpropagation

    optimizer.step() #update weights

    total_loss += loss.item()

  return total_loss / len(data_loader)

def eval_function(data_loader, model):

  model.eval() 
  total_loss = 0.0

  with torch.no_grad():
    for images, masks in tqdm(data_loader):

      images = images.to(DEVICE)
      masks = masks.to(DEVICE, dtype=torch.long)


      c_next,h_next, logit = model(images)
    
    # make sure gradients are 0

      ce_weights = calculate_weights(masks)
      ce_weights = torch.tensor(ce_weights,dtype=torch.float).to(DEVICE)
      loss_per_frame = LovaszLoss(mode = 'multiclass', ignore_index=-1)(logit, masks)
      criterion = nn.CrossEntropyLoss(weight = ce_weights, ignore_index=-1)
      ce_logit = criterion(logit, masks)
      # ce_logit = ce_logit * 0.25
        
      loss = (loss_per_frame + ce_logit) 
      #loss = loss_per_frame

      total_loss += loss.item()

  return total_loss / len(data_loader)


counter = 0
epoch_start = 0
# epoch_start = checkpoint['epoch']

# best_valid_loss = checkpoint['best_loss']
# training_loss = checkpoint['train_loss']

print(epoch_start, best_valid_loss)
for epoch in range(epoch_start,EPOCHS):


  
  train_loss = train_function(trainloader, new_model, optimizer)
  valid_loss = eval_function(validloader, new_model)
  train_losses.append(train_loss)
  valid_losses.append(valid_loss)


  if valid_loss < best_valid_loss: #if best valid loss then upate new model
    torch.save(new_model.state_dict(), f'{DATA_URL}Models/U-net/unet_paper_structure_2.pt')
    print("Saved model")
    logger.info(f"Saved model: {valid_loss} - {epoch}") 
    best_valid_loss = valid_loss
    counter = 0 
  

  scheduler.step()

  if epoch % number_epoch_to_save == 0:

    torch.save({
              'epoch': epoch,
              'model_state_dict': new_model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(),
              'loss': valid_loss,
              'best_loss': best_valid_loss,
              'train_loss': train_loss
              }, f'{DATA_URL}Models/U-net/batchsize_continue.pt')
  
  #lrs.append(scheduler.get_last_lr())
  print(f"Epoch : {epoch+1} Train_loss : {train_loss} Valid_loss : {valid_loss} Learning rate:  ")
  logger.info(f"Epoch : {epoch+1} Train_loss : {train_loss} Valid_loss : {valid_loss} Learning rate: ") 

  if counter > 25:
    print(f"Early stopping: {epoch}, best valid loss : {best_valid_loss}")
    logger.info(f"Early stopping") 
    break
  counter+=1
    