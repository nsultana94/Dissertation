# -*- coding: utf-8 -*-
"""conv lstm.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1pk_9y1txgv_KprxWLG0C-PkaUjAF-E3J

# Installation
"""

import logging 
logging.basicConfig(filename="std.log", 
					format='%(asctime)s %(message)s', 
					filemode='w')

logger=logging.getLogger() 
logger.setLevel(logging.DEBUG) 

import torch 
import cv2

import numpy as np 

import glob
import matplotlib.pyplot as plt
import pandas as pd
from networks import SegmentationModel, ConvLSTMCell, LSTMModel, Initializer
from dataloader import get_train_augs, get_test_augs, get_valid_augs, SegmentationDataset
from train_functions import train_function, eval_function
DEVICE = torch.device('cuda') 
# DEVICE = 'cuda' #Cuda as using GPU



EPOCHS = 50 #25 training iterations
LR = 0.00001 #decay learning rate
BATCH_SIZE = 4
HEIGHT = 288
WIDTH = 480
ENCODER = 'resnet34'
WEIGHTS = 'imagenet'
DATA_URL = "/cs/student/projects1/2019/nsultana/"


training_images = (glob.glob(f"{DATA_URL}updated_data/Data/Training/*.npz"))
print(len(training_images))

testing_images = (glob.glob(f"{DATA_URL}updated_data/Data/Testing/*.npz"))
print(len(testing_images))
validation_images = (glob.glob(f"{DATA_URL}updated_data/Data/Validation/*.npz"))
print(len(validation_images))

"""# Set up model"""


model = SegmentationModel()
model = model.to(device =DEVICE); #i.e CUDA


"""# Set up dataset and data loader"""



trainset = SegmentationDataset("Training", get_train_augs(), training_images)
validset = SegmentationDataset("Validation", get_valid_augs(), validation_images)
testset = SegmentationDataset("Testing", get_test_augs(), testing_images)

from torch.utils.data import DataLoader
trainloader = DataLoader(trainset, batch_size = 4, shuffle = True,num_workers=2) #every epoch batches shuffles
validloader = DataLoader(validset, batch_size = 4, shuffle = True,num_workers=2)




"""# Training model"""

model.load_state_dict(torch.load(f'{DATA_URL}Models/best_model_aug.pt'))
model_summary = model.show()
encoder = model_summary.encoder
initializer = Initializer()
decoder = model_summary.decoder
head = model_summary.segmentation_head


for name,param in encoder.named_parameters():
  param.requires_grad = False


for name,param in head.named_parameters():
  param.requires_grad = False


for name,param in decoder.named_parameters():
  param.requires_grad = False

convlstm  = ConvLSTMCell(input_size = 512, hidden_size = 512)
new_model = LSTMModel(initializer,encoder,convlstm,decoder, head)
new_model = new_model.to(device = DEVICE)




# new_model.load_state_dict(torch.load(f'{DATA_URL}Models/unet_initialization.pt'))
# encoder = new_model.getEncoder()
# head = new_model.getHead()



# checkpoint = torch.load(f'{DATA_URL}Models/continue_model.pt')
# new_model.load_state_dict(checkpoint['model_state_dict'])
# initializer = new_model.getInitializer()
# encoder = new_model.getEncoder()
# decoder = new_model.getDecoder()
# convlstm = new_model.getLSTM()
# for name,param in initializer.named_parameters():
#   param.requires_grad = False

# for name,param in convlstm.named_parameters():
#   param.requires_grad = False

# for name,param in encoder.named_parameters():
#   param.requires_grad = True

# for name,param in decoder.named_parameters():
  # param.requires_grad = True
epoch_start = 0
# epoch_start = checkpoint['epoch']

# loss = checkpoint['loss']
best_valid_loss = np.inf
# best_valid_loss = checkpoint['best_loss']
# training_loss = checkpoint['train_loss']

EPOCHS = 100


valid_losses = []
train_losses = []

# train_losses = checkpoint['train_losses']
# valid_losses = checkpoint['valid_losses']

lrs = []

#0.12911548332047107 epoch 28 
LR = 0.0001
optimizer = torch.optim.Adam(new_model.parameters(), lr = LR)
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.95)
#scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, 0.000001, 0.00001,5, cycle_momentum = False)
# df  = pd.read_csv("Book1.csv")
# # valid_losses.append(0.6590140003766587)
# # valid_losses.append(0.65166707270181)

# # train_losses.append(0.5935223279872311)
# # train_losses.append(0.5846759660415491)
# df['training1'] = train_losses
# df['validation1'] = valid_losses
# df.to_csv("new.csv")

number_epoch_to_save = 5
 
counter = 0

print(epoch_start, best_valid_loss)

for epoch in range(epoch_start,EPOCHS):


  train_loss = train_function(trainloader, new_model, optimizer, model)
  valid_loss = eval_function(validloader, new_model, model)
  train_losses.append(train_loss)
  valid_losses.append(valid_loss)
  
  
  if valid_loss < best_valid_loss: #if best valid loss then upate new model
    torch.save(new_model.state_dict(), f'{DATA_URL}Models/of_init_longer_corrected.pt')
    print("Saved model")
    logger.info(f"Saved model: {valid_loss} - {epoch}") 
    best_valid_loss = valid_loss
    counter = 0

  scheduler.step()

  if counter > 25:
    print(f"Early stopping: {epoch}, best valid loss : {best_valid_loss}")
    logger.info(f"Early stopping") 
    break
    
    
        

  if epoch % number_epoch_to_save == 0:

    torch.save({
              'epoch': epoch,
              'model_state_dict': new_model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(),
              'loss': valid_loss,
              'best_loss': best_valid_loss,
              'train_loss': train_loss,
              'train_losses': train_losses,
              'valid_losses': valid_losses
              }, f'{DATA_URL}Models/continue_model.pt')
  
#   #lrs.append(scheduler.get_last_lr())
  
  print(f"Epoch : {epoch+1} Train_loss : {train_loss} Valid_loss : {valid_loss} Learning rate: {scheduler.get_last_lr()} ")
  logger.info(f"Epoch : {epoch+1} Train_loss : {train_loss} Valid_loss : {valid_loss} Learning rate: {scheduler.get_last_lr()}") 
  counter +=1
  np.savez('losses.npz', train=train_losses, valid=valid_losses)




# """# Only training conv lstm and decoder + encoder"""



# # encoder = new_model.getEncoder()
# # decoder = new_model.getDecoder()

# # for name,param in encoder.named_parameters():
# #   print(param.requires_grad)
# #   break

# # new_model.load_state_dict(torch.load(f'{DATA_URL}Models/conv_lstm_triplet_loss_continued.pt'))

# # initializer = new_model.getInitializer()
# # encoder = new_model.getEncoder()
# # decoder = new_model.getDecoder()
# # convlstm = new_model.getLSTM()
# # for name,param in initializer.named_parameters():
# #   param.requires_grad = False

# # for name,param in convlstm.named_parameters():
# #   param.requires_grad = True

# # for name,param in encoder.named_parameters():
# #   param.requires_grad = True

# # for name,param in decoder.named_parameters():
# #   param.requires_grad = True

# # train_losses = []
# # valid_losses = []


# # LR = 0.00001
# # optimizer = torch.optim.Adam(new_model.parameters(), lr = LR)





