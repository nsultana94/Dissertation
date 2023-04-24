import logging 
logging.basicConfig(filename="init4.log", 
                    format='%(asctime)s %(message)s', 
                    filemode='w')

logger=logging.getLogger() 
logger.setLevel(logging.DEBUG) 
import torch 
import cv2
import random
import numpy as np 
from tqdm import tqdm
from torch import nn # neural netowrk 

import glob
import matplotlib.pyplot as plt
import pandas as pd
from networks import SegmentationModel, ConvLSTMCell, LSTMModel, Initializer2
from dataloader import get_train_augs, get_test_augs, get_valid_augs, SegmentationDataset
from train_functions import train_function, eval_function
from segmentation_models_pytorch.losses import DiceLoss, LovaszLoss, FocalLoss, JaccardLoss
DEVICE = torch.device('cuda') 

EPOCHS = 30 #25 training iterations
LR = 0.00001 #decay learning rate
BATCH_SIZE = 8
HEIGHT = 288
WIDTH = 480
ENCODER = 'resnet34'
WEIGHTS = 'imagenet'
DATA_URL = "/cs/student/projects1/2019/nsultana/"

""" weighting function """
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


"""initialiser network"""

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

    def forward(self, images, masks):
        length = images.shape[1] - 2
        image_0 = images[:,0,:,:,:]
        image_1 = images[:,1,:,:,:]
        image_2 = images[:,2,:,:,:]
        if masks!=None:
            mask = masks[:,2,:,:]
            mask = mask.contiguous().long()


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
            
            x_tilda = self.encoder(image_2)
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
        ce_weights = calculate_weights(masks)
        ce_weights = torch.tensor(ce_weights,dtype=torch.float).to(DEVICE)
        for i in range(0, len(x_tildas)):
            
            h = cur_layer_input[i,:,:,:,:]
            x_tilda = x_tildas[i]
            x_tilda[5] = h

            decoder_output = self.decoder(*x_tilda)
            logits_mask = self.head(decoder_output)

            lovasz_loss = LovaszLoss(mode = 'multiclass', ignore_index=-1)(logits_mask, mask)

        
            criterion = nn.CrossEntropyLoss(weight = ce_weights, ignore_index=-1)
            cross_entropy_loss = criterion(logits_mask, mask)

            hidden_loss = nn.MSELoss()
            cell_loss = nn.MSELoss()

            # hidden_state_loss = hidden_loss(h, h0) 
            cell_state_loss = cell_loss(c, c0)

        loss = (lovasz_loss + cross_entropy_loss + (cell_state_loss))
        return c, h, logits_mask, loss
    


class UnetInitialiser(nn.Module):
    def __init__(self, encoder,convlstm,decoder, head, sizes):
        super(UnetInitialiser, self).__init__()
        self.encoder = encoder
        self.convlstm = convlstm
        self.decoder = decoder
        self.head = head




    def forward(self, images, masks = None):

        
        
        image_0 = images[:,0,:,:,:]
        image_1 = images[:,1,:,:,:]
        image_2 = images[:,2,:,:,:]
        if masks!=None:
            mask = masks[:,2,:,:]
            mask = mask.contiguous().long()

        
        fv1 = self.encoder(image_0)
        fv2 = self.encoder(image_1)
        x_tilda = self.encoder(image_2)
        features = x_tilda.copy()
        
        # for feature in fv1:
        #     print(feature.shape)
        cell_states = []
        hidden_states = []


        c0 = fv1[5]
        h0 = fv2[5]

        x_tilda = self.encoder(image_2)
        feature_vector = x_tilda[5]

        h_next,c_next = self.convlstm(feature_vector, h0, c0)
        x_tilda[5] = h_next
        decoder_output = self.decoder(*x_tilda)

        logits_mask = self.head(decoder_output)

        if masks==None:
            return c_next,h_next, logits_mask

        # loss calculation 
        lovasz_loss = LovaszLoss(mode = 'multiclass', ignore_index=-1)(logits_mask, mask)
        
        ce_weights = calculate_weights(masks)
        ce_weights = torch.tensor(ce_weights,dtype=torch.float).to(DEVICE)
        
        criterion = nn.CrossEntropyLoss(weight = ce_weights, ignore_index=-1)
        cross_entropy_loss = criterion(logits_mask, mask)

        hidden_loss = nn.MSELoss()
        cell_loss = nn.MSELoss()

        cell_state_loss = cell_loss(c_next, c0)



        loss = (lovasz_loss + cross_entropy_loss + (cell_state_loss))
        return c_next, h_next, logits_mask, loss
    
 
# """ train and eval functions """



def train_function(data_loader, model, optimizer):
    
    model.train()
    total_loss = 0.0
    
    for images, masks in tqdm(data_loader):
        optimizer.zero_grad()
        images = images.to(device = DEVICE)
        masks = masks.to(device = DEVICE, dtype=torch.long)
        c_next, h_next, logits_mask, loss = model(images, masks)
        # print(loss)
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(data_loader)

def eval_function(data_loader, model):
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for images, masks in tqdm(data_loader):
            images = images.to(device = DEVICE)
            masks = masks.to(device = DEVICE, dtype=torch.long)
            c_next, h_next, logits_mask, loss = model(images, masks)
            total_loss += loss.item()

    return total_loss / len(data_loader)



""" Images """
training_images = (glob.glob(f"{DATA_URL}new_data/Training1/*.npz"))



testing_images = (glob.glob(f"{DATA_URL}new_data/Testing/*.npz"))

validation_images = (glob.glob(f"{DATA_URL}new_data/Validation/*.npz"))


"""# Set up model"""

model = SegmentationModel()
model = model.to(device =DEVICE); #i.e CUDA
# model.load_state_dict(torch.load(f'{DATA_URL}Models/best_model_aug.pt'))
"""# Set up dataset and data loader"""


trainset = SegmentationDataset("Training", get_train_augs(), training_images)
validset = SegmentationDataset("Validation", get_valid_augs(), validation_images)
testset = SegmentationDataset("Testing", get_test_augs(), testing_images)

from torch.utils.data import DataLoader
trainloader = DataLoader(trainset, batch_size = BATCH_SIZE, shuffle = True,num_workers=2) #every epoch batches shuffles
validloader = DataLoader(validset, batch_size = BATCH_SIZE, shuffle = True,num_workers=2)



model_summary = model.show()
encoder = model_summary.encoder
decoder = model_summary.decoder
head = model_summary.segmentation_head

sizes = [64,128,256,512]


    
convlstm  = ConvLSTMCell(input_size = 512, hidden_size = 512)

initialiser = ConvLSTM(512, 512, 2, encoder, decoder, head)
initialiser = initialiser.to(device = DEVICE)



encoder = initialiser.encoder
decoder = initialiser.decoder
convlstms = initialiser.layers
head = initialiser.head


for name,param in initialiser.named_parameters():
    logger.info(f"{name}, {param.requires_grad}")


LR = 0.0001
optimizer = torch.optim.Adam(initialiser.parameters(), lr = LR)

lambda1 = lambda1 = lambda epoch : pow((1 - epoch / EPOCHS), 0.9)

scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.95)



best_valid_loss = np.Inf



epoch_start = 0

print(epoch_start, best_valid_loss)
counter = 0
for epoch in range(epoch_start,EPOCHS):



    train_loss = train_function(trainloader, initialiser, optimizer)
    valid_loss = eval_function(validloader, initialiser)
    if valid_loss < best_valid_loss: #if best valid loss then upate new model
        torch.save(initialiser.state_dict(), f'{DATA_URL}Models/U-net/2layers_3kernels.pt')
        print("Saved model")
        logger.info(f"Saved model: {valid_loss} - {epoch}") 
        best_valid_loss = valid_loss
        counter = 0
    torch.save({
                    'epoch': epoch,
                    'model_state_dict': initialiser.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_loss': best_valid_loss,
                    }, f'{DATA_URL}Models/kernel.pt')
    scheduler.step() 
    print(f"Epoch : {epoch+1} Train_loss : {train_loss} Valid_loss : {valid_loss} Learning rate:  ")
    logger.info(f"Epoch : {epoch+1} Train_loss : {train_loss} Valid_loss : {valid_loss} Learning rate:{scheduler.get_last_lr()} ") 
    if counter > 25:
        print(f"Early stopping: {epoch}, best valid loss : {best_valid_loss}")
        logger.info(f"Early stopping") 
        break
  