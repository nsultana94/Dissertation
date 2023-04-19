
import torch 

from torch import nn # neural netowrk 


import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss, LovaszLoss, FocalLoss, JaccardLoss


import torchvision.models as models
import torch.nn.functional as f
import numpy as np


DEVICE = torch.device('cuda')
EPOCHS = 50 #25 training iterations
LR = 0.00001 #decay learning rate
BATCH_SIZE = 4
HEIGHT = 288
WIDTH = 480
ENCODER = 'resnet34'
WEIGHTS = 'imagenet'
DATA_URL = "/cs/student/projects1/2019/nsultana/"

class SegmentationModel(nn.Module):

    def __init__(self):
        super(SegmentationModel, self).__init__() 

        self.architecture = smp.Unet(
                encoder_name = ENCODER,
                encoder_weights = WEIGHTS, 
                in_channels = 3, #Input is RGB
                classes = 8, # binary segmentation problem 
                decoder_use_batchnorm = True,
                activation = None #no sigmoid or softmax
                #aux_params = {'dropout': 0.2, 'classes' :8}
                

        )
    
    def show(self):
        return self.architecture

    def forward(self, images, masks = None):
        logits = self.architecture(images) #probabilities / predictions
     
        
        if masks!= None:
            weights = [4.116647326424263, 24.600245093614593, 191.78790779880697, 240.94195047235274, 7.334747505863925, 10.620043927212807, 2.219872768361696, 38.32265526553685]

            class_weights=torch.tensor(weights,dtype=torch.float).to(DEVICE)
            lovasz = LovaszLoss(mode = 'multiclass')(logits, masks)
            #loss_fn = nn.CrossEntropyLoss()#binary cross entropy loss

            loss = lovasz
            return logits, loss

        return logits

""" initialiser network code adapted from  https://github.com/shashankvkt/video_object_segmentation"""

class Initializer(nn.Module):
    def __init__(self):
        super(Initializer, self).__init__()
        self.new_layer = nn.Conv2d(in_channels=4, out_channels=64, kernel_size=3)

        
        self.pretrained_model = models.vgg16(weights='DEFAULT')
        self.model = nn.Sequential(*list(self.pretrained_model.features.children())[2:31])
        self.interp = nn.functional.interpolate

        self.c0 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1)

        self.h0 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1)
        
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.relu(self.new_layer(inputs))
        
        x = self.model(x)
        c0 = self.relu(self.c0(x))

        h0 = self.relu(self.h0(x))

        c0 = self.interp(c0, size=(9,15), mode='bilinear', align_corners=False)
        h0 = self.interp(h0, size=(9,15), mode='bilinear', align_corners=False)
        return c0,h0

""" conv lstm  cell code adapted from  https://github.com/ndrplz/ConvLSTM_pytorch/blob/master/convlstm.py """   
class ConvLSTMCell(nn.Module):
        """
        Generate a convolutional LSTM cell
        """

        def __init__(self, input_size, hidden_size):
                super().__init__()
                kernel_size = 3
                self.input_size = input_size
                self.hidden_size = hidden_size
                self.conv = nn.Conv2d(input_size + hidden_size, 4 * hidden_size, kernel_size, padding=kernel_size // 2)

                torch.nn.init.xavier_uniform_(self.conv.weight)

                self.drop = nn.Dropout(p=0.3)
                self.relu = nn.ReLU()

                self.sigmoid = nn.Sigmoid()

        def forward(self, input_, hiddenState, cellState):
                prev_hidden = hiddenState
                prev_cell = cellState
                stacked_inputs = torch.cat([input_, prev_hidden], 1)
                combined_conv = self.conv(stacked_inputs)
                in_gate, remember_gate, out_gate, cell_gate = torch.split(combined_conv, self.hidden_size, dim=1)

                in_gate = self.sigmoid(in_gate)
                remember_gate = self.sigmoid(remember_gate)
                out_gate = self.sigmoid(out_gate)

                cell_gate = self.relu(cell_gate)

                cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
                
                hidden = out_gate * self.relu(cell)

                return hidden, cell


class LSTMModel(nn.Module):
        def __init__(self, initializer, encoder,convlstm,decoder, head):
                super(LSTMModel, self).__init__()
                self.initializer = initializer
                self.encoder = encoder
                self.convlstm = convlstm
                self.decoder = decoder
                self.head = head
        
        def getInitializer(self):
            return self.initializer

        def getLSTM(self):
            return self.convlstm
        
        def getEncoder(self):
            return self.encoder

        def getDecoder(self):
            return self.decoder

        def getHead(self):
            return self.head

        def forward(self, images, masks):

            
            logits = []
            if images.dim()!=5:
                images = images.unsqueeze(0)
            
            if masks.dim()!=4:
                masks = masks.unsqueeze(0)
     
            first_image = images[:,0,:,:,:]
            
            
            
            if len(masks) > 1:
              mask = masks[:,0,:,:]
              mask = mask.to(device = DEVICE).unsqueeze(0)
              # mask = mask.unsqueeze(0)
            else:
              mask = masks[:,0,:,:]
              mask = mask.to(device = DEVICE).unsqueeze(0)

            mask = np.swapaxes(mask,0,1)
            
            
            try:
                c0,h0 = self.initializer(torch.cat((first_image,masks),1))

            except:
                print(first_image.shape, mask.shape, masks.shape, masks[:,0,:,:].unsqueeze(0).shape)
            cell_states = []

            length = images.shape[1]
            for i in range(1,length):
             
                image = images[:,i,:,:,:]
                
         

                x_tilda = self.encoder(image)
                features = x_tilda
                feature_vector = x_tilda[5]
                h_next,c_next = self.convlstm(feature_vector, h0, c0)
                decoder = self.decoder

                x_tilda[5] = h_next
                decoder_output = decoder(*x_tilda)
                c0 = c_next
                h0 = h_next
                
                
                logits_mask = self.head(decoder_output)
                
                logits.append(logits_mask)
                cell_states.append(c_next)
            
            logits = torch.stack(logits, dim=1)
            cell_states = torch.stack(cell_states)
            if logits.dim()!= 5:
                logits_mask.unsqueeze(0)
            
            logits = logits.transpose(2,1) # makes in the dimension of classes, no of images, width, height so 8,7,480,288
         
            return logits



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
        

        cell_states = []
        hidden_states = []


        c0 = fv1[5]
        h0 = fv2[5]

        x_tilda = self.encoder(image_2)
        feature_vector = x_tilda[5]

        c_next,h_next = self.convlstm(feature_vector, h0, c0)
        x_tilda[5] = h_next
        decoder_output = self.decoder(*x_tilda)

        logits_mask = self.head(decoder_output)

        if masks==None:
            return c0, feature_vector, c_next,h_next, logits_mask

        # loss calculation 
        lovasz_loss = LovaszLoss(mode = 'multiclass', ignore_index=-1)(logits_mask, mask)
        
        ce_weights = calculate_weights(masks)
        ce_weights = torch.tensor(ce_weights,dtype=torch.float).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight = ce_weights, ignore_index=-1)
        cross_entropy_loss = criterion(logits_mask, mask)

        hidden_loss = nn.MSELoss()
        cell_loss = nn.MSELoss()


        hidden_state_loss = hidden_loss(h_next, feature_vector)
        cell_state_loss = cell_loss(c_next, c0)


        loss = (lovasz_loss + cross_entropy_loss + (cell_state_loss + hidden_state_loss))

    
        return c_next,h_next, logits_mask

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