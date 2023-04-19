
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

# class Initializer(nn.Module):
# 	def __init__(self):
# 		super(Initializer, self).__init__()
        
# 		self.new_layer = nn.Conv2d(in_channels=5, out_channels=64, kernel_size=3)
#     self.drop = nn.Dropout(p=0.5)
        
# 		self.pretrained_model = models.vgg16(weights='DEFAULT')
# 		self.model = nn.Sequential(*list(self.pretrained_model.features.children())[2:31])
# 		self.interp = nn.functional.interpolate

# 		self.c0 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1)

# 		self.h0 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1)
        
# 		self.relu = nn.ReLU()

# 	def forward(self, inputs):
#     x = self.relu(self.new_layer(inputs))
        
#     x = self.model(x)
#     c0 = self.relu(self.c0(x))
#     h0 = self.relu(self.h0(x))
#     c0 = self.interp(c0, size=(9,15), mode='bilinear', align_corners=False)
#     h0 = self.interp(h0, size=(9,15), mode='bilinear', align_corners=False)
#     return c0,h0

# class ConvLSTMCell(nn.Module):
#     """
#     Generate a convolutional LSTM cell
#     """

#     def __init__(self, input_size, hidden_size, height, width):
#         super().__init__()
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.conv = nn.Conv2d(input_size + hidden_size, 4 * hidden_size, 3, padding=3 // 2)
#         self.drop = nn.Dropout(p=0.3)
                
#         self.W_ci = nn.Parameter(torch.zeros(1, hidden_size, height, width))
#         self.W_co = nn.Parameter(torch.zeros(1, hidden_size, height, width))
#         self.W_cf = nn.Parameter(torch.zeros(1, hidden_size, height, width))


#     def forward(self, input_, hiddenState, cellState):
                

#         prev_hidden = hiddenState
#         prev_cell = cellState
                
#         # data size is [batch, channel, height, width]
#         stacked_inputs = torch.cat((input_, prev_hidden), 1)
                
#         combined_conv = self.conv(stacked_inputs)
#         combined_conv = self.drop(combined_conv)
                
#         # chunk across channel dimension -> get conv inputs
#         in_gate, forget_gate, out_gate, cell_gate = combined_conv.chunk(4, 1)


#         # apply sigmoid non linearity 
#         in_gate = torch.sigmoid(in_gate + self.W_ci * prev_cell)
#         # print(in_gate)

#         in_gate = self.drop(in_gate)

#         # print(in_gate)
#         forget_gate = torch.sigmoid(forget_gate + self.W_cf * prev_cell)
#         forget_gate = self.drop(forget_gate)
                
                

#         # apply tanh non linearity instead of sigmoid
#         cell_gate = torch.tanh(cell_gate)
#         cell_gate = self.drop(cell_gate)

#         # compute current cell and hidden state
#         cell = (forget_gate * prev_cell) + (in_gate * cell_gate)
#         out_gate = torch.sigmoid(out_gate + self.W_co * cell)
#         #out_gate = self.drop(out_gate)

                
#         hidden = out_gate * torch.relu(cell)
                

#         return hidden, cell

# class ConvLSTMCell(nn.Module):
#         """
#         Generate a convolutional LSTM cell
#         """

#         def __init__(self, input_size, hidden_size):
#                 super().__init__()
#                 self.kernel_size = 3
#                 self.input_size = input_size
#                 self.hidden_size = hidden_size
#                 self.padding = self.kernel_size // 2
#                 #self.conv = nn.Conv2d(input_size + hidden_size, 4 * hidden_size, kernel_size, padding=kernel_size // 2)
#                 self.Wxi = nn.Conv2d(self.input_size, self.hidden_size, self.kernel_size, 1, self.padding, bias=True)
#                 self.Whi = nn.Conv2d(self.hidden_size, self.hidden_size, self.kernel_size, 1, self.padding, bias=False)
#                 self.Wxf = nn.Conv2d(self.input_size, self.hidden_size, self.kernel_size, 1, self.padding, bias=True)
#                 self.Whf = nn.Conv2d(self.hidden_size, self.hidden_size, self.kernel_size, 1, self.padding, bias=False)
#                 self.Wxc = nn.Conv2d(self.input_size, self.hidden_size, self.kernel_size, 1, self.padding, bias=True)
#                 self.Whc = nn.Conv2d(self.hidden_size, self.hidden_size, self.kernel_size, 1, self.padding, bias=False)
#                 self.Wxo = nn.Conv2d(self.input_size, self.hidden_size, self.kernel_size, 1, self.padding, bias=True)
#                 self.Who = nn.Conv2d(self.hidden_size, self.hidden_size, self.kernel_size, 1, self.padding, bias=False)
#                 self.Wci = nn.Parameter(torch.zeros(1, hidden_size, 9, 15)).cuda()
#                 self.Wcf = nn.Parameter(torch.zeros(1, hidden_size, 9, 15)).cuda()
#                 self.Wco = nn.Parameter(torch.zeros(1, hidden_size, 9, 15)).cuda()


#                 self.drop = nn.Dropout(p=0.2)
#                 self.relu = nn.ReLU()
#                 # self.batchnorm = nn.BatchNorm2d(512)
#         def forward(self, x, h, c):
#                 it = torch.sigmoid(self.Wxi(x) + self.Whi(h) + self.Wci * c) 
                
#                 ft = torch.sigmoid(self.Wxf(x) + self.Whf(h) + self.Wcf * c)
                
#                 ct = (ft * c) + (it * self.relu(self.Wxc(x) + self.Whc(h)))
                
#                 ot = torch.sigmoid(self.Wxo(x) + self.Who(h) + self.Wco * ct)
#                 ht = ot * self.relu(ot)
#                 # ht = self.drop(ht)
#                 # ct = self.drop(ct)

#                 return ht, ct

# class ConvLSTMCell(nn.Module):
#         """
#         Generate a convolutional LSTM cell
#         """

#         def __init__(self, input_size, hidden_size):
#                 super().__init__()
#                 kernel_size = 3
#                 self.input_size = input_size
#                 self.hidden_size = hidden_size
#                 #self.conv = nn.Conv2d(input_size + hidden_size, 4 * hidden_size, kernel_size, padding=kernel_size // 2)
#                 self.conv3 = nn.Conv2d(input_size + hidden_size, 2 * hidden_size, kernel_size, padding=kernel_size // 2)
#                 self.conv5 = nn.Conv2d(input_size + hidden_size, 2 * hidden_size, 5, padding=5 // 2)
#                 self.conv1 = nn.Conv2d((input_size + hidden_size) * 2 , 4 * hidden_size, 1)
#                 #nn.Conv2d(in_channels=input_size + hidden_size, out_channels=n_classes, kernel_size=1)
#                 torch.nn.init.xavier_uniform_(self.conv3.weight)
#                 torch.nn.init.xavier_uniform_(self.conv5.weight)
#                 torch.nn.init.xavier_uniform_(self.conv1.weight)
#                 #torch.nn.init.xavier_uniform_(self.conv.weight)
#                 # bias_0 = torch.zeros(512)
#                 # bias_1 = torch.ones(512)
#                 # bias_weight = torch.cat((bias_0, bias_1,bias_0, bias_0 ), 0)
#                 # self.conv3.bias = nn.Parameter(bias_weight)

#                 # torch.nn.init.uniform_(self.conv.bias)
#                 #torch.nn.init.ones_(self.conv.bias)
#                 self.drop = nn.Dropout(p=0.3)
#                 self.relu = nn.ReLU()
#                 self.tanh = nn.Tanh()
#                 self.sigmoid = nn.Sigmoid()
#                 # self.batchnorm = nn.BatchNorm2d(512)
#         def forward(self, input_, hiddenState, cellState):
                

#                 prev_hidden = hiddenState
#                 prev_cell = cellState

#                 # prev_hidden = self.batchnorm(prev_hidden)

#                 # input_ = self.batchnorm(input_)

#                 # data size is [batch, channel, height, width]
#                 stacked_inputs = torch.cat([input_, prev_hidden], 1)
#                 combined_conv_3 = self.conv3(stacked_inputs)
#                 combined_conv_5 = self.conv5(stacked_inputs)

#                 combined_conv = torch.cat((combined_conv_3, combined_conv_5), dim=1)
#                 # # # print(combined_conv.shape)
#                 combined_conv = self.conv1(combined_conv)

#                 #combined_conv = self.conv(stacked_inputs)
                
#                 # combined_conv = self.drop(combined_conv)

#                 # # chunk across channel dimension -> get conv inputs
#                 # in_gate1, remember_gate1, out_gate1, cell_gate1 = combined_conv_3.chunk(4, 1)
#                 # in_gate2, remember_gate2, out_gate2, cell_gate2 = combined_conv_5.chunk(4, 1)

#                 # in_gate = torch.cat((in_gate1, in_gate2), dim=1)
#                 # remember_gate =torch.cat((remember_gate1, remember_gate2), dim=1)
#                 # out_gate =torch.cat((out_gate1, out_gate2), dim=1)
#                 # cell_gate =torch.cat((cell_gate1, cell_gate2), dim=1)

#                 in_gate, remember_gate, out_gate, cell_gate = torch.split(combined_conv, self.hidden_size, dim=1)


#                 in_gate = self.sigmoid(in_gate)

#                 remember_gate = self.sigmoid(remember_gate)

#                 out_gate = self.sigmoid(out_gate)


#                 # apply tanh non linearity instead of sigmoid
#                 cell_gate = self.relu(cell_gate)


#                 # compute current cell and hidden state
#                 cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
                
#                 hidden = out_gate * self.relu(cell)

#                 return hidden, cell

            
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
                # self.conv3 = nn.Conv2d(input_size + hidden_size, 2 * hidden_size, kernel_size, padding=kernel_size // 2)
                # self.conv5 = nn.Conv2d(input_size + hidden_size, 2 * hidden_size, 5, padding=5 // 2)
                # self.conv1 = nn.Conv2d((input_size + hidden_size) * 2 , 4 * hidden_size, 1)
                #nn.Conv2d(in_channels=input_size + hidden_size, out_channels=n_classes, kernel_size=1)
                # torch.nn.init.xavier_uniform_(self.conv3.weight)
                # torch.nn.init.xavier_uniform_(self.conv5.weight)
                # torch.nn.init.xavier_uniform_(self.conv1.weight)
                torch.nn.init.xavier_uniform_(self.conv.weight)
                # bias_0 = torch.zeros(512)
                # bias_1 = torch.ones(512)
                # bias_weight = torch.cat((bias_0, bias_1,bias_0, bias_0 ), 0)
                # self.conv3.bias = nn.Parameter(bias_weight)

                # torch.nn.init.uniform_(self.conv.bias)
                #torch.nn.init.ones_(self.conv.bias)
                self.drop = nn.Dropout(p=0.3)
                self.relu = nn.ReLU()
                self.tanh = nn.Tanh()
                self.sigmoid = nn.Sigmoid()
                # self.batchnorm = nn.BatchNorm2d(512)
        def forward(self, input_, hiddenState, cellState):
                

                prev_hidden = hiddenState
                prev_cell = cellState

                # prev_hidden = self.batchnorm(prev_hidden)

                # input_ = self.batchnorm(input_)

                # data size is [batch, channel, height, width]
                stacked_inputs = torch.cat([input_, prev_hidden], 1)
                # combined_conv_3 = self.conv3(stacked_inputs)
                # combined_conv_5 = self.conv5(stacked_inputs)

                # combined_conv = torch.cat((combined_conv_3, combined_conv_5), dim=1)
                # # # print(combined_conv.shape)
                # combined_conv = self.conv1(combined_conv)

                combined_conv = self.conv(stacked_inputs)
                
                # combined_conv = self.drop(combined_conv)

                # # chunk across channel dimension -> get conv inputs
                # in_gate1, remember_gate1, out_gate1, cell_gate1 = combined_conv_3.chunk(4, 1)
                # in_gate2, remember_gate2, out_gate2, cell_gate2 = combined_conv_5.chunk(4, 1)

                # in_gate = torch.cat((in_gate1, in_gate2), dim=1)
                # remember_gate =torch.cat((remember_gate1, remember_gate2), dim=1)
                # out_gate =torch.cat((out_gate1, out_gate2), dim=1)
                # cell_gate =torch.cat((cell_gate1, cell_gate2), dim=1)

                in_gate, remember_gate, out_gate, cell_gate = torch.split(combined_conv, self.hidden_size, dim=1)

                # apply sigmoid non linearity 
                in_gate = self.sigmoid(in_gate)
               # in_gate = self.drop(in_gate)
                remember_gate = self.sigmoid(remember_gate)
               # remember_gate = self.drop(remember_gate)
                out_gate = self.sigmoid(out_gate)
                #out_gate = self.drop(out_gate)

                # apply tanh non linearity instead of sigmoid
                cell_gate = self.relu(cell_gate)
              #  out_gate = self.drop(out_gate)

                # compute current cell and hidden state
                cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
                
                hidden = out_gate * self.relu(cell)

                return hidden, cell
# class ConvLSTM(nn.Module):
#     def __init__(self, convlstm, num_layers):
#         super(ConvLSTM).__init__()
#         self.convlstms = nn.ModuleList([convlstm for i in range(num_layers)])
    
# 	def forward(self, )


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
            #logits.append(logit_mask)
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


class Initializer2(nn.Module):
    def __init__(self, encoder,convlstm,decoder, head):
        super(Initializer2, self).__init__()
        self.encoder = encoder
        self.convlstm = convlstm
        self.decoder = decoder
        self.head = head


    def getLSTM(self):
        return self.convlstm

    def getEncoder(self):
        return self.encoder

    def getDecoder(self):
        return self.decoder

    def getHead(self):
        return self.head



    def forward(self, image):
        

        x_tilda = self.encoder(image)
        

        features = x_tilda
        feature_vector = x_tilda[5]

        c0 = torch.zeros(feature_vector.shape).to(DEVICE)
        h0 = torch.zeros(feature_vector.shape).to(DEVICE)

        c_next,h_next = self.convlstm(feature_vector, h0, c0)
        decoder = self.decoder

        x_tilda[5] = h_next
        decoder_output = decoder(*x_tilda)

        logits_mask = self.head(decoder_output)
    
        return c_next,h_next, logits_mask

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
        
# class UnetInitialiser(nn.Module):
#     def __init__(self, encoder,convlstm,decoder, head, sizes):
#         super(UnetInitialiser, self).__init__()
#         self.encoder = encoder
#         self.convlstm = convlstm
#         self.decoder = decoder
#         self.head = head
#         self.convlstms = nn.ModuleList([ConvLSTMCell(input_size = sizes[i], hidden_size = sizes[i]) for i in range(len(sizes))])


#     def getLSTM(self):
#         return self.convlstm

#     def getEncoder(self):
#         return self.encoder

#     def getDecoder(self):
#         return self.decoder

#     def getHead(self):
#         return self.head



#     def forward(self, images, masks = None):

        
        
#         image_0 = images[:,0,:,:,:]
#         image_1 = images[:,1,:,:,:]
#         image_2 = images[:,2,:,:,:]
#         if masks!=None:
#             mask = masks[:,2,:,:]
#             mask = mask.contiguous().long()

        
#         fv1 = self.encoder(image_0)
#         fv2 = self.encoder(image_1)
#         x_tilda = self.encoder(image_2)
#         # for feature in fv1:
#         #     print(feature.shape)
#         cell_states = []
#         hidden_states = []
#         for i in range(2, len(fv1)):
#             c0 = fv1[i]
#             h0 = fv2[i]
#             feature_vector = x_tilda[i]

#             c_next,h_next = self.convlstms[i-2](feature_vector, h0, c0)
#             cell_states.append(c_next)
#             hidden_states.append(h_next)
           

#             x_tilda[i] = h_next

#         # c0 = fv1[5]
#         # h0 = fv2[5]

#         # # print(c0.shape, h0.shape)

#         # x_tilda = self.encoder(image_2)
#         # feature_vector = x_tilda[4]
        


#         # c_next,h_next = self.convlstm(feature_vector, h0, c0)
        

#         # x_tilda[5] = h_next
#         decoder_output = self.decoder(*x_tilda)

#         logits_mask = self.head(decoder_output)

#         if masks==None:
#             return cell_states,hidden_states, logits_mask

#         # loss calculation 
#         lovasz_loss = LovaszLoss(mode = 'multiclass', ignore_index=-1)(logits_mask, mask)
        
#         ce_weights = calculate_weights(masks)
#         ce_weights = torch.tensor(ce_weights,dtype=torch.float).to(DEVICE)
#         criterion = nn.CrossEntropyLoss(weight = ce_weights, ignore_index=-1)
#         cross_entropy_loss = criterion(logits_mask, mask)

#         hidden_loss = nn.MSELoss()
#         cell_loss = nn.MSELoss()

#         hidden_state_loss = hidden_loss(h_next, feature_vector)
#         cell_state_loss = cell_loss(c_next, c0)
#         # print("hidden ", hidden_state_loss)
#         # print("cell ", cell_state_loss)
#         # print("lovasz ",  lovasz_loss)
#         # print("cross entropy ",  cross_entropy_loss)

#         loss = (lovasz_loss + cross_entropy_loss + (cell_state_loss + hidden_state_loss))
#         return cell_states, hidden_states, logits_mask, loss
    

class Network(nn.Module):
        def __init__(self, initializer, encoder,convlstm,decoder, head):
                super(Network, self).__init__()
                self.initializer = initializer
                self.encoder = encoder
                self.decoder = decoder
                self.head = head
                self.convlstm = convlstm

        def forward(self, images, masks = None):
            logits = []
            if images.dim()!=5:
                images = images.unsqueeze(0)

            c_original, feature_vector, c0,h0, logits_mask = self.initializer(images)
            cell_states = []
            #logits.append(logit_mask)
            length = images.shape[1]
            # logits.append(logits_mask)
            for i in range(3,length):
                
                image = images[:,i,:,:,:]
                x_tilda = self.encoder(image)

                feature_vector = x_tilda[5]

                c_next,h_next = self.convlstm(feature_vector, h0, c0)
                decoder = self.decoder

                x_tilda[5] = h_next
                decoder_output = decoder(*x_tilda)

                c0 = c_next
                h0 = h_next
                
                logits_mask = self.head(decoder_output)
                
                #logits_mask = logits_mask.squeeze(0)
                logits.append(logits_mask)
                
            logits = torch.stack(logits, dim=1)
            
            if logits.dim()!= 5:
                logits_mask.unsqueeze(0)
            
            logits = logits.transpose(2,1) # makes in the dimension of classes, no of images, width, height so 8,7,480,288
             
            return logits

class UnetInitialiser2(nn.Module):
    def __init__(self, encoder):
        super(UnetInitialiser2, self).__init__()
        self.encoder = encoder
        self.c0 = nn.Conv2d(in_channels=4, out_channels=3, kernel_size=1)
        self.conv1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1)
        self.relu = nn.ReLU()
        #self.convlstms = nn.ModuleList([ConvLSTMCell(input_size = sizes[i], hidden_size = sizes[i]) for i in range(len(sizes))])

    def forward(self, images, masks = None):
        
        image_0 = images[:,0,:,:,:]

        if masks!=None:
            mask = masks[:,0,:,:]
            mask = mask.contiguous().long()

        input = self.c0(torch.cat((image_0,mask.unsqueeze(1)),1))

        feature = self.encoder(input)
        c0 = self.conv1(feature[5])
        h0 = self.conv1(feature[5])
        c0 = self.relu(c0)
        h0 = self.relu(h0)
        
        return c0, h0

class LSTMModel2(nn.Module):
        def __init__(self, initializer, encoder,convlstm,decoder, head):
                super(LSTMModel2, self).__init__()
                self.initializer = initializer
                self.encoder = encoder
                self.convlstm = convlstm
                self.decoder = decoder
                self.head = head
                self.conv1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1)
                self.relu = nn.ReLU()


        def forward(self, images, masks):

            
            logits = []
            if images.dim()!=5:
                images = images.unsqueeze(0)
            
            if masks.dim()!=4:
                masks = masks.unsqueeze(0)

            c0,h0 = self.initializer(images, masks)

            cell_states = []

            length = images.shape[1]
            for i in range(1,length):
             
                image = images[:,i,:,:,:]
                x_tilda = self.encoder(image)
                features = x_tilda
                feature_vector = x_tilda[5]
                feature_vector = self.relu(self.conv1(feature_vector))
                c_next,h_next = self.convlstm(feature_vector, h0, c0)
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
         
            return logits, cell_states