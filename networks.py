
import torch 

from torch import nn # neural netowrk 


import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss, LovaszLoss, FocalLoss, JaccardLoss


import torchvision.models as models
import torch.nn.functional as f


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
        activation = None #no sigmoid or softmax

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

class ConvLSTMCell(nn.Module):
    """
    Generate a convolutional LSTM cell
    """

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.conv = nn.Conv2d(input_size + hidden_size, 4 * hidden_size, 3, padding=3 // 2)
        self.drop = nn.Dropout(p=0.2)

    def forward(self, input_, hiddenState, cellState):
        

        prev_hidden = hiddenState
        prev_cell = cellState
        
        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat((input_, prev_hidden), 1)
        
        combined_conv = self.conv(stacked_inputs)
        
        # chunk across channel dimension -> get conv inputs
        in_gate, remember_gate, out_gate, cell_gate = combined_conv.chunk(4, 1)


        # apply sigmoid non linearity 
        in_gate = torch.sigmoid(in_gate)
        in_gate = self.drop(in_gate)
        remember_gate = torch.sigmoid(remember_gate)
        remember_gate = self.drop(remember_gate)
        out_gate = torch.sigmoid(out_gate)
        out_gate = self.drop(out_gate)

        # apply tanh non linearity instead of sigmoid
        cell_gate = torch.relu(cell_gate)
        cell_gate = self.drop(cell_gate)

        # compute current cell and hidden state
        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)

        cell = self.drop(cell)
        hidden = out_gate * torch.relu(cell)

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
      
      first_image = images[0]
      first_image = first_image.to(DEVICE).unsqueeze(0)
      if len(masks) > 1:
        mask = masks[0]
        mask = mask.to(device = DEVICE).unsqueeze(0)
        mask = mask.unsqueeze(0)
      else:
        mask = masks
      #mask = np.swapaxes(mask,1,2)

   

      c0,h0 = self.initializer(torch.cat((first_image,mask),1))
      print(c0.shape,h0.shape)
      for i in range(1,len(images)):
        image = images[i,:,:,:]
        image = image.to(device = DEVICE).unsqueeze(0)

        x_tilda = self.encoder(image)
        features = x_tilda
        feature_vector = x_tilda[5]
        c_next,h_next = self.convlstm(feature_vector, h0, c0)
        decoder = self.decoder

        features[5] = h_next
        decoder_output = decoder(*features)
        c0 = c_next
        h0 = h_next
        
        logits_mask = self.head(decoder_output)
        logits_mask = logits_mask.squeeze(0)
        logits.append(logits_mask)
        
      logits = torch.stack(logits)

      logits = logits.transpose(1,0) # makes in the dimension of classes, no of images, width, height so 8,7,480,288
      
      return logits