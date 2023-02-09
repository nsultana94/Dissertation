
from tqdm import tqdm
DEVICE = 'cuda' #Cuda as using GPU
import torch 

def train_function(data_loader, model, optimizer):

  model.train()
  total_loss = 0.0

  for images, masks in tqdm(data_loader):

    images = images.to(DEVICE)
    masks = masks.to(DEVICE, dtype=torch.long)
    
    # make sure gradients are 0
    optimizer.zero_grad()
    logits, loss = model(images, masks)
    
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


      logits, loss = model(images, masks)


      total_loss += loss.item()

  return total_loss / len(data_loader)