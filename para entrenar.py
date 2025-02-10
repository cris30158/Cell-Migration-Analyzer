import sys

import torch 
import cv2

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 

from sklearn.model_selection import train_test_split
from tqdm import tqdm

import helper

"""# Task : 2 Setup Configurations"""

CSV_FILE = 'C:/Users/cristina/Desktop/proyectoo/Machinelearning/programasml/array.csv'
DATA_DIR = 'C:/Users/cristina/Desktop/proyectoo/Machinelearning/programasml'

DEVICE ='cpu'
EPOCHS = 25
LR= 0.003
BATCH_SIZE= 3
IMG_SIZE = 512

ENCODER= 'timm-efficientnet-b0'
WEIGHTS = 'imagenet'

df = pd.read_csv(CSV_FILE)
df.head()

idx =2
row = df.iloc[idx]
image_path= row.images
mask_path = row.masks

image = cv2.imread(image_path)
image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

mask =cv2.imread(mask_path,cv2.IMREAD_GRAYSCALE)/255.0

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
        
ax1.set_title('IMAGE')
ax1.imshow(image)

ax2.set_title('GROUND TRUTH')
ax2.imshow(mask,cmap = 'gray')

train_df, valid_df = train_test_split(df,test_size=0.2,random_state=42)
len(train_df)
#%%
"""# Task 3 : Augmentation Functions

albumentation documentation : https://albumentations.ai/docs/
"""

import albumentations as A

def get_train_augs():
  return A.Compose([
      A.Resize(IMG_SIZE,IMG_SIZE),
      A.HorizontalFlip(p=0.5),
      A.VerticalFlip(p=0.5)
  ])

def get_valid_augs():
  return A.Compose([A.Resize(IMG_SIZE,IMG_SIZE),])

# def Resize():
#   return A.Compose([A.Resize(2048,1536),])
#%%
"""# Task 4 : Create Custom Dataset """

from torch.utils.data import Dataset

class SegmentationDataset(Dataset):
  def __init__(self,df,augmentations):
    self.df = df
    self.augmentations =augmentations
  def __len__(self):
    return len(self.df)
  def __getitem__(self,idx):
    row = self.df.iloc[idx]
    image_path= row.images
    mask_path =row.masks

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

    mask =cv2.imread(mask_path,cv2.IMREAD_GRAYSCALE) #(h,w)
    mask = np.expand_dims(mask,axis=-1)
    if self.augmentations:
      data=self.augmentations(image=image,mask=mask)
      image=data['image']
      mask= data['mask']

    image = np.transpose(image,(2,0,1)).astype(np.float32)
    mask = np.transpose(mask,(2,0,1)).astype(np.float32)
    
    image=torch.Tensor(image)/255.0
    mask = torch.round(torch.Tensor(mask)/255.0)
    return image,mask

trainset= SegmentationDataset(train_df,get_train_augs())
validset= SegmentationDataset(valid_df,get_valid_augs())

print(f"size of trainset: {len(trainset)}")
print(f"size of validset: {len(validset)}")

idx =7
image,mask = trainset[idx]
helper.show_image(image,mask)
#%%
"""# Task 5 : Load dataset into batches"""

from torch.utils.data import DataLoader

trainloader = DataLoader(trainset,batch_size=BATCH_SIZE,shuffle=True)
validloader = DataLoader(validset,batch_size=BATCH_SIZE)

print(f"total no of batches trainloader: {len(trainloader)}")
print(f"total no of batches validloader: {len(validloader)}")

for images,masks in trainloader:
  print(f"one batche image shape:{images.shape}")
  print(f"one batche mask shape:{masks.shape}")
  break;
#%%
"""# Task 6 : Create Segmentation Model

segmentation_models_pytorch documentation : https://smp.readthedocs.io/en/latest/
"""

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss
from torch import nn

class SegmentationModel(nn.Module):
  def __init__(self):
      super(SegmentationModel,self).__init__()

      self.backbone = smp.Unet(
          encoder_name=ENCODER,
          encoder_weights= WEIGHTS,
          in_channels= 3,
          classes= 1,
          activation = None
      )

  def forward(self, images, masks = None):
    logits = self.backbone(images)

    if masks !=None:
      return logits, DiceLoss(mode='binary')(logits,masks) + nn.BCEWithLogitsLoss()(logits,masks)
    return logits

model = SegmentationModel()
model.to(DEVICE);
#%%
"""# Task 7 : Create Train and Validation Function """

def train_fn(dataloader, model, optimizer):
  
  model.train()

  total_loss=0.0

  for images, masks in tqdm(dataloader):
    images = images.to(DEVICE)
    masks = masks.to(DEVICE)

    optimizer.zero_grad()
    logits, loss=model(images, masks)
    loss.backward()
    optimizer.step()
    total_loss+=loss.item()
  return total_loss/ len(dataloader)

def eval_fn(dataloader, model):
  
  model.eval()

  total_loss=0.0
  with torch.no_grad():
    for images, masks in tqdm(dataloader):
      images = images.to(DEVICE)
      masks = masks.to(DEVICE)

      
      logits, loss=model(images, masks)
      
      total_loss+=loss.item()
  return total_loss/ len(dataloader)
#%%
"""# Task 8 : Train Model"""

optimizer = torch.optim.Adam(model.parameters(),lr =LR)

best_loss= np.Inf
for i in range(EPOCHS):
  train_loss = train_fn(trainloader, model, optimizer)
  valid_loss= eval_fn(validloader, model)

  if valid_loss< best_loss:
    torch.save(model.state_dict(), "best-model.pt")
    print("saved-model")
    best_loss=valid_loss
  
  print(f"Epoh:{i+1} Trainloss: {train_loss} validloss: {valid_loss}")
#%%

dr = pd.read_csv('C:/Users/cristina/Desktop/proyectoo/Machinelearning/programasml/array.csv')
dr.head()

posese= SegmentationDataset(dr,get_valid_augs())
#%%
"""# Task 9 : Inference"""

idx =1
model.load_state_dict(torch.load('C:/Users/cristina/Desktop/proyectoo/Machinelearning/programasml/best-model.pt'))
image, mask =posese[idx]

logits_mask = model(image.to(DEVICE).unsqueeze(0))
pred_mask= torch.sigmoid(logits_mask)
pred_mask=(pred_mask>0.5)*1.0

helper.show_image(image, pred_mask.detach().cpu().squeeze(0))

"""# For updates about upcoming and current guided projects follow me on...

Twitter : @parth_AI

Linkedin : www.linkedin.com/in/pdhameliya
pred_mask1= pred_mask*255.0
jamon=pred_mask1.detach().cpu().squeeze(0)
jamon1=jamon.permute(1,2,0)
jamon1  = jamon1.cpu().numpy()
cv2.imwrite('prueba.jpg', jamon1)

"""
#%%
class SegmentationDataset1(Dataset):
  def __init__(self,augmentations):
    self.augmentations =augmentations
  def __len__(self):
    return 1
  def __getitem__(self,idx):
    image_path= 'C:/TFG_Irene/ImagenesDL/prueba2.png'

    image = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)

    if self.augmentations:
      data=self.augmentations(image=image,mask=mask)
      image=data['image']
    return image

# trainset223= SegmentationDataset1(Resize())