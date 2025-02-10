def elmodulo(CSVpath,Modelpath,index):
    
    import torch 
    import cv2
    import os
    import numpy as np 
    import pandas as pd
    
    """# Task : 2 Setup Configurations"""
    
    DEVICE ='cpu'
    IMG_SIZE = 512
    
    ENCODER= 'timm-efficientnet-b0'
    WEIGHTS = 'imagenet'
    import albumentations as A

    def get_train_augs():
        return A.Compose([
          A.Resize(IMG_SIZE,IMG_SIZE),
          A.HorizontalFlip(p=0.5),
          A.VerticalFlip(p=0.5)
      ])

    def get_valid_augs():
        return A.Compose([
          A.Resize(IMG_SIZE,IMG_SIZE),
      ])
    def Resize():
        return A.Compose([
          A.Resize(2048,1536),
      ])
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
    model.to(DEVICE)
    dr = pd.read_csv(CSVpath)
    dr.head()

    posese= SegmentationDataset(dr,get_valid_augs())
    #%%
    """# Task 9 : Inference"""

    idx =index
    model.load_state_dict(torch.load(Modelpath))
    image, mask =posese[idx]

    logits_mask = model(image.to(DEVICE).unsqueeze(0))
    pred_mask= torch.sigmoid(logits_mask)
    pred_mask=(pred_mask>0.5)*1.0

    impath=os.path.dirname(CSVpath)+'/imagen.jpg'
    pred_mask1= pred_mask*255.0
    jamon=pred_mask1.detach().cpu().squeeze(0)
    jamon1=jamon.permute(1,2,0)
    jamon2  = jamon1.cpu().numpy()
    jamon3= (jamon2[:,:,0]).tolist()
    cv2.imwrite(impath, jamon2)
    return jamon3