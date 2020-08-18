import os
import cv2
import torch
import numpy as np
import PIL.Image as Image
from torch.utils.data import Dataset
import random

class trainDataset(Dataset):
    def __init__(self, data, path, transform = None, transform2 = None):
        self.data = data
        self.path = path
        self.transform = transform
        self.transform2 = transform2

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
       
        name = self.data.image_name.values[idx]      
        image = Image.open(os.path.join(self.path, f'{name}.jpg'))
        image = np.array(image)

        if self.transform is not None:
            image = self.transform(image=image)['image']
        
        if self.transform2 is not None:
            # image = self.transform2(image, random.uniform(0.9, 1))
            image = self.transform2(image)

        image = image.astype(np.float32)
        # image /= 255
        image = np.transpose(image, (2, 0, 1))
        if 'test' in self.path:
            return torch.tensor(image)
        target = self.data.target.values[idx]
        return torch.tensor(image), torch.tensor(target, dtype = torch.float)


# check TTA
class ttaDataset(Dataset):
    def __init__(self, data, path, transform = None, transform2 = None):
        self.data = data
        self.path = path
        self.transform = transform
        self.transform2 = transform2

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        temp = []
        name = self.data.image_name.values[idx]        
        image = Image.open(os.path.join(self.path, f'{name}.jpg'))
        if self.transform: 
            image = self.transform(image)   # crop            
            for i in range(image.shape[0]):                
                if self.transform2:
                    img = np.transpose(image[i], (1,2,0))
                    img = img.numpy().astype(np.float32)
                    img = self.transform2(image = img)['image'] # crop
                    img = np.transpose(img, (2, 0, 1))                 
                    temp.append(img)
            image = torch.stack([torch.tensor(x) for x in temp])        
        
        if 'test' in self.path:
            return torch.tensor(image)
        target = self.data.target.values[idx]
        return torch.tensor(image), torch.tensor(target, dtype = torch.float)



class meta_trainDataset(Dataset):
     
  def __init__(self, data, path, transform = None):
      self.data = data
      self.path = path
      self.transform = transform
      

  def __len__(self):
      return len(self.data)

  def __getitem__(self, idx):
      
      name = self.data.image_name.values[idx]     
      image = Image.open(os.path.join(self.path, f'{name}.jpg'))
      image = np.array(image)
      if self.transform is not None:
          image = self.transform(image=image)['image']      
      image = image.astype(np.float32)
      image /= 255
      image = np.transpose(image, (2, 0, 1))        

      if 'test' in self.path: 
          meta = self.data.iloc[idx].drop('image_name').values.astype(np.float32)
          return torch.tensor(image), meta

      meta = self.data.iloc[idx].drop(['target', 'fold', 'image_name']).values.astype(np.float32)
      target = self.data.target.values[idx]
      return torch.tensor(image), torch.tensor(meta), torch.tensor(target, dtype = torch.float)


class meta_ttaDataset(Dataset):
    def __init__(self, data, path, transform = None, transform2 = None):
        self.data = data
        self.path = path
        self.transform = transform
        self.transform2 = transform2

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        temp = []
        name = self.data.image_name.values[idx]        
        image = Image.open(os.path.join(self.path, f'{name}.png'))
        if self.transform: 
            image = self.transform(image)   # crop            
            for i in range(image.shape[0]):                
                if self.transform2:
                    img = np.transpose(image[i], (1,2,0))
                    img = img.numpy().astype(np.float32)
                    img = self.transform2(image = img)['image'] # crop
                    img = np.transpose(img, (2, 0, 1))                 
                    temp.append(img)
            image = torch.stack([torch.tensor(x) for x in temp])        
        
        if 'test' in self.path:
            meta = self.data.iloc[idx].drop('image_name').values.astype(np.float32)
            return torch.tensor(image), meta
        target = self.data.target.values[idx]
        return torch.tensor(image), torch.tensor(target, dtype = torch.float)