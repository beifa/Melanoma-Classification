import os
import torch
import joblib
import numpy as np
from torch.utils.data import Dataset

PATH_PNG_224 = r'C:\Users\pka\kaggle\melanoma\input\siim-isic-melanoma-classification\png_224\train'

class trainDataset(Dataset):

    def __init__(self, data, transform = None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
       
        name = self.data.image_name.values[idx]
        
       
        image = cv2.imread(os.path.join(PATH_PNG_224, f'{name}.png'))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            image = self.transform(image=image)['image']
        image = image.astype(np.float32)
        image /= 255
        image = np.transpose(image, (2, 0, 1))
        target = self.data.target.values[idx]
      
        
        return torch.tensor(image), torch.tensor(target, dtype = torch.float)

