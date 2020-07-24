import os
import torch
import joblib
import numpy as np
from torch.utils.data import Dataset

PATH_PKL_128 = r'C:\Users\pka\kaggle\melanoma\input\siim-isic-melanoma-classification\pkl_128'

class trainDataset(Dataset):

    def __init__(self, data, transform = None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):       
        name = self.data.image_name.values[idx]        
        image = joblib.load(os.path.join(PATH_PKL_128, f'{name}.pkl'))
        if self.transform:
            image = self.transform(image)
        image = image.astype(np.float32)
        image /= 255
        image = np.transpose(image, (2,0,1))
        target = self.data.target.values[idx]      
        
        return torch.tensor(image), torch.tensor(target, dtype = torch.float)

