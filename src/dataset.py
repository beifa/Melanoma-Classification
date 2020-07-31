import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

class trainDataset(Dataset):
    def __init__(self, data, path, transform = None):
        self.data = data
        self.path = path
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
       
        name = self.data.image_name.values[idx]      
        image = cv2.imread(os.path.join(self.path, f'{name}.png'))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            image = self.transform(image=image)['image']
        image = image.astype(np.float32)
        image /= 255
        image = np.transpose(image, (2, 0, 1))
        if 'test' in self.path:
            return torch.tensor(image)
        target = self.data.target.values[idx]
        return torch.tensor(image), torch.tensor(target, dtype = torch.float)


