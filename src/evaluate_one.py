import os
import cv2
import torch
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.cuda import amp
import albumentations as A
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler
#--my
from utils import seed_everything
from dataset import trainDataset, meta_trainDataset
from hub import MODEL_HUB   
device = torch.device("cuda")





PATH_SUB = '/home/pka/kaggle/Melanoma-Classification/submit'
PATH = '/home/pka/kaggle/Melanoma-Classification/input'
PATH_LOG = '/home/pka/kaggle/Melanoma-Classification/log'
PATH_MODEL = '/home/pka/kaggle/Melanoma-Classification/model'
PATH_PNG_224 = '/home/pka/kaggle/Melanoma-Classification/input/train'
PATH_JPG_512 = '/home/pka/kaggle/Melanoma-Classification/input/train512'
PATH_JPG_512_TEST = '/home/pka/kaggle/Melanoma-Classification/input/test512'



def train_func(dataloader, model):
  pred = []
  bar = tqdm(dataloader)
  for img in bar:
    img = img.to(device)    
    y_ = model(img)                 
    pred.append(y_.view(-1))
  return pred

# def train_func(dataloader, model):
#   pred = []
#   bar = tqdm(dataloader)
#   for img, meta in bar:
#     img = img.to(device) 
#     meta = meta.to(device)
#     y_ = model(img, meta)                 
#     pred.append(y_.view(-1))
#   return pred

transform_test = A.Compose([
    #A.Normalize(mean, std, max_pixel_value=255.0, always_apply=True)
    A.Resize(224,224, p =1)
])


if __name__ == "__main__":
    SEED = 13
    seed_everything(SEED)
    test_df = pd.read_csv(os.path.join(PATH, 'test_meta.csv'))
    # testd = meta_trainDataset(test_df, PATH_PNG_224_TEST, transform = transform_test)
    testd = trainDataset(test_df, PATH_JPG_512, transform = transform_test)
    testl =  DataLoader(testd, batch_size=16, sampler=SequentialSampler(testd), num_workers = 4)
  
    model = MODEL_HUB['res50']
    #name = 'eff_bz32_lr0.0001_shlReduceLROnPlateau_opAdam_lfBCEWithLogitsLoss_f0_epoch3_score0.643_best_fold.pth'
    

    list_names =  [
      'res50_bz32_lr0.0001_shlReduceLROnPlateau_opAdam_lfFocalLoss_0.8661734639611427_final',      
    ]

    
    temp = []
    score = 0
    for i in range(len(list_names)):
        print(f'load --> {list_names[i]}')
        model.load_state_dict(torch.load(os.path.join(PATH_MODEL, list_names[i] + '.pth')))
        model.to(device)
        model.eval()
        with torch.no_grad():
            pred = train_func(testl, model)
            predicts = torch.cat(pred)
        name = list_names[i]
    p = predicts.cpu().numpy()
    

    clock = '_'.join(time.ctime().split(':')) 


    #----------v1
    subm = pd.read_csv(os.path.join(PATH, 'sample_submission.csv'))
    subm['target'] = p
    subm.to_csv(os.path.join(PATH_SUB, f'{clock}_{name}_{score/5}.csv'), index=False)
    print('Savedd.....!@#$%^&*()_')