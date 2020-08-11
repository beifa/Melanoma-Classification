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




PATH_SUB = '/home/pka/kaggle/melanoma/'
PATH = '/home/pka/kaggle/melanoma/input'
PATH_SUB = '/home/pka/kaggle/melanoma/submit'
PATH_MODEL = '/home/pka/kaggle/melanoma/model'
PATH_MODEL_TTA = '/home/pka/kaggle/melanoma/backmodel/tta'
PATH_PNG_224_TEST = '/home/pka/kaggle/melanoma/input/test'



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
])


if __name__ == "__main__":
    SEED = 13
    seed_everything(SEED)
    test_df = pd.read_csv(os.path.join(PATH, 'test_meta.csv'))
    # testd = meta_trainDataset(test_df, PATH_PNG_224_TEST, transform = transform_test)
    testd = trainDataset(test_df, PATH_PNG_224_TEST, transform = transform_test)
    testl =  DataLoader(testd, batch_size=16, sampler=SequentialSampler(testd), num_workers = 4)
  
    model = MODEL_HUB['res50']
    #name = 'eff_bz32_lr0.0001_shlReduceLROnPlateau_opAdam_lfBCEWithLogitsLoss_f0_epoch3_score0.643_best_fold.pth'
    

    list_names =  [
      'res50_bz32_lr0.0001_shlReduceLROnPlateau_opAdam_lfFocalLoss_f0_epoch5_score0.901_best_fold',      
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