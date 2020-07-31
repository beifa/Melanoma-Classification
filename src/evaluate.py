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
from dataset import trainDataset
from hub import MODEL_HUB   
device = torch.device("cuda")


PATH = r'C:\Users\pka\kaggle\melanoma\input\siim-isic-melanoma-classification'
PATH_MODEL = r'C:\Users\pka\kaggle\melanoma\model'
PATH_SUB = r'C:\Users\pka\kaggle\melanoma\submit'
PATH_PNG_224_TEST = r'C:\Users\pka\kaggle\melanoma\input\siim-isic-melanoma-classification\png_224\test'



def train_func(dataloader, model):
  pred = []
  bar = tqdm(dataloader)
  for img in bar:
    img = img.to(device)
    y_ = model(img)                 
    pred.append(y_.view(-1))
  return pred

transform_test = A.Compose([
    #A.Normalize(mean, std, max_pixel_value=255.0, always_apply=True)
])


if __name__ == "__main__":
    SEED = 13
    seed_everything(SEED)
    test_df = pd.read_csv(os.path.join(PATH, 'test.csv'))
    testd = trainDataset(test_df, PATH_PNG_224_TEST, transform = transform_test)
    testl =  DataLoader(testd, batch_size=16, sampler=SequentialSampler(testd), num_workers = 4)
  
    model = MODEL_HUB['eff']
    #name = 'eff_bz32_lr0.0001_shlReduceLROnPlateau_opAdam_lfBCEWithLogitsLoss_f0_epoch3_score0.643_best_fold.pth'
    
    list_names =  [
        'eff_bz32_lr0.0001_shlReduceLROnPlateau_opAdam_lfBCEWithLogitsLoss_f0_epoch1_score0.857_best_fold',
        'eff_bz32_lr0.0001_shlReduceLROnPlateau_opAdam_lfBCEWithLogitsLoss_f1_epoch4_score0.892_best_fold',
        'eff_bz32_lr0.0001_shlReduceLROnPlateau_opAdam_lfBCEWithLogitsLoss_f2_epoch4_score0.881_best_fold',
        'eff_bz32_lr0.0001_shlReduceLROnPlateau_opAdam_lfBCEWithLogitsLoss_f3_epoch4_score0.878_best_fold',
        'eff_bz32_lr0.0001_shlReduceLROnPlateau_opAdam_lfBCEWithLogitsLoss_f4_epoch3_score0.884_best_fold'
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
        temp.append(predicts.cpu().numpy())
        name = list_names[i]
        if name.endswith('best_fold'):
            score += float(name[-15:-10])  
    print(f'Average scores: {score / 5}')
    f0, f1, f2, f3, f4 = temp
    p = (f0 + f1 + f2 + f3 + f4) / 5


    clock = '_'.join(time.ctime().split(':')) 
    #save
    # if name.endswith('best_fold.pth'):
    #     score = name[-19:-14]
    # else:
    #     score = name[-15:-10]

    #----------v1
    subm = pd.read_csv(os.path.join(PATH, 'sample_submission.csv'))
    subm['target'] = p
    subm.to_csv(os.path.join(PATH_SUB, f'{clock}_{name}_{score/5}.csv'), index=False)

    #----------v2
    subm = pd.read_csv(os.path.join(PATH, 'sample_submission.csv'))
    subm['target'] =torch.sigmoid(torch.tensor(p)).cpu().numpy()
    subm.to_csv(os.path.join(PATH_SUB, f'{clock}_{name}_{score/5}_submit_test2.csv'), index=False)
    print('Savedd.....!@#$%^&*()_')