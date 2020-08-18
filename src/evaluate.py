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

def train_func_meta(dataloader, model):
  pred = []
  bar = tqdm(dataloader)
  for img, meta in bar:
    img = img.to(device) 
    meta = meta.to(device)
    y_ = model(img, meta)                 
    pred.append(y_.view(-1))
  return pred

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225) 

transform_test = A.Compose([
  A.Normalize(mean, std, max_pixel_value=255.0, always_apply=True),
  A.Resize(224,224, p =1)
])


if __name__ == "__main__":

  meta = True

  SEED = 13
  seed_everything(SEED)
  test_df = pd.read_csv(os.path.join(PATH, 'test_meta.csv'))
  if meta:
    testd = meta_trainDataset(test_df, PATH_JPG_512_TEST, transform = transform_test)
    func = train_func_meta
  else:
    testd = trainDataset(test_df, PATH_JPG_512_TEST, transform = transform_test, transform2=None)
    func = train_func
  testl =  DataLoader(testd, batch_size=16, sampler=SequentialSampler(testd), num_workers = 4)

  model = MODEL_HUB['senet154_meta']  
  
  # list_names =  [
  #      'res50_bz32_lr0.0001_shlReduceLROnPlateau_opAdam_lfFocalLoss_f0_epoch7_score0.920_best_fold',
  #      'res50_bz32_lr0.0001_shlReduceLROnPlateau_opAdam_lfFocalLoss_f1_epoch5_score0.909_best_fold',
  #      'res50_bz32_lr0.0001_shlReduceLROnPlateau_opAdam_lfFocalLoss_f2_epoch5_score0.906_best_fold',
  #      'res50_bz32_lr0.0001_shlReduceLROnPlateau_opAdam_lfFocalLoss_f3_epoch10_score0.922_best_fold',
  #      'res50_bz32_lr0.0001_shlReduceLROnPlateau_opAdam_lfFocalLoss_f4_epoch5_score0.909_best_fold'       
  #     ]

  list_names =  [
    '0.9036254840777491_final',
    '0.91520_best_fold',
    '0.884279_best_fold',
    '0.9036254840777491_final',    
  ]

  
  temp = []
  score = 0
  for i in range(len(list_names)):
      print(f'load --> {list_names[i]}')
      model.load_state_dict(torch.load(os.path.join(PATH_MODEL, list_names[i] + '.pth')))
      model.to(device)
      model.eval()
      with torch.no_grad():
          pred = func(testl, model)
          predicts = torch.cat(pred)
          temp.append(predicts.cpu().numpy()) 
                    
          name = list_names[i]
          if name.endswith('best_fold'):
            try:
              score += float(name[-15:-10])  
            except:
              print('len value small')
              score += float(name[-14:-10])


  print(f'Average scores: {score / 4}')
  f0, f1, f2, f3 = temp
  p = (f0 + f1 + f2 + f3) / 4
  

  clock = '_'.join(time.ctime().split(':')) 


  #----------v1
  subm = pd.read_csv(os.path.join(PATH, 'sample_submission.csv'))
  subm['target'] = p
  subm.to_csv(os.path.join(PATH_SUB, f'{clock}_{name}_{score/4}.csv'), index=False)

  # #----------v2
  # subm = pd.read_csv(os.path.join(PATH, 'sample_submission.csv'))
  # subm['target'] =torch.sigmoid(torch.tensor(p)).cpu().numpy()
  # subm.to_csv(os.path.join(PATH_SUB, f'{clock}_{name}_{score/5}_submit_test2.csv'), index=False)
  # print('Savedd.....!@#$%^&*()_')