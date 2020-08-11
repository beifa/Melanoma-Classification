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
from torchvision import transforms
#--my
from utils import seed_everything
from dataset import trainDataset, ttaDataset, meta_ttaDataset
from hub import MODEL_HUB   
device = torch.device("cuda")


PATH_SUB = '/home/pka/kaggle/melanoma/'
PATH = '/home/pka/kaggle/melanoma/input'
PATH_SUB = '/home/pka/kaggle/melanoma/submit'
PATH_MODEL = '/home/pka/kaggle/melanoma/model'
PATH_MODEL_TTA = '/home/pka/kaggle/melanoma/backmodel/tta'
PATH_PNG_224_TEST = '/home/pka/kaggle/melanoma/input/test'



# def train_func(dataloader, model):
#   pred = []
#   bar = tqdm(dataloader)
#   for img in bar:
#     img = img.to(device)
#     bs, ncrops, c, h, w = img.size()            
#     y_ = model(img.view(-1, c, h, w))
#     y_avg = y_.view(bs, ncrops, -1).mean(1)               
#     pred.append(y_avg.view(-1))
#   return pred


def train_func(dataloader, model):
  pred = []
  bar = tqdm(dataloader)
  for img, meta in bar:
    img = img.to(device) 
    meta = meta.to(device)
    meta = meta.repeat(5,1,1)
    bs, ncrops, c, h, w = img.size()
    #print(img.view(-1, c, h, w).shape, meta.shape)            
    y_ = model(img.view(-1, c, h, w), meta.view(-1, 11))
    y_avg = y_.view(bs, ncrops, -1).mean(1)               
    pred.append(y_avg.view(-1))
  return pred

if __name__ == "__main__":
    SEED = 13
    seed_everything(SEED)

    trf = transforms.Compose([
        transforms.FiveCrop((224,224)), # this is a list of PIL Images  
        transforms.Lambda(lambda crops:     
                      torch.stack([transforms.ToTensor()(crop) for crop in crops]))
                      ])
    #default
    trf2 = A.Compose([

        # A.Flip(),        
        # A.ShiftScaleRotate(shift_limit=0.055,
        #                 scale_limit=0.21,
        #                 rotate_limit=180,
        #                 p=0.5),

        #test bad
        # A.VerticalFlip(p=1),
        # A.HorizontalFlip(p=1)

        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=45),
        
    ])

    # trf2 = A.Compose([

    #     A.Flip(),
        
    #     A.ShiftScaleRotate(shift_limit=0.055,
    #                     scale_limit=0.21,
    #                     rotate_limit=180,
    #                     p=0.5),
    #     A.RandomBrightnessContrast(
    #         brightness_limit=0.2, 
    #         contrast_limit=0.2,
    #         brightness_by_max=True,
    #         always_apply=False,
    #         p=0.5
    #     ),
    #     A.Downscale(
    #         scale_min=0.15,
    #         scale_max=0.25,
    #         interpolation=0,
    #         always_apply=False,
    #         p=0.2)
    # ])





    # test_df = pd.read_csv(os.path.join(PATH, 'test.csv'))
    # testd = ttaDataset(test_df, PATH_PNG_224_TEST, transform = trf, transform2= trf2)
    # testl =  DataLoader(testd, batch_size=8, sampler=SequentialSampler(testd), num_workers = 0)
  
    # model = MODEL_HUB['res50']
    # #name = 'eff_bz32_lr0.0001_shlReduceLROnPlateau_opAdam_lfBCEWithLogitsLoss_f0_epoch3_score0.643_best_fold.pth'
    
    # # list_names =  [
    # #     'eff_bz32_lr0.0001_shlReduceLROnPlateau_opAdam_lfBCEWithLogitsLoss_f0_epoch1_score0.857_best_fold',
    # #     'eff_bz32_lr0.0001_shlReduceLROnPlateau_opAdam_lfBCEWithLogitsLoss_f1_epoch4_score0.892_best_fold',
    # #     'eff_bz32_lr0.0001_shlReduceLROnPlateau_opAdam_lfBCEWithLogitsLoss_f2_epoch4_score0.881_best_fold',
    # #     'eff_bz32_lr0.0001_shlReduceLROnPlateau_opAdam_lfBCEWithLogitsLoss_f3_epoch4_score0.878_best_fold',
    # #     'eff_bz32_lr0.0001_shlReduceLROnPlateau_opAdam_lfBCEWithLogitsLoss_f4_epoch3_score0.884_best_fold'
    # #     ]
    
    # list_names =  [
    #     'res50_bz8_lr0.0001_shlReduceLROnPlateau_opAdam_lfBCEWithLogitsLoss_f0_epoch10_score0.890_best_fold',
    #     'res50_bz8_lr0.0001_shlReduceLROnPlateau_opAdam_lfBCEWithLogitsLoss_f1_epoch10_score0.895_best_fold',
    #     'res50_bz8_lr0.0001_shlReduceLROnPlateau_opAdam_lfBCEWithLogitsLoss_f2_epoch10_score0.900_best_fold',
    #     'res50_bz8_lr0.0001_shlReduceLROnPlateau_opAdam_lfBCEWithLogitsLoss_f3_epoch8_score0.898_best_fold',
    #     'res50_bz8_lr0.0001_shlReduceLROnPlateau_opAdam_lfBCEWithLogitsLoss_f4_epoch5_score0.896_best_fold' 
       
    #     ]

    

    
    # temp = []
    # score = 0
    # for i in range(len(list_names)):
    #     print(f'load --> {list_names[i]}')
    #     model.load_state_dict(torch.load(os.path.join(PATH_MODEL_TTA, list_names[i] + '.pth')))
    #     model.to(device)
    #     model.eval()
    #     with torch.no_grad():
    #         pred = train_func(testl, model)
    #         predicts = torch.cat(pred)
    #         temp.append(predicts.cpu().numpy()) 
                      
    #         name = list_names[i]
    #         if name.endswith('best_fold'):
    #             score += float(name[-15:-10])  

    # print(f'Average scores: {score / 5}')
    # f0, f1, f2, f3, f4 = temp
    # p = (f0 + f1 + f2 + f3 + f4) / 5

    # clock = '_'.join(time.ctime().split(':')) 

    # #----------v1
    # subm = pd.read_csv(os.path.join(PATH, 'sample_submission.csv'))
    # subm['target'] = p
    # subm.to_csv(os.path.join(PATH_SUB, f'{clock}_{name}_{score/5}.csv'), index=False)

    # #----------v2
    # subm = pd.read_csv(os.path.join(PATH, 'sample_submission.csv'))
    # subm['target'] =torch.sigmoid(torch.tensor(p)).cpu().numpy()
    # subm.to_csv(os.path.join(PATH_SUB, f'{clock}_{name}_{score/5}_submit_test2.csv'), index=False)
    # print('Savedd.....!@#$%^&*()_')

    SEED = 13
    seed_everything(SEED)
    test_df = pd.read_csv(os.path.join(PATH, 'test_meta.csv'))
    # testd = meta_trainDataset(test_df, PATH_PNG_224_TEST, transform = transform_test)
    testd = meta_ttaDataset(test_df, PATH_PNG_224_TEST, transform = trf, transform2= trf2)
    testl =  DataLoader(testd, batch_size=16, sampler=SequentialSampler(testd), num_workers = 4)
  
    model = MODEL_HUB['res50_meta']
    #name = 'eff_bz32_lr0.0001_shlReduceLROnPlateau_opAdam_lfBCEWithLogitsLoss_f0_epoch3_score0.643_best_fold.pth'
    

    list_names =  [
      'res50_meta_bz32_lr0.0003_shlReduceLROnPlateau_opAdam_lfBCEWithLogitsLoss_f0_epoch15_score0.909_best_fold',
      
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
    print('#@$%^&*(')