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


PATH_SUB = '/home/pka/kaggle/Melanoma-Classification/submit'
PATH = '/home/pka/kaggle/Melanoma-Classification/input'
PATH_LOG = '/home/pka/kaggle/Melanoma-Classification/log'
PATH_MODEL = '/home/pka/kaggle/Melanoma-Classification/model'
PATH_PNG_224 = '/home/pka/kaggle/Melanoma-Classification/input/train'
PATH_JPG_512 = '/home/pka/kaggle/Melanoma-Classification/input/train512'
PATH_JPG_512_TEST = '/home/pka/kaggle/Melanoma-Classification/input/test512'



def train_func(dataloader, model, f = 'mean'):
  pred = []
  bar = tqdm(dataloader)
  for img in bar:
    img = img.to(device)
    bs, ncrops, c, h, w = img.size()            
    y_ = model(img.view(-1, c, h, w))
    if f == 'mean':
      y_avg = y_.view(bs, ncrops, -1).mean(1)
      pred.append(y_avg.view(-1))      
    else:
      pr = y_.view(bs, ncrops, -1)
      pr = pr.cpu()
      idx  = np.argmax((abs(0.5 - pr)), axis = 1)
      y_avg  = torch.tensor([pr[i, idx[i]].item() for i in range(len(pr))])
      pred.append(y_avg)
  return pred

#meta
# def train_func(dataloader, model):
#   pred = []
#   bar = tqdm(dataloader)
#   for img, meta in bar:
#     img = img.to(device) 
#     meta = meta.to(device)
#     meta = meta.repeat(5,1,1)
#     bs, ncrops, c, h, w = img.size()
#     #print(img.view(-1, c, h, w).shape, meta.shape)            
#     y_ = model(img.view(-1, c, h, w), meta.view(-1, 11))
#     y_avg = y_.view(bs, ncrops, -1).mean(1)               
#     pred.append(y_avg.view(-1))
#   return pred

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
      A.Resize(224,224, p =1),
      A.VerticalFlip(p=1),
      A.HorizontalFlip(p=1), 
      # A.Flip(),
      # A.RandomBrightnessContrast(
      #       brightness_limit=0.2, 
      #       contrast_limit=0.2,
      #       brightness_by_max=True,
      #       always_apply=False,
      #       p=0.5
      # )
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





    test_df = pd.read_csv(os.path.join(PATH, 'test_meta.csv'))
    #testd = meta_trainDataset(test_df, PATH_JPG_512_TEST, transform = transform_test)
    testd = ttaDataset(test_df, PATH_JPG_512_TEST, transform = trf, transform2= trf2)
    testl =  DataLoader(testd, batch_size=16, sampler=SequentialSampler(testd), num_workers = 4)
  
    model = MODEL_HUB['eff']
    #name = 'eff_bz32_lr0.0001_shlReduceLROnPlateau_opAdam_lfBCEWithLogitsLoss_f0_epoch3_score0.643_best_fold.pth'
    
    list_names =  [
      'eff_bz32_lr0.0001_shlReduceLROnPlateau_opAdam_lfFocalLoss_f0_epoch18_score0.903_best_fold',
      'eff_bz32_lr0.0001_shlReduceLROnPlateau_opAdam_lfFocalLoss_f1_epoch12_score0.894_best_fold',
      'eff_bz32_lr0.0001_shlReduceLROnPlateau_opAdam_lfFocalLoss_f2_epoch8_score0.872_best_fold',
      'eff_bz32_lr0.0001_shlReduceLROnPlateau_opAdam_lfFocalLoss_f3_epoch16_score0.916_best_fold',
      'eff_bz32_lr0.0001_shlReduceLROnPlateau_opAdam_lfFocalLoss_f4_epoch5_score0.888_best_fold'
    ]
    
    # list_names =  [
    #      'res50_bz32_lr0.0001_shlReduceLROnPlateau_opAdam_lfFocalLoss_f0_epoch7_score0.920_best_fold',
    #      'res50_bz32_lr0.0001_shlReduceLROnPlateau_opAdam_lfFocalLoss_f1_epoch5_score0.909_best_fold',
    #      'res50_bz32_lr0.0001_shlReduceLROnPlateau_opAdam_lfFocalLoss_f2_epoch5_score0.906_best_fold',
    #      'res50_bz32_lr0.0001_shlReduceLROnPlateau_opAdam_lfFocalLoss_f3_epoch10_score0.922_best_fold',
    #      'res50_bz32_lr0.0001_shlReduceLROnPlateau_opAdam_lfFocalLoss_f4_epoch5_score0.909_best_fold'       
    #     ]

    

    
    temp = []
    score = 0
    for i in range(len(list_names)):
        print(f'load --> {list_names[i]}')
        model.load_state_dict(torch.load(os.path.join(PATH_MODEL, list_names[i] + '.pth')))
        model.to(device)
        model.eval()
        with torch.no_grad():
            pred = train_func(testl, model, f ='argmax')
            predicts = torch.cat(pred)
            temp.append(predicts.cpu().numpy()) 
                      
            name = list_names[i]
            if name.endswith('best_fold'):
                score += float(name[-15:-10])  

    print(f'Average scores: {score / 5}')
    f0, f1, f2, f3, f4 = temp
    p = (f0 + f1 + f2 + f3 + f4) / 5

    clock = '_'.join(time.ctime().split(':')) 

    #----------v1
    subm = pd.read_csv(os.path.join(PATH, 'sample_submission.csv'))
    subm['target'] = p
    subm.to_csv(os.path.join(PATH_SUB, f'{clock}_{name}_{score/5}.csv'), index=False)

    #----------v2
    subm = pd.read_csv(os.path.join(PATH, 'sample_submission.csv'))
    subm['target'] =torch.sigmoid(torch.tensor(p)).cpu().numpy()
    subm.to_csv(os.path.join(PATH_SUB, f'{clock}_{name}_{score/5}_submit_test2.csv'), index=False)
    print('Savedd.....!@#$%^&*()_')

    #One 

    # SEED = 13
    # seed_everything(SEED)
    # test_df = pd.read_csv(os.path.join(PATH, 'test_meta.csv'))
    # #testd = meta_trainDataset(test_df, PATH_JPG_512_TEST, transform = transform_test)
    # testd = ttaDataset(test_df, PATH_JPG_512_TEST, transform = trf, transform2= trf2)
    # testl =  DataLoader(testd, batch_size=16, sampler=SequentialSampler(testd), num_workers = 4)
  
    # model = MODEL_HUB['res50']
    # #name = 'eff_bz32_lr0.0001_shlReduceLROnPlateau_opAdam_lfBCEWithLogitsLoss_f0_epoch3_score0.643_best_fold.pth'
    

    # list_names =  [
    #   'res50_bz32_lr0.0001_shlReduceLROnPlateau_opAdam_lfFocalLoss_0.8661734639611427_final',
      
    # ]

    
    # temp = []
    # score = 0
    # for i in range(len(list_names)):
    #     print(f'load --> {list_names[i]}')
    #     model.load_state_dict(torch.load(os.path.join(PATH_MODEL, list_names[i] + '.pth')))
    #     model.to(device)
    #     model.eval()
    #     with torch.no_grad():
    #         pred = train_func(testl, model, f ='mean')
    #         predicts = torch.cat(pred)
    #     name = list_names[i]
    # p = predicts.cpu().numpy()
    
    

    # clock = '_'.join(time.ctime().split(':')) 


    # #----------v1
    # subm = pd.read_csv(os.path.join(PATH, 'sample_submission.csv'))
    # subm['target'] = p
    # subm.to_csv(os.path.join(PATH_SUB, f'{clock}_{name}_{score/5}.csv'), index=False)
    # print('#@$%^&*(')