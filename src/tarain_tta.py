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
from torchvision import transforms
from torch.utils.data import Dataset
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from torch.utils.data.sampler import RandomSampler, SequentialSampler

from utils import seed_everything
from dataset import trainDataset, ttaDataset
from early_stopping import EarlyStopping 

from hub import MODEL_HUB   

import warnings
warnings.simplefilter("ignore")

import argparse


PATH = '/home/pka/kaggle/melanoma/input'
PATH_LOG = '/home/pka/kaggle/melanoma/log'
PATH_MODEL = '/home/pka/kaggle/melanoma/model'
PATH_PNG_224 = '/home/pka/kaggle/melanoma/input/train'

device = torch.device("cuda")

def calc_loss(loss_func, target, pred, opt, scaler):
    """
    criterion : name loss func
    
    """
    list_func_loss = {
        'BCELoss' : nn.BCELoss(),
        'BCEWithLogitsLoss' : nn.BCEWithLogitsLoss()
    }    
   
    if loss_func == 'BCELoss':
        #0-1
        loss = list_func_loss[loss_func](torch.sigmoid(pred).view(-1), target)      
    else:       
        loss = list_func_loss[loss_func](pred.view(-1), target)    
    if opt is not None:
        if scaler is not None:            
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()            
        else:            
            loss.backward()
            opt.step()
            opt.zero_grad()
    return loss.item()


def accuracy(labels, predict):    
    #take tensor
    return ((predict == labels).sum().item()) / len(labels) 

def matrix_scores(val):
    # val : out matrix confusion_matrix sklearn
    
    assert val.shape == (2, 2), 'data error'
    tn, fp, fn, tp = val.ravel()    
    recall = tp/(tp+fn)
    precision = tp/(tp+fp)  
    if (recall + precision) == 0:
        F1 = 0
    else:
        F1 = (2 * (recall * precision)) / (recall + precision)
    return recall, precision, F1, tp, fn

def help_scaler(model, img, tr, loss_func, opt, scaler):
    if scaler is not None:
        opt.zero_grad()
        with amp.autocast():
            bs, ncrops, c, h, w = img.size()            
            y_ = model(img.view(-1, c, h, w))
            y_avg = y_.view(bs, ncrops, -1).mean(1)
            loss = calc_loss(loss_func, tr, y_avg, opt, scaler)   
    else:        
        bs, ncrops, c, h, w = img.size()            
        y_ = model(img.view(-1, c, h, w))
        y_avg = y_.view(bs, ncrops, -1).mean(1)
        loss = calc_loss(loss_func, tr, y_avg, opt, None)
    return y_, loss
    

def train_func(dataloader, model, loss_func, opt, scaler):    
    train_loss = []
    pred = []
    label = []
    bar = tqdm(dataloader)
    for (img, tr) in bar:
        img, tr = img.to(device), tr.to(device)
        if opt is not None:
            y_, loss = help_scaler(model, img, tr, loss_func, opt, scaler)           
        else:                     
            y_ = model(img)            
            loss = calc_loss(loss_func, tr, y_, opt, None)       

        train_loss.append(loss)          
        pred.append(y_.view(-1))
        label.append(tr)   
        
    if opt:
        return train_loss
    else:
        return train_loss, pred, label



if __name__ == "__main__":

    trf = transforms.Compose([
    transforms.FiveCrop((224,224)), # this is a list of PIL Images   
    
    transforms.Lambda(lambda crops:     
                      torch.stack([transforms.ToTensor()(crop) for crop in crops]))
                      ])

    trf2 = A.Compose([

        A.Flip(),
        
        A.ShiftScaleRotate(shift_limit=0.055,
                        scale_limit=0.21,
                        rotate_limit=180,
                        p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=0.2, 
            contrast_limit=0.2,
            brightness_by_max=True,
            always_apply=False,
            p=0.5
        ),
        A.Downscale(
            scale_min=0.15,
            scale_max=0.25,
            interpolation=0,
            always_apply=False,
            p=0.2)
    ])

    transform_val = A.Compose([])
    
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('-f', '--fold', type=int, help = 'add fold number', default=0)    
    parser.add_argument('-e', '--epoch', type=int, help = 'add count epoch', default= 10)    
    parser.add_argument('-n', '--num_workers', type=int, help = 'num_workers', default= 4)
    parser.add_argument('-m', '--model', type=str, help = 'name model to train : res50, eff(name, out)', default = 'res50')
    parser.add_argument('-l', '--loss_func', type=str, help = 'loss func : BCEWithLogitsLoss, BCELoss ', default = 'BCEWithLogitsLoss')
    parser.add_argument('-d', '--debag', type=bool, help = 'small data set', default = False)
    


    pars = parser.parse_args()

    params = {
        'SEED': 13,
        'batch_size': 8,
        'lr': 1e-4,
        'num_workers' : pars.num_workers,
        'epoch': pars.epoch,
        'fold': pars.fold,
        'model': pars.model,
        'loss_func' : pars.loss_func
    }
    log = True
    seed_everything(params['SEED'])    
    model = MODEL_HUB[params['model']]
    kernel = params['model']   
    model.to(device)  

    if pars.debag:
        print('Debag....')
        print(pars)
        df = pd.read_csv(os.path.join(PATH, 'train_folds.csv')).head(1000)
    else:
        df = pd.read_csv(os.path.join(PATH, 'train_folds.csv'))
    tr_idx = np.where(df.fold != params['fold'])
    vl_idx = np.where(df.fold == params['fold'])

    td = ttaDataset(df.loc[tr_idx], PATH_PNG_224, transform= trf, transform2=trf2)
    vd = trainDataset(df.loc[vl_idx], PATH_PNG_224, transform= transform_val)


    dl =  DataLoader(td, batch_size=params['batch_size'], sampler=RandomSampler(td),  drop_last=True, num_workers=params['num_workers'])
    vl =  DataLoader(vd, batch_size=params['batch_size'], sampler=SequentialSampler(vd), num_workers=params['num_workers'])


    opt = optim.Adam(model.parameters(), lr = params['lr'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, params['epoch'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                               opt, mode='max', factor=0.1,
                               patience = 5, verbose=True
                               )
    #'BCEWithLogitsLoss', 'BCELoss'
    loss_func =  params['loss_func']
    scaler = amp.GradScaler()  
    print(f"Fold num: {params['fold']}")  
    roc_baseline = 0

    #init_param_model = name,  fold lr loss func arg seed size image, scheduler
    init = f"{params['model']}_bz{params['batch_size']}_lr{params['lr']}_shl{type(scheduler).__name__}_op{type(opt).__name__}_lf{params['loss_func']}"
    es = EarlyStopping(5, 'max')  
    for e in range(params['epoch']):        
        print('Epoch ....... ', e)   
        model.train()
        if loss_func == 'BCELoss': scaler = None
        loss_train = train_func(dl, model, loss_func, opt, scaler)

        model.eval()
        with torch.no_grad():
            loss_val, pred, label = train_func(vl, model, loss_func, opt= None, scaler=None)  
       
        predicts = torch.cat(pred)
        labels = torch.cat(label)
        
        roc = roc_auc_score(labels.cpu().numpy(), predicts.cpu().numpy())    
        predicts = torch.sigmoid(predicts).round()
        cm_out = confusion_matrix(labels.cpu().numpy(), predicts.cpu().numpy())
        recall, precision, F1,  tp, fn  = matrix_scores(cm_out)

        if log:
            print('log...')
            l_rate = opt.param_groups[0]['lr']
            lg = time.ctime() + ' ' +  f'Fpoch: {e}, lr: {l_rate}, ratio : {tp/fn}, rocauc: {roc}'
            print(lg)
            with open(os.path.join(PATH_LOG, f'log_{kernel}.txt'), 'a') as app:
                app.write(lg + '\n')
        if roc > roc_baseline:
            print('Best ({:.6f} --> {:.6f}).  Saving model ...'.format(roc_baseline, roc))            
            torch.save(model.state_dict(), os.path.join(PATH_MODEL, f"{init}_f{params['fold']}_epoch{e}_score{roc.round(3)}_best_fold.pth"))
            roc_baseline = roc
        print('------------')
        print(f'Recall {recall}, precision {precision}, F1 {F1}')
        print(f'TruePositive : {tp}, FalseNegative : {fn}, ratio : {tp/fn}')
        print(f'train loss: {np.mean(loss_train)} ---- val loss {np.mean(loss_val)}')
        print('\n')
        torch.cuda.empty_cache()   
        scheduler.step(roc)
        #scheduler.step(e -1)
        es(roc, model, PATH_MODEL)
        if es.early_stop:
            print('Erlystopp')
            break
    torch.save(model.state_dict(), os.path.join(PATH_MODEL, f'{init}_{roc}_final.pth'))