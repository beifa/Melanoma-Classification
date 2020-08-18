import os
import torch
from torch import nn
import pretrainedmodels
import torch.nn.functional as F
from utils import Mish
from efficientnet_pytorch import model as eff_net


pre_train = {
    'efficientnet-b0' : 'efficientnet-b0-355c32eb.pth',
    'efficientnet-b1' : 'efficientnet-b1-f1951068.pth',
    'efficientnet-b2' : 'efficientnet-b2-8bb594d6.pth',
    'efficientnet-b3' : 'efficientnet-b3-5fb5a3c3.pth',
    'efficientnet-b4' : 'efficientnet-b4-6ed6700e.pth',
    }

class Res50(nn.Module):
    
    def __init__(self, name):
        super(Res50, self).__init__()        
        self.model = pretrainedmodels.__dict__[name](pretrained = 'imagenet')       
        self.l1 = nn.Linear(2048, 1)
        
        
    def forward(self, x):      
        bs, _, _, _ = x.shape
        x = self.model.features(x)
        x = F.adaptive_avg_pool2d(x, 1).view(bs, -1)
        x = self.l1(x)
        return x


class Eff_b_(nn.Module):
    
    def __init__(self, name, out):
        super(Eff_b_, self).__init__()
        self.eff_net = eff_net.EfficientNet.from_pretrained(name)
        self.fc = nn.Linear(self.eff_net._fc.out_features, out)
        
    def current_net(self, x):
        return self.eff_net(x)
    
    def forward(self, x):
        x = self.current_net(x)
        x = self.fc(x)
        return x


class Res50_meta(nn.Module):
    
    def __init__(self):
        super(Res50_meta, self).__init__()        
        self.model = pretrainedmodels.__dict__['se_resnext50_32x4d'](pretrained = 'imagenet')       
        self.l1 = nn.Linear(2048, 1024)
        self.meta = nn.Sequential(
            nn.Linear(11, 111), 
            nn.BatchNorm1d(111),
            nn.ReLU(),
            nn.Linear(111, 11),
            nn.BatchNorm1d(11),
            nn.ReLU()

        )
        self.l0 = nn.Linear(1024 + 11, 1)
        
        
    def forward(self, x, meta):      
        bs, _, _, _ = x.shape

        x = self.model.features(x)        
        x = F.adaptive_avg_pool2d(x, 1).view(bs, -1)         
        x = self.l1(x)
        f = self.meta(meta)        
        out = torch.cat((x, f), axis = 1)    
        out = self.l0(out)         
        return out


class Eff_b_meta_v2(nn.Module):
    
    def __init__(self, name, out):
        super(Eff_b_meta_v2, self).__init__()
        self.eff_net = eff_net.EfficientNet.from_pretrained(name)
        self.l1 = nn.Linear(self.eff_net._fc.out_features, 1024)
        # self.meta = nn.Sequential(
        #                 nn.Linear(11, 1024), 
        #                 nn.BatchNorm1d(1024),
        #                 Mish()              
        #                 )
        
        self.meta = nn.Sequential(
                        nn.Linear(11, 512),                    
                        nn.Linear(512, 1536),
                        nn.BatchNorm1d(1536),
                        nn.Linear(1536, 1024),
                        nn.Dropout(0.2),
                        Mish()          
                        )
        self.fc = nn.Linear(1024, out)
        
    def current_net(self, x):
        return self.eff_net(x)
    
    def forward(self, x, meta):
        x = self.current_net(x)
        x = self.l1(x)        
        f = self.meta(meta)  
        out = x * f        
        out = self.fc(out)
       
        return out


class Res50meta_v2(nn.Module):
    
    def __init__(self, name):
        super(Res50meta_v2, self).__init__()
        self.model = pretrainedmodels.__dict__[name](pretrained = 'imagenet')        
        # self.model = pretrainedmodels.__dict__['senet154'](pretrained=None)        
        # self.model.load_state_dict(
        #     torch.load(
        #         "../input/pretrained-model-weights-pytorch/senet154-c7b49a05.pth"
        #     )
        # )
        self.l1 = nn.Linear(2048, 1024)               
        self.meta = nn.Sequential(
                        nn.Linear(11, 512),                    
                        nn.Linear(512, 1536),
                        nn.BatchNorm1d(1536),
                        nn.Linear(1536, 1024),
                        nn.Dropout(0.2),
                        Mish()          
                        )
        self.fc = nn.Linear(1024, 1)
        
        
        
    def forward(self, x, meta):       
        bs, _, _, _ = x.shape
        x = self.model.features(x)        
        x = F.adaptive_avg_pool2d(x, 1).view(bs, -1)          
        f = self.meta(meta) 
        x = self.l1(x)
        out = x * f        
        out = self.fc(out)       
        return out 