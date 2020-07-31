import os
import torch
from torch import nn
import pretrainedmodels
import torch.nn.functional as F
from utils.mish_activation import Mish



import sys
sys.path = [
    'C:\\Users\\pka\\kaggle\\EfficientNet-PyTorch-master',
] + sys.path
from efficientnet_pytorch import model as eff_net

PATH_MODEL = r'C:\Users\pka\kaggle\EfficientNet (Standard Training & Advprop)'

# from os import walk
# f = []
# for (dirpath, dirnames, filenames) in walk('C:\\Users\\pka\\kaggle\\EfficientNet (Standard Training & Advprop)'):
#     f.extend(filenames)
#     break

pre_train = {
    'b0' : 'efficientnet-b0-355c32eb.pth',
    'b1' : 'efficientnet-b1-f1951068.pth',
    'b2' : 'efficientnet-b2-8bb594d6.pth',
    'b3' : 'efficientnet-b3-5fb5a3c3.pth',
    'b4' : 'efficientnet-b4-6ed6700e.pth',
    'b5' : 'efficientnet-b5-b6417697.pth',
    'b6' : 'efficientnet-b6-c76e70fd.pth',
    'b7' : 'efficientnet-b7-dcc49843.pth'
    }

class Res50(nn.Module):
    
    def __init__(self):
        super(Res50, self).__init__()        
        self.model = pretrainedmodels.__dict__['se_resnext50_32x4d'](pretrained = 'imagenet')       
        self.l1 = nn.Linear(2048, 1)
        
        
    def forward(self, x):      
        bs, _, _, _ = x.shape
        x = self.model.features(x)
        x = F.adaptive_avg_pool2d(x, 1).view(bs, -1)
        x = self.l1(x)
        return x


class Eff_b0(nn.Module):
    
    def __init__(self, name, out):
        super(Net, self).__init__()
        self.eff_net = eff_net.EfficientNet.from_name(name)
        self.eff_net.load_state_dict(torch.load(os.path.join(
                                                PATH_MODEL, pre_train[name])))
        self.fc = nn.Linear(self.eff_net._fc.out_features, out)
        
    def current_net(self, x):
        return self.eff_net(x)
    
    def forward(self, x):
        x = self.current_net(x)
        x = self.fc(x)
        return x