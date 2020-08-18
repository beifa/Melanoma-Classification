from model import Res50, Eff_b_, Res50_meta, Eff_b_meta_v2, Res50meta_v2

"""
'se_resnext50_32x4d'
'se_resnet152'
'senet154'
"""
MODEL_HUB = {
    'res50'        : Res50('se_resnext50_32x4d'),
    'res50_meta'   : Res50_meta(),
    'senet154_meta': Res50meta_v2('senet154'),
    'eff_meta'     : Eff_b_meta_v2(name='efficientnet-b3', out=1),
    'eff4_meta'    : Eff_b_meta_v2(name='efficientnet-b4', out=1),
    'eff'          : Eff_b_('efficientnet-b0', 1),
    'eff2'         : Eff_b_('efficientnet-b2', 1),
    'eff3'         : Eff_b_('efficientnet-b3', 1),
    'eff1'         : Eff_b_('efficientnet-b1', 1),
    'eff4'         : Eff_b_('efficientnet-b4', 1),   
    'senet154'     : Res50('senet154'),
    'se_resnet152' : Res50('se_resnet152')
}