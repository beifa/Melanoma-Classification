from model import Res50, Eff_b_, Res50_meta


MODEL_HUB = {
    'res50'      : Res50(),
    'res50_meta' : Res50_meta(),
    'eff'        : Eff_b_('efficientnet-b0', 1),
    'eff2'       : Eff_b_('efficientnet-b2', 1),
    'eff3'       : Eff_b_('efficientnet-b3', 1),
    'eff1'       : Eff_b_('efficientnet-b1', 1),
    'eff4'       : Eff_b_('efficientnet-b4', 1),
    'eff5'       : Eff_b_('efficientnet-b5', 1),
    'eff6'       : Eff_b_('efficientnet-b6', 1),
    'eff7'       : Eff_b_('efficientnet-b7', 1)   
}