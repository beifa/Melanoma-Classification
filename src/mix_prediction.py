import os
import time
import pandas as pd

PATH_SUB = '/home/pka/kaggle/Melanoma-Classification/submit'
PATH = '/home/pka/kaggle/Melanoma-Classification/input'

s1 = 'Thu Aug 13 17_31_04 2020_eff_bz32_lr0.0001_shlReduceLROnPlateau_opAdam_lfFocalLoss_f4_epoch5_score0.888_best_fold_0.8946_submit_test2.csv'
s2 = 'Thu Aug 13 13_37_38 2020_res50_bz32_lr0.0001_shlReduceLROnPlateau_opAdam_lfFocalLoss_f4_epoch5_score0.909_best_fold_0.9132000000000001_submit_test2.csv'

if __name__ == "__main__": 
    clock = '_'.join(time.ctime().split(':')) 
    subm = pd.read_csv(os.path.join(PATH, 'sample_submission.csv'))   
    subm1 = pd.read_csv(os.path.join(PATH_SUB,  s1))
    subm2 = pd.read_csv(os.path.join(PATH_SUB, s2))

    mix = (subm1.target.values + subm2.target.values) / 2

    subm['target'] = mix
    subm.to_csv(os.path.join(PATH_SUB, f'{clock}_mix.csv'), index=False)
    print('Save.......@#$%^&*(')