import os
import time
import pandas as pd
PATH = '/home/pka/kaggle/melanoma/input'
PATH_SUB = '/home/pka/kaggle/melanoma/submit'
s1 = 'Sun Aug  9 22_36_48 2020_eff_bz32_lr0.0001_shlReduceLROnPlateau_opAdam_lfFocalLoss_f0_epoch13_score0.876_best_fold_0.892_submit_test2.csv'
s2 = 'Sun Aug  9 18_33_43 2020_res50_bz32_lr0.0001_shlReduceLROnPlateau_opAdam_lfFocalLoss_f4_epoch7_score0.918_best_fold_0.9134_submit_test2.csv'

if __name__ == "__main__": 
    clock = '_'.join(time.ctime().split(':')) 
    subm = pd.read_csv(os.path.join(PATH, 'sample_submission.csv'))   
    subm1 = pd.read_csv(os.path.join(PATH_SUB,  s1))
    subm2 = pd.read_csv(os.path.join(PATH_SUB, s2))

    mix = (subm1.target.values + subm2.target.values) / 2

    subm['target'] = mix
    subm.to_csv(os.path.join(PATH_SUB, f'{clock}_mix.csv'), index=False)
    print('Save.......@#$%^&*(')