import os
import pandas as pd
PATH = r'C:\Users\pka\kaggle\melanoma\input\siim-isic-melanoma-classification'
PATH_SUB = r'C:\Users\pka\kaggle\melanoma\submit'
s1 = r'C:\Users\pka\kaggle\melanoma\submit\Thu Jul 30 19_06_25 2020_res50_bz32_lr0.0001_shlReduceLROnPlateau_opAdam_lfBCEWithLogitsLoss_f0_epoch4_score0.883_best_fold_0.8984.csv'
s2 = r'C:\Users\pka\kaggle\melanoma\submit\Thu Jul 30 20_29_16 2020_eff_bz32_lr0.0001_shlReduceLROnPlateau_opAdam_lfBCEWithLogitsLoss_f4_epoch3_score0.884_best_fold_0.8784000000000001.csv'

if __name__ == "__main__": 
    subm = pd.read_csv(os.path.join(PATH, 'sample_submission.csv'))   
    subm1 = pd.read_csv(s1)
    subm2 = pd.read_csv(s2)

    mix = (subm1.target.values + subm2.target.values) / 2

    subm['target'] = mix
    subm.to_csv(os.path.join(PATH_SUB, f'mix.csv'), index=False)
    print('Save.......@#$%^&*(')