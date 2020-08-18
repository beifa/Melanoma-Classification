import os
import time
import pandas as pd
import numpy as np 

PATH_SUB = '/home/pka/kaggle/Melanoma-Classification/submit'
PATH = '/home/pka/kaggle/Melanoma-Classification/input'

s1 = 'Mon Aug 17 20_33_46 2020_0.9036254840777491_final_35159.8_submit_test2.csv'
s2 = 'Sun Aug 16 23_01_20 2020_epoch13_score0.867_best_fold_0.8642_submit_test2.csv'
s3 = 'Sun Aug 16 20_13_38 2020_epoch11_score0.858_best_fold_0.8625999999999999_submit_test2.csv'
s4 = 'Sun Aug 16 19_48_29 2020_f0epoch13_score0.85_0.33695652173913043_best_fold_2608.6_submit_test2.csv'
s5 = 'Sun Aug 16 19_21_09 2020_eff3_bz24_lr0.0001_shlStepLR_opAdam_lfBCEWithLogitsLoss_f4_epoch10_score0.887_best_fold_0.8958_submit_test2.csv'
s6 = 'Sun Aug 16 09_36_46 2020_f5_epoch23_score0.842_0.38016528925619836_best_fold_46855.0_submit_test2.csv'
s7 = 'Thu Aug 13 17_31_04 2020_eff_bz32_lr0.0001_shlReduceLROnPlateau_opAdam_lfFocalLoss_f4_epoch5_score0.888_best_fold_0.8946_submit_test2.csv'
s8 = 'Thu Aug 13 13_37_38 2020_res50_bz32_lr0.0001_shlReduceLROnPlateau_opAdam_lfFocalLoss_f4_epoch5_score0.909_best_fold_0.9132000000000001_submit_test2.csv'




if __name__ == "__main__": 
    clock = '_'.join(time.ctime().split(':')) 
    subm = pd.read_csv(os.path.join(PATH, 'sample_submission.csv'))   
    subm1 = pd.read_csv(os.path.join(PATH_SUB, s1))
    subm2 = pd.read_csv(os.path.join(PATH_SUB, s2))
    subm3 = pd.read_csv(os.path.join(PATH_SUB, s3))
    subm4 = pd.read_csv(os.path.join(PATH_SUB, s4))
    subm5 = pd.read_csv(os.path.join(PATH_SUB, s5))
    subm6 = pd.read_csv(os.path.join(PATH_SUB, s6))
    subm7 = pd.read_csv(os.path.join(PATH_SUB, s7))
    subm8 = pd.read_csv(os.path.join(PATH_SUB, s8))   


    mix = (subm1.target.values + subm2.target.values + subm3.target.values + subm4.target.values + subm5.target.values \
                               + subm6.target.values + subm7.target.values + subm8.target.values) / 8

    subm['target'] = mix
    subm.to_csv(os.path.join(PATH_SUB, f'{clock}_mix.csv'), index=False)
    print('Save.......@#$%^&*(')