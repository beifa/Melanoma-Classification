import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, GroupKFold
SEED = 13

"""
we have miss data in anatom_site_general_challenge i fill None and a drop other 68 images no target 1

file_1:
    Fold v1
    make StratifiedKFold for sex, age and tarfet
    and after make group and use GroupKFold
    Fold v2
    add v1 patient_id

    and 5&10 folds split for each version

file_2: 
    skf by 5 slits for target&patient_id


#####
alt:
we have target = 1 == 584 values this is uniques
we take all images by  patient_id and make simple pretrain and after add other
how todo i no idea...:(( is joke

and we get 6927 images for 584 > 0.08430778114623935 10 percent
5 fold stratif or group 
"""

def make_folds(df: pd.DataFrame, sf_count: int, gf_count: int,) -> None:
  features = ['target', 'sex', 'age_approx', 'patient_id']
  for j, f in enumerate(features):
    skf = StratifiedKFold(sf_count, random_state=SEED, shuffle=True)
    split_idx = list(skf.split(df['image_name'], df[f]))
    zeros = np.zeros(len(df))
    df[f'g_{j+1}'] = 0
    for i in range(sf_count):
      zeros[split_idx[i][1]] = i
    df[f'g_{j+1}'] = zeros
  """
  4 sjf folds for diff target
  
  """
  agr_group_v1 = df[['g_1', 'g_2', 'g_3']].sum(axis =1).values
  agr_group_v2 = df[['g_1', 'g_2', 'g_3', 'g_4']].sum(axis =1).values

 
  gf = GroupKFold(gf_count) 
  for g, name in zip([agr_group_v1, agr_group_v2], ['Gfold_v1', 'Gfold_v2']):
    split_idx = list(gf.split(df.image_name, df.target, groups = g))
    zeros = np.zeros(len(df))
    df[name] = 0
    for i in range(gf_count):
      zeros[split_idx[i][1]] = i
    df[name] = zeros
  assert all(df.Gfold_v1.values == df.Gfold_v2 ) == False, 'not be True, something wrong !!!'

if __name__ == "__main__":
    
    PATH = '/home/pka/kaggle/melanoma/input'
    
    #v1.0.0
    
    # df = pd.read_csv(os.path.join(PATH, 'train.csv'))
    # sk = StratifiedKFold(5, shuffle=True, random_state= SEED)
    # split_idx = list(sk.split(df.image_name, df.target))
    # zeros = np.zeros(len(df))
    # df['fold'] = 0
    # for i in range(5):
    #     zeros[split_idx[i][1]] = i
        
    # df['fold'] = zeros
    # df.to_csv(os.path.join(PATH, 'train_folds.csv'), index = False)
    # print(df.fold.value_counts())
    
    #v2.0.0

    df = pd.read_csv(os.path.join(PATH, 'train.csv'))
    df.anatom_site_general_challenge = df.anatom_site_general_challenge.fillna('None')
    df = df.dropna() # 68

    #Make pretrain dataset
    target_names = df[df.target == 1].patient_id.values
    df_topretrain = df[df.patient_id.isin(target_names)]
    df_topretrain.to_csv(os.path.join(PATH, 'pretarin_dataset.csv'), index = False)
    print('#$%^&*(....saved')
    df_topretrain.info()

    #Make folds
    for i in [5, 10]:
        df_tofolds = df.copy()
        make_folds(df_tofolds, i, i)
        df_tofolds = df_tofolds.drop([f'g_{i}' for i in range(1, 5)], axis =1)
        df_tofolds.to_csv(os.path.join(PATH, f'train_folds_{i}split.csv'), index = False)

    print('#$%^&*(....saved')

    #add patient_id
    df_tofolds = df.copy()

    skf = StratifiedKFold(random_state=SEED, shuffle=True)
    split_idx = list(skf.split(df_tofolds['image_name'], df_tofolds['patient_id']))
    zeros = np.zeros(len(df_tofolds))
    df_tofolds[f'stkf_patient'] = 0
    for i in range(5):
        zeros[split_idx[i][1]] = i
    df_tofolds[f'stkf_patient'] = zeros

    #add patient_id
    skf = StratifiedKFold(random_state=SEED, shuffle=True)
    split_idx = list(skf.split(df_tofolds['image_name'], df_tofolds['target']))
    zeros = np.zeros(len(df_tofolds))
    df_tofolds[f'stkf_target'] = 0
    for i in range(5):
        zeros[split_idx[i][1]] = i
    df_tofolds[f'stkf_target'] = zeros


    df_tofolds.to_csv(os.path.join(PATH, 'train_folds_5split_skf.csv'), index = False)
    print('#$%^&*(....saved')



