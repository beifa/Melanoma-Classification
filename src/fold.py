import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
SEED = 13

if __name__ == "__main__":
    
    PATH = r'C:\Users\pka\kaggle\melanoma\input\siim-isic-melanoma-classification'

    df = pd.read_csv(os.path.join(PATH, 'train.csv'))
    sk = StratifiedKFold(5, shuffle=True, random_state= SEED)
    split_idx = list(sk.split(df.image_name, df.target))
    zeros = np.zeros(len(df))
    df['fold'] = 0
    for i in range(5):
        zeros[split_idx[i][1]] = i
        
    df['fold'] = zeros
    df.to_csv(os.path.join(PATH, 'train_folds.csv'), index = False)
    print(df.fold.value_counts())