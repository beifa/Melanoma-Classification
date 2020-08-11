import os
import pandas as pd
import numpy as np
SEED = 13

if __name__ == "__main__":
    
    PATH = '/home/pka/kaggle/melanoma/input'

    train = pd.read_csv(os.path.join(PATH, 'train_folds.csv'))
    test = pd.read_csv(os.path.join(PATH, 'test.csv'))

    anatom_merge = pd.concat([train.anatom_site_general_challenge, test.anatom_site_general_challenge], axis = 0)
    dummies = pd.get_dummies(anatom_merge, dummy_na=True, dtype=np.uint8)
    train = train.join(dummies.iloc[:train.shape[0]])
    train.columns = list(train.columns[:-1]) + ['wof']
    test = test.join(dummies.iloc[train.shape[0]:])
    test.columns = list(test.columns[:-1]) + ['wof']

    train['sex'] = train['sex'].map({'male': 1,'female': 0})
    test['sex'] = test['sex'].map({'male': 1,'female': 0})

    #train
    patient_id_series = train.groupby('patient_id')['image_name'].count().reset_index()
    patient_id_series.columns = ['patient_id','count_image']
    train = train.merge(patient_id_series, how='left', on = 'patient_id')
    #test
    patient_id_series_test = test.groupby('patient_id')['image_name'].count().reset_index()
    patient_id_series_test.columns = ['patient_id','count_image']
    test = test.merge(patient_id_series_test, how='left', on = 'patient_id')


    #train['age_approx'] /= train['age_approx'].max()
    train['age_approx'] = train['age_approx'].fillna(0)
    train['age_count'] = train.age_approx / train.count_image


    #test['age_approx'] /= test['age_approx'].max()
    test['age_approx'] = test['age_approx'].fillna(0)
    test['age_count'] = test.age_approx / test.count_image

    #train['count_image'] /= train.count_image.max()
    #test['count_image'] /= test.count_image.max()

    train = train.drop([
                'patient_id',
                'anatom_site_general_challenge',
                'diagnosis',
                'benign_malignant'
    ], axis= 1)

    #norm
    # norm_df = train.drop(['target', 'fold', 'image_name'], axis = 1)
    # norm_df =(norm_df - norm_df.min()) / (norm_df.max() - norm_df.min())
    # train = train[['image_name', 'fold', 'target']].join(norm_df)
    from sklearn.preprocessing import Normalizer
    n = Normalizer()
    n = n.fit(train.drop(['target', 'fold', 'image_name'], axis = 1).fillna(0))
    norm = n.transform(train.drop(['target', 'fold', 'image_name'], axis = 1).fillna(0))
    norm_test = n.transform(test.drop(['patient_id', 'anatom_site_general_challenge', 'image_name'], axis = 1).fillna(0))

    norm_train = pd.DataFrame(norm, columns=train.drop(['target', 'fold', 'image_name'], axis = 1).columns)
    norm_test = pd.DataFrame(norm_test, columns=test.drop(['patient_id', 'anatom_site_general_challenge', 'image_name'], axis = 1).columns)

    train = train[['image_name', 'fold', 'target']].join(norm_train)
    test = test[['image_name']].join(norm_test)

    print(train.head())
    print(test.head())

    train.to_csv(os.path.join(PATH, 'train_meta.csv'), index = False)
    test.to_csv(os.path.join(PATH, 'test_meta.csv'), index = False)

