import os
import cv2
import joblib
import pydicom
import pandas as pd
import numpy as np
from tqdm import tqdm
from pydicom.data import get_testdata_file



if __name__ == "__main__":    

    PATH = r'C:\Users\pka\kaggle\melanoma\input\siim-isic-melanoma-classification'
    PATH_DCM = r'C:\Users\pka\kaggle\melanoma\input\siim-isic-melanoma-classification\train'
    PATH_PKL_128 = r'C:\Users\pka\kaggle\melanoma\input\siim-isic-melanoma-classification\pkl_128'

    df = pd.read_csv(os.path.join(PATH, 'train.csv'))

    with tqdm(total = len(df)) as bar:
        for i in range(len(df)):
            name = df.image_name[i]
            image =  os.path.join(PATH_DCM, f'{name}.dcm')
            img = pydicom.dcmread(image)
            new_img = cv2.resize(img.pixel_array, (128, 128))    
            joblib.dump(new_img, os.path.join(PATH_PKL_128, f'{name}.pkl'))
            bar.update(1)