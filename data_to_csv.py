
import numpy as np
import cv2
import os
import pandas as pd
df_size = pd.DataFrame(columns=['label', 'hight', 'width'])
size = 32
labels=[f'p{p}' for p in range(size*size)]
labels.insert(0,'labels')
df_pix = pd.DataFrame(columns=labels)


TRAIN_DIR = 'data/train/'
i = 0
for class_idx, classname in enumerate(os.listdir(TRAIN_DIR)):
    for filename in os.listdir(os.path.join(TRAIN_DIR, classname)):
        img = cv2.imread(os.path.join(TRAIN_DIR, classname,
                                      filename), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_NEAREST)
        img_vec = np.concatenate(img).tolist()
        img_vec.insert(0,class_idx)
        df_pix.loc[i] = img_vec
        df_size.loc[i] = [class_idx, img.shape[0], img.shape[1]]
        i += 1
df_pix.to_csv('train_data.csv', encoding='utf-8')
