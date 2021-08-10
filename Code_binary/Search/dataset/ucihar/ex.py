import csv
import os
import numpy as np
import pandas as pd

os.getcwd()
# UCI-HAR
data = np.load('./Code/Search/dataset/ucihar/train_x.npy')
data.shape
data = data.astype(np.float32)
data = data[:1000].reshape(-1, 128)

f = open('./Code/Search/dataset/ucihar/ucihar.csv', 'w', newline='')
wr = csv.writer(f)
count = 0
for d in data:
    wr.writerow(d)
    count += 1
    if count == 1000:
        break
f.close


df = pd.read_csv('./Code/Search/dataset/ucihar/ucihar.csv')
df