import numpy as np

from glob import glob
import os
import time

data_z = glob(os.path.join("./gdata", 'relu5_1', '*.npy'))
data = np.load(data_z[0])
print (data.shape)
print(data)
