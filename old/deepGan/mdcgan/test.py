import os
from glob import glob

import numpy as np

data_z = glob(os.path.join("./gdata", 'relu5_1', '*.npy'))
data = np.load(data_z[0])
print (data.shape)
print(data)
