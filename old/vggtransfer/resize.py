import os.path
from glob import glob

from PIL import Image

data_real = glob(os.path.join("faces", '*.jpg'))[0:1000]
for i in data_real:
    name = i[6:]
    name = name[:-4]
    print(i)
    print(name)
    img = Image.open(i)
    out = img.resize((192,192),Image.ANTIALIAS)
    out.save("./resize" + name, 'png')
    print(name+"  success resized!")
