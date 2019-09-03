import numpy as np
from PIL import Image

img = Image.open('mug.jpg')
w, h = 224, 224
resized_img = img.resize((w, h), Image.BILINEAR)
np_img = np.array(resized_img)
print(np_img.shape)

with open('pybytes.jpg', 'wb') as f:
    bs = np_img.tobytes()
    print(len(bs))
    f.write(bs)
