import cv2
from PIL import Image
import numpy as np
for t in range(26):
    a=Image.open(str(t)+".jpg").convert('L')
    img = a.resize((28,28))
    im=np.array(img)
    im=im.reshape(1,28,28,1)
