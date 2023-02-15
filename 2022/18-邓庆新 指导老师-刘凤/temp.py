from PIL import Image
import numpy as np

img=Image.open("./visual/00000.png")
img_array=np.array(img)
print(img_array.shape)