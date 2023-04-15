import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from Functions_Copy import *

#opencam()
img = cv2.imread(r"C:\Users\tcsuy\Documents\GitHub\doccument-scanner\captured_image.jpg")
resize(img)
blankpage(img)

plt.figure(figsize = (10,7))
plt.imshow(img[:,:,::-1])
plt.show()