import cv2
import os
import numpy as np
from Functions_Copy import *

# opencam()
img = cv2.imread(r"C:\Users\tcsuy\Documents\GitHub\doccument-scanner\captured_image1.jpg", cv2.IMREAD_ANYCOLOR)

img, original = resize(img)
blank = blankpage(img)
canny = edgeDetector(blank)
page, con = contour(img, canny)
corners = contourApproximate(img, page)
desCoor, maxWidth, maxHeight = desCoordinates(corners)
result = warp_Per(corners, desCoor, original, maxWidth, maxHeight)

# show_img(result)

plt.figure(figsize = (10,7))
plt.imshow(result)
plt.title("Document")
plt.show()

'''
plt.figure(figsize=[20,10]); 
plt.subplot(121); plt.imshow(original[:,:,::-1]); plt.axis('off'); plt.title("Original image")
plt.subplot(122); plt.imshow(result[:,:,::-1]); plt.axis('off'); plt.title("Scanned Form");
'''