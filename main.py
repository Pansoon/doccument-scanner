import cv2
import os
import numpy as np

from Functions_Copy import *

i = 1

opencam()
img = cv2.imread(r"C:\Users\tcsuy\Documents\GitHub\doccument-scanner\captured_image.jpg", cv2.IMREAD_ANYCOLOR)

img, original = resize(img)
blank = blankpage(img)
canny = edgeDetector(blank)
page, con = contour(img, canny)
corners = contourApproximate(img, page)
desCoor, maxWidth, maxHeight = desCoordinates(corners)
result = warp_Per(corners, desCoor, original, maxWidth, maxHeight)

plt.figure(figsize = (10,7))
plt.imshow(result)
plt.axis("off")
while True:
    path = (r"C:\Users\tcsuy\Documents\GitHub\doccument-scanner\result_%d.png") %i
    if os.path.isfile(path) is True:
        i += 1
    else:
        cv2.imwrite("result_%d.png" %i, result)
        break

plt.title("Scanned picture")
plt.show()