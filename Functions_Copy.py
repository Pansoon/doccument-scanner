import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

def opencam():

    cap = cv2.VideoCapture(0)

    i = 0

    if not os.path.exists('capture'):
            os.makedirs('capture')

    capture_key = ord('c')
    while True:
        # Read the current frame from the camera
        ret, frame = cap.read()

        # Show the current frame in a window
        cv2.imshow('Camera', frame)

        # Wait for a key to be pressed
        key = cv2.waitKey(1)

        # Check if the capture key was pressed
        if key == capture_key:
            # Save the current frame to a file
            cv2.imwrite('captured_image.jpg', frame)
            print('Image captured!')
            i += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the camera and close all windows
    cap.release()
    cv2.destroyAllWindows()

    return

def resize(img):
     # Resize image to workable size
    dim_limit = 1080
    max_dim = max(img.shape)
    if max_dim > dim_limit:
        resize_scale = dim_limit / max_dim
        img = cv2.resize(img, None, fx=resize_scale, fy=resize_scale)

    # Making Copy of original image.
    orig_img = img.copy()
    return img, orig_img

def blankpage(img):
    # Repeated Closing operation to remove text from the document.
    kernel = np.ones((5,5),np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations= 3)
    mask = np.zeros(img.shape[:2],np.uint8)
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    rect = (20,20,img.shape[1]-20,img.shape[0]-20)
    cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img_r = img*mask2[:,:,np.newaxis]
    return img_r