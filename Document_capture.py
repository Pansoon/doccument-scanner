import cv2 
import os

cap = cv2.VideoCapture(0)

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
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows() 