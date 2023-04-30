import cv2
import numpy as np

def opencam():

    cap = cv2.VideoCapture(0)

    i = 0

    capture_key = ord('c')
    while True:
        ret, frame = cap.read()

        cv2.imshow('Camera', frame)

        key = cv2.waitKey(1)
        if key == capture_key:
            # Save the current frame to a file
            cv2.imwrite('captured_image.jpg', frame)
            print('Image captured!')
            i += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

    return

def resize(img):
    dim_limit = 1080
    max_dim = max(img.shape)
    if max_dim > dim_limit:
        resize_scale = dim_limit / max_dim
        img = cv2.resize(img, None, fx=resize_scale, fy=resize_scale)
    orig_img = img.copy()

    return img, orig_img

def blankpage(img):
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

def edgeDetector(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (11, 11), 0)
    canny = cv2.Canny(gray, 100, 200)
    canny = cv2.dilate(canny, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))

    return canny

def contour(img, canny):
    con = np.zeros_like(img)
    contours, hierarchy = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    page = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    con = cv2.drawContours(con, page, -1, (0, 255, 255), 3)

    return page, con

def _order_points(pts):
	rect = np.zeros((4, 2), dtype='float32')
	pts = np.array(pts)
	s = pts.sum(axis=1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	diff = np.diff(pts, axis=1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]

	return rect.astype('int').tolist()

def contourApproximate(img, page):
    con = np.zeros_like(img)
    for c in page:
        epsilon = 0.02 * cv2.arcLength(c, True)
        corners = cv2.approxPolyDP(c, epsilon, True)
        if len(corners) == 4:
            break
    
    cv2.drawContours(con, c, -1, (0, 255, 255), 3)
    cv2.drawContours(con, corners, -1, (0, 255, 0), 10)
    corners = sorted(np.concatenate(corners).tolist())
    for index, c in enumerate(corners):
        character = chr(65 + index)
        cv2.putText(con, character, tuple(c), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 0, 0), 5, cv2.LINE_AA)
    corners = _order_points(corners)

    return corners

def desCoordinates(corners):
    (tl, tr, br, bl) = corners
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    destination_corners =   [
		                    [0, 0],
		                    [maxWidth, 0],
		                    [maxWidth, maxHeight],
		                    [0, maxHeight]
                            ]
    
    return destination_corners, maxWidth, maxHeight

def warp_Per(corners, destination_corners, orig_img, maxWidth, maxHeight):
    homography = cv2.getPerspectiveTransform(np.float32(corners), np.float32(destination_corners))
    final = cv2.warpPerspective(orig_img, np.float32(homography), (maxWidth, maxHeight), flags=cv2.INTER_LINEAR)

    return final