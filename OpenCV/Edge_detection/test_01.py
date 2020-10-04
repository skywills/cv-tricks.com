import cv2
import numpy as np
#img = cv2.imread("test.png")
def auto_canny(image, sigma=0.5):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
	# return the edged image
	return edged    

img = cv2.imread("D:\My work\MASTERS WORK\SAND - UNIFORM\sand_180pxfor1cm(130,120,75).jpg")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
blurred = cv2.blur(gray, (7,7))
canny = auto_canny(blurred,3)


## find the non-zero min-max coords of canny
pts = np.argwhere(canny>0)
y1,x1 = pts.min(axis=0)
y2,x2 = pts.max(axis=0)

## crop the region
cropped = img[y1:y2, x1:x2]
cv2.imshow("blurred", blurred)
cv2.imshow("canny", canny)
cv2.imshow("cropped", cropped)
tagged = cv2.rectangle(img.copy(), (x1,y1), (x2,y2), (0,255,0), 3, cv2.LINE_AA)
cv2.imshow("tagged", tagged)
cv2.waitKey()
