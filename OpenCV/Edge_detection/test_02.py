import cv2
import argparse
import numpy as np
import skimage
import skimage.feature
from skimage.color import rgb2gray
from skimage.transform import resize, rescale
import random as rng
parser = argparse.ArgumentParser(
        description='This sample shows how to define custom OpenCV deep learning layers in Python. '
                    'Holistically-Nested Edge Detection (https://arxiv.org/abs/1504.06375) neural network '
                    'is used as an example model. Find a pre-trained model at https://github.com/s9xie/hed.')
parser.add_argument('--input', help='Path to image or video. Skip to capture frames from camera',default="/home/williamkhoo/Desktop/projects/github/id-card-detector/test_images/1287648340850339842.jpg")
parser.add_argument('--write_video', help='Do you want to write the output video', default=False)
parser.add_argument('--prototxt', help='Path to deploy.prototxt',default='deploy.prototxt', required=False)
parser.add_argument('--caffemodel', help='Path to hed_pretrained_bsds.caffemodel',default='hed_pretrained_bsds.caffemodel', required=False)
parser.add_argument('--width', help='Resize input image to a specific width', default=500, type=int)
parser.add_argument('--height', help='Resize input image to a specific height', default=500, type=int)
parser.add_argument('--savefile', help='Specifies the output video path', default='output.mp4', type=str)
args = parser.parse_args()

class CropLayer(object):
    def __init__(self, params, blobs):
        self.xstart = 0
        self.xend = 0
        self.ystart = 0
        self.yend = 0

    # Our layer receives two inputs. We need to crop the first input blob
    # to match a shape of the second one (keeping batch size and number of channels)
    def getMemoryShapes(self, inputs):
        inputShape, targetShape = inputs[0], inputs[1]
        batchSize, numChannels = inputShape[0], inputShape[1]
        height, width = targetShape[2], targetShape[3]

        self.ystart = (inputShape[2] - targetShape[2]) // 2
        self.xstart = (inputShape[3] - targetShape[3]) // 2
        self.yend = self.ystart + height
        self.xend = self.xstart + width

        return [[batchSize, numChannels, height, width]]

    def forward(self, inputs):
        return [inputs[0][:,:,self.ystart:self.yend,self.xstart:self.xend]]

def thresh_callback(src_gray, val):
    threshold = val
    
    canny_output = cv2.Canny(src_gray, threshold, threshold * 2)
    
    
    contours, _ = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    
    contours_poly = [None]*len(contours)
    boundRect = [None]*len(contours)
    centers = [None]*len(contours)
    radius = [None]*len(contours)
    for i, c in enumerate(contours):
        contours_poly[i] = cv2.approxPolyDP(c, 3, True)
        boundRect[i] = cv2.boundingRect(contours_poly[i])
        centers[i], radius[i] = cv2.minEnclosingCircle(contours_poly[i])
    
    
    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
    
    
    for i in range(len(contours)):
        color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
        cv2.drawContours(drawing, contours_poly, i, color)
        cv2.rectangle(drawing, (int(boundRect[i][0]), int(boundRect[i][1])), \
          (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), color, 2)
        cv2.circle(drawing, (int(centers[i][0]), int(centers[i][1])), int(radius[i]), color, 2)
    
    
    return drawing        

def get_canny_border(image):
    canny = auto_canny(image, 0.1)
    #cv2.imshow('blurred',blurred)
    cv2.imshow('canny',canny)
    cv2.waitKey(0)    
    pts = np.argwhere(canny>0)
    y1,x1 = pts.min(axis=0)
    y2,x2 = pts.max(axis=0)    
    return canny,(x1,y1,x2,y2)
   
    

def auto_canny(image, sigma=0.5):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
	# return the edged image
	return edged    

cv2.dnn_registerLayer('Crop', CropLayer)

# Load the model.
net = cv2.dnn.readNet(args.prototxt, args.caffemodel)


# load the input image and grab its dimensions
image = cv2.imread(args.input)
# image =cv2.equalizeHist(img)
# image = cv2.pyrMeanShiftFiltering(image1,10,20)

## Create a display window
kWinName = 'Holistically-Nested_Edge_Detection'
cv2.namedWindow(kWinName, cv2.WINDOW_AUTOSIZE)

inp = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(args.width, args.height),
                            mean=(104.00698793, 116.66876762, 122.67891434),
                            swapRB=False, crop=False)

net.setInput(inp)
out = net.forward()
out = out[0, 0]
out = cv2.resize(out, (image.shape[1], image.shape[0]))
out = 255 * out
out = out.astype(np.uint8)
#out=cv2.cvtColor(out,cv2.COLOR_GRAY2BGR)
gray, (x1,y1,x2,y2) = get_canny_border(out)
image1 = image[y1:y2, x1:x2]
tagged = cv2.rectangle(image.copy(), (x1,y1), (x2,y2), (0,255,0), 3, cv2.LINE_AA)
#con=np.concatenate((image,gray),axis=1)
cv2.imshow('gray',out)
cv2.imshow('test',tagged)
cv2.imshow(kWinName,image1)

cv2.waitKey(0)