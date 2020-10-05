import time
import cv2
import argparse
import numpy as np
import skimage
import skimage.feature
from skimage.color import rgb2gray
from skimage.transform import resize, rescale

parser = argparse.ArgumentParser(
        description='This sample shows how to define custom OpenCV deep learning layers in Python. '
                    'Holistically-Nested Edge Detection (https://arxiv.org/abs/1504.06375) neural network '
                    'is used as an example model. Find a pre-trained model at https://github.com/s9xie/hed.')
parser.add_argument('--input', help='Path to image or video. Skip to capture frames from camera')
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

def edge_detection(image,resize_scale, sigma, l_thresh, h_thresh):
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray,(image.shape[1]//resize_scale,image.shape[0]//resize_scale))
#     rgb = cv2.cvtColor(resized,cv2.COLOR_BGR2RGB)
    #blur = cv2.GaussianBlur(resized, (5, 5),0)
    blur = cv2.blur(resized,(5,5))
    
    edges = skimage.feature.canny(
    image=blur/255.0,
    sigma=sigma,
    low_threshold=l_thresh,
    high_threshold=h_thresh,
    )
    return image, edges

def border_detection(image,edges):
    x = [i for i in range(edges.shape[0]) if np.count_nonzero(edges[i] == True, axis = 0)>0]
    
#     for i in range(0,edges.shape[0]):
#         if (edges[i].any() == True):
#             x.append(i)
    y = [i for i in range(edges.shape[1]) if np.count_nonzero(edges[:,i] == True, axis = 0)>0]
#     for i in range(0,edges.transpose().shape[0]):
#         if (edges.transpose()[i].any() == True):
#             y.append(i)
    if ((len(x)>0) and (len(y)>0)):
        image = image[min(x):max(x),min(y):max(y)]
    
    return image# , min(x),max(x),min(y),max(y)


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
#cv2.imshow('hed', out)
#cv2.waitKey(0)
binary = cv2.threshold(out, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
cnts = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
cnt = None
max_area = x = y = w = h = 0
for c in cnts:
    area = cv2.contourArea(c)
    if area > max_area:
        x, y, w, h = cv2.boundingRect(c)
        max_area = area
        cnt = c

cv2.drawContours(image, [cnt], 0, (0,  255, 0), 3)
cv2.rectangle(image, (x,y), (x+w, y+h), (255, 0, 0), 3)
#cv2.imshow('used ',out)
#cv2.imshow(kWinName,image)
cv2.imshow('orgi', image)
cv2.waitKey(0)

#1.5 sec