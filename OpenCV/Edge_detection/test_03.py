import cv2
import argparse
import numpy as np
import skimage
import skimage.feature
from skimage.color import rgb2gray
from skimage.transform import resize, rescale
from boundbox import BoundBox
parser = argparse.ArgumentParser(
        description='This sample shows how to define custom OpenCV deep learning layers in Python. '
                    'Holistically-Nested Edge Detection (https://arxiv.org/abs/1504.06375) neural network '
                    'is used as an example model. Find a pre-trained model at https://github.com/s9xie/hed.')
parser.add_argument('--input', help='Path to image or video. Skip to capture frames from camera',default="")
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
 


cv2.dnn_registerLayer('Crop', CropLayer)

# Load the model.
net = cv2.dnn.readNet(args.prototxt, args.caffemodel)


# load the input image and grab its dimensions
image = cv2.imread(args.input)
# image =cv2.equalizeHist(img)
# image = cv2.pyrMeanShiftFiltering(image1,10,20)
height, width, channel = image.shape

# we use fastNlMeansDenoisingColored to reduce the noise
noise_reduced_image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
H, W = args.height, args.width
noise_reduced_image_resized = cv2.resize(noise_reduced_image, (H, W))

# we keep the original ratio to the image to calculate the bounding box sizes
height_ratio = height/H
width_ratio = width/W



inp = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(args.width, args.height),
                            mean=(104.00698793, 116.66876762, 122.67891434),
                            swapRB=False, crop=False)

net.setInput(inp)
out = net.forward()
out = cv2.resize(out[0, 0], (W, H))
out = (255 * out).astype("uint8")
contours, hierarchy = cv2.findContours(out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

biggest = None
max_area = 0
epsilon = 0.03
for i in contours:
    area = cv2.contourArea(i)
    #we check for contours with area greater than 100 because we don't need very small ones
    if area > 100:
        peri = cv2.arcLength(i, True)
        #here the value of epsilon should be fixed propely to match your enviornment
        approx = cv2.approxPolyDP(i, epsilon*peri, True)
        if area > max_area and len(approx) == 4:
            rectangle = approx
            max_area = area

rect = rectangle.reshape(4, 2)
reshaped_rect = np.zeros((4, 2), dtype="int32")
reshaped_rect[:, 0] = rect[:, 0] * width_ratio
reshaped_rect[:, 1] = rect[:, 1] * height_ratio      
box = BoundBox.box_from_array(reshaped_rect)
transformed_image = box.perspective_wrap(noise_reduced_image)      
cv2.imshow('gray',transformed_image)
cv2.imshow('test',image)
cv2.waitKey(0)