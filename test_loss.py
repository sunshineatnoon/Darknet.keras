import numpy as np
import theano.tensor as T
import theano
import os
from PIL import Image
from PIL import ImageDraw

'''
y_pred,y_true = T.dmatrices('y_pred', 'y_true')
scale_vector = []
scale_vector.extend([2]*4)
scale_vector.extend([0]*21)
scale_vector = scale_vector * 2
scale_vector = np.reshape(np.asarray(scale_vector),(1,len(scale_vector)))

#Only calculate cords loss here, so mask all classfication predictions to zeros
y_pred_masked = y_pred * scale_vector
y_true_maksed = y_true * scale_vector
#kick out all cells that don't have any objects
keep = y_true_maksed > 0
y_pred_cal = y_pred_masked * keep
loss = T.sum(T.square(y_true_maksed - y_pred_cal))

#prediction loss
scale_vector = []
scale_vector.extend([0]*4)
scale_vector.extend([1]*21)
scale_vector = scale_vector * 2
scale_vector = np.reshape(np.asarray(scale_vector),(1,len(scale_vector)))
#only calculate classification loss here so kick out all cords information
y_pred_classification = y_pred * scale_vector
y_true_classification = y_true * scale_vector
loss = loss + T.sum(T.square(y_pred_classification - y_true_classification))

f = theano.function([y_pred,y_true], loss)

y_pred = np.asarray([0.5,0.5,0.1,0.2,0.1,0.2,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.6,0.5,0.5,0.1,0.2,0.1,0.2,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.6])
y_true = np.asarray([0.5,0.1,0.2,0.4,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
y_pred = np.expand_dims(y_pred,axis=0)
y_true = np.expand_dims(y_true,axis=0)
print f(y_pred,y_true)
'''
labels = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train","tvmonitor"]
imgPath = os.path.join(os.getcwd(),'images/bike_resize.jpg')
img = Image.open(imgPath)
drawable = ImageDraw.Draw(img)

preds = [0.16017964,0.815,0.89554453,0.79857658,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]
thresh = 0.1
'''
if(preds[24] > thresh):

    row = 3
    col = 4
    centerx = 64 * col + 64 * preds[0]
    centery = 64 * row + 64 * preds[1]

    h = preds[2] * preds[2]
    h = h * 448.0
    w = preds[3] * preds[3]
    w = w * 448.0

    left = centerx - w/2.0
    right = centerx + w/2.0
    up = centery - h/2.0
    down = centery + h/2.0

    if(left < 0): left = 0
    if(right > 448): right = 447
    if(up < 0): up = 0
    if(down > 448): down = 447

    drawable.rectangle([left,up,right,down],outline='red')
    print 'Class is: ',labels[np.argmax(preds[4:24])]
    print np.max(preds[4:24])
    img.save('dog_detection.jpg')
'''
if(preds[24] > thresh):

    row = 3
    col = 4
    centerx = 64 * col + 64 * preds[0]
    centery = 64 * row + 64 * preds[1]

    h = preds[2] * preds[2]
    h = h * 448.0
    w = preds[3] * preds[3]
    w = w * 448.0

    left = centerx - w/2.0
    right = centerx + w/2.0
    up = centery - h/2.0
    down = centery + h/2.0

    if(left < 0): left = 0
    if(right > 448): right = 447
    if(up < 0): up = 0
    if(down > 448): down = 447

    drawable.rectangle([left,up,right,down],outline='red')
    print 'Class is: ',labels[np.argmax(preds[4:24])]
    print np.max(preds[4:24])
    img.save('dog_detection.jpg')
